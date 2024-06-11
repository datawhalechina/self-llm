import glob
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from transformers import AutoModel, AutoTokenizer
from transformers.integrations import deepspeed
from transformers import AutoModel, AutoTokenizer

from vlm.modeling_minicpmv import MiniCPMV
from vlm.modeling_minicpmv import LlamaTokenizerWrapper

from dataset import SupervisedDataset, data_collator
from trainer import CPMTrainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    llm_type: str = field(default="minicpm")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None
   
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    transform,
    data_collator=None,
    llm_type="minicpm",
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            transform,
            tokenizer,
            slice_config=slice_config,
            llm_type=llm_type,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= partial(data_collator, max_length=max_length),
    )


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Trainable': trainable_params}


local_rank = 0


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) : 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
    
    model = MiniCPMV.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizerWrapper.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)
        
    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
            
        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        model.base_model.resampler.requires_grad_(True)
        model.base_model.llm.model.embed_tokens.weight.requires_grad_(True)
        if training_args.tune_vision:
            model.base_model.vpm.requires_grad_(True)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))

    llm_type = training_args.llm_type    
    
    rank0_print(f'llm_type={llm_type}')

    
    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        transform=model.transform,
        data_collator=data_collator,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=model.config.patch_size,
        query_nums=model.config.query_num,
        batch_vision=batch_vision,
        max_length=training_args.model_max_length,
    )
    
    trainer = CPMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
