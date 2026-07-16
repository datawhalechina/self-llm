#!/usr/bin/env python
"""LoRA SFT for MiniMax-M3 BF16 using DeepSpeed ZeRO-3 CPU parameter offload."""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.integrations import HfDeepSpeedConfig


@dataclass
class Args:
    model: str
    data: str
    output_dir: str
    deepspeed: str
    max_length: int = 256
    max_steps: int = -1
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    project: str = "minimax-m3-prep"
    experiment: str = "minimax-m3-bf16-zero3-lora"


class SwanLabLogCallback(TrainerCallback):
    def __init__(self, swanlab_module):
        self.swanlab = swanlab_module

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        clean_logs = {
            key: value
            for key, value in logs.items()
            if isinstance(value, (int, float, str, bool))
        }
        if clean_logs:
            self.swanlab.log(clean_logs, step=state.global_step)


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/MiniMax-M3-BF16")
    parser.add_argument("--data", default="/root/autodl-fs/experiments/minimax-m3-8gpu/data/tiny_qa.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--project", default=os.environ.get("FT_SWANLAB_PROJECT", "minimax-m3-prep"))
    parser.add_argument("--experiment", default=os.environ.get("FT_SWANLAB_EXPERIMENT", "minimax-m3-bf16-zero3-lora"))
    ns = parser.parse_args()
    return Args(**vars(ns))


def init_swanlab(args: Args):
    os.environ.pop("SWANLAB_PROJECT", None)
    os.environ.pop("SWANLAB_EXPERIMENT", None)
    if os.environ.get("SWANLAB_WORKSPACE") == "":
        os.environ.pop("SWANLAB_WORKSPACE", None)

    import swanlab

    api_key = os.environ.get("SWANLAB_API_KEY")
    if api_key:
        try:
            swanlab.login(api_key=api_key)
        except TypeError:
            swanlab.login(api_key)
    swanlab.init(
        project=args.project,
        experiment_name=args.experiment,
        config=asdict(args),
    )
    return swanlab


def format_example(example, tokenizer):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            thinking_mode="disabled",
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Keep this object alive for the whole model load: it enables ZeRO-3 sharded
    # initialization instead of briefly materializing the complete BF16 model.
    ds_config = HfDeepSpeedConfig(args.deepspeed)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    if config.model_type != "minimax_m3_vl":
        raise ValueError(f"Expected minimax_m3_vl, found {config.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.data, split="train")

    def tokenize(example):
        encoded = tokenizer(
            format_example(example, tokenizer),
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        deepspeed=args.deepspeed,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="steps" if args.max_steps > 0 else "no",
        save_steps=max(args.max_steps, 1),
        save_total_limit=1,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        report_to=[],
        remove_unused_columns=False,
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    # Apply LoRA only to MiniMax-M3 language attention; keep all MoE experts frozen.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            r"^model\.language_model\.layers\.\d+\.self_attn\."
            r"(q_proj|k_proj|v_proj|o_proj)$"
        ),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    swanlab = init_swanlab(args) if int(os.environ.get("RANK", "0")) == 0 else None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[SwanLabLogCallback(swanlab)] if swanlab else [],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero:
        tokenizer.save_pretrained(args.output_dir)
        Path(args.output_dir, "run_metadata.json").write_text(
            json.dumps(asdict(args), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if swanlab:
        swanlab.finish(async_log_timeout=30)


if __name__ == "__main__":
    main()
