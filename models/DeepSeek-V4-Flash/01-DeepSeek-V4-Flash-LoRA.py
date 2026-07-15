#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


SELF_LLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = SELF_LLM_ROOT / "dataset" / "huanhuan.jsonl"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/RedHatAI/DeepSeek-V4-Flash-BF16"
DEFAULT_OUTPUT_DIR = "./output/DeepSeek-V4-Flash-LoRA"
DEFAULT_SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人--甄嬛。"
# o_a_proj is a grouped linear layer. Injecting it as a regular Linear layer
# can cause a grouped-shape mismatch during the forward pass.
DEFAULT_LORA_TARGETS = "q_a_proj,q_b_proj,kv_proj,o_b_proj"


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash LoRA fine-tuning")
    parser.add_argument("--model-path", default=os.getenv("DS_V4_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--data-path", default=os.getenv("DS_V4_DATA_PATH", str(DEFAULT_DATA_PATH)))
    parser.add_argument("--output-dir", default=os.getenv("DS_V4_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--num-proc", type=int)
    parser.add_argument("--swanlab-project", default="")
    parser.add_argument("--swanlab-experiment", default="huanhuan-r16-a32-200step")
    return parser.parse_args()


def parse_targets(value):
    return [target.strip() for target in value.split(",") if target.strip()]


def encode_chat_text(system_prompt, user_content, assistant_content=None):
    text = (
        "<｜begin▁of▁sentence｜>"
        f"{system_prompt}"
        "<｜User｜>"
        f"{user_content}"
        "<｜Assistant｜></think>"
    )
    if assistant_content is not None:
        text += f"{assistant_content}<｜end▁of▁sentence｜>"
    return text


def build_process_func(tokenizer, max_length, system_prompt):
    def process_func(example):
        user_content = str(example.get("instruction") or "") + str(example.get("input") or "")
        output = str(example.get("output") or "")

        prompt_text = encode_chat_text(system_prompt, user_content)
        full_text = encode_chat_text(system_prompt, user_content, output)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=False)

        input_ids = full["input_ids"]
        attention_mask = full.get("attention_mask", [1] * len(input_ids))
        prompt_length = len(prompt_ids)
        labels = [-100] * prompt_length + input_ids[prompt_length:]

        if tokenizer.eos_token_id is not None and (
            not input_ids or input_ids[-1] != tokenizer.eos_token_id
        ):
            input_ids.append(tokenizer.eos_token_id)
            attention_mask.append(1)
            labels.append(tokenizer.eos_token_id)

        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return process_func


def build_swanlab_callback(args):
    if not args.swanlab_project:
        return None

    from swanlab.integration.transformers import SwanLabCallback

    return SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
    )


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    data_path = Path(args.data_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data path not found: {data_path}")

    gpu_count = torch.cuda.device_count()
    print(f"visible GPUs: {gpu_count}")
    if gpu_count != args.expected_gpus:
        print(f"warning: expected {args.expected_gpus} GPUs, got {gpu_count}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    map_kwargs = {"num_proc": args.num_proc} if args.num_proc else {}
    tokenized_dataset = dataset.map(
        build_process_func(tokenizer, args.max_length, args.system_prompt),
        remove_columns=dataset.column_names,
        **map_kwargs,
    )

    device_map = None if args.device_map.lower() == "none" else args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=parse_targets(args.lora_targets),
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to=[],
        save_on_each_node=True,
    )

    callbacks = []
    swanlab_callback = build_swanlab_callback(args)
    if swanlab_callback is not None:
        callbacks.append(swanlab_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
