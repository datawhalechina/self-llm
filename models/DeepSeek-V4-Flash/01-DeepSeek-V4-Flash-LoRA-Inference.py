#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch
from peft.functional import cast_adapter_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "/root/autodl-tmp/deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_ADAPTER_PATH = "./output/DeepSeek-V4-Flash-LoRA"
DEFAULT_SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人--甄嬛。"


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash LoRA inference")
    parser.add_argument("--model-path", default=os.getenv("DS_V4_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--adapter-path",
        default=os.getenv("DS_V4_ADAPTER_PATH", DEFAULT_ADAPTER_PATH),
    )
    parser.add_argument("--prompt", default="你是谁？")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def encode_chat_text(system_prompt, user_content):
    return (
        "<｜begin▁of▁sentence｜>"
        f"{system_prompt}"
        "<｜User｜>"
        f"{user_content}"
        "<｜Assistant｜></think>"
    )


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    adapter_path = Path(args.adapter_path).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter path not found: {adapter_path}")

    tokenizer_path = adapter_path if (adapter_path / "tokenizer_config.json").is_file() else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_map = None if args.device_map.lower() == "none" else args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    load_info = model.load_adapter(
        str(adapter_path),
        adapter_name="default",
        low_cpu_mem_usage=True,
    )
    if load_info.missing_keys or load_info.unexpected_keys or load_info.mismatched_keys:
        raise RuntimeError(
            "LoRA adapter load produced incompatible keys: "
            f"missing={len(load_info.missing_keys)}, "
            f"unexpected={len(load_info.unexpected_keys)}, "
            f"mismatched={len(load_info.mismatched_keys)}"
        )
    cast_adapter_dtype(model, adapter_name="default")
    model.set_adapter("default")
    model.config.use_cache = True
    model.eval()

    text = encode_chat_text(args.system_prompt, args.prompt)
    inputs = tokenizer(text, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][inputs["input_ids"].shape[1] :]
    generate_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    generate_text = generate_text.split("<｜end▁of▁sentence｜>", 1)[0].strip()
    print(generate_text)


if __name__ == "__main__":
    main()
