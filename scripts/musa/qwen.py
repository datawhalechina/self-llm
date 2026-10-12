"""Single-GPU Qwen inference / LoRA tutorial using local model and data files."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["infer", "lora"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="请介绍一下你自己。")
    parser.add_argument("--data-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["float32", "bf16"], default="bf16")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be a local model directory")
    if args.action == "lora":
        if args.data_json is None or not args.data_json.is_file() or args.output_dir is None:
            parser.error("lora requires an existing --data-json and --output-dir")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error("--output-dir must be empty")
    if args.max_steps < 1 or args.max_length < 8 or args.max_new_tokens < 1:
        parser.error("invalid step count or sequence length")

    import torchada  # Load through run.py so worker processes inherit this too.
    import torch_musa
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Trainer, TrainingArguments,
    )

    if not torch.musa.is_available() or not torch.cuda.is_available():
        raise RuntimeError("Start with scripts/musa/run.py inside the leased MUSA container")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), local_files_only=True,
        torch_dtype=dtype, attn_implementation="eager",
    ).to("cuda:0")

    def check_device():
        if any(p.device.type != "musa" or p.device.index != 0 for p in model.parameters()):
            raise RuntimeError("Model parameters did not land on MUSA device 0")

    def prompt_text(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    check_device()
    if args.action == "infer":
        model.eval()
        inputs = tokenizer(prompt_text(args.prompt), return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        if output.device.type != "musa":
            raise RuntimeError("Generation output did not land on MUSA")
        print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
        print(json.dumps({"action": "infer", "device": str(output.device), "status": "PASS"}))
        return

    from peft import LoraConfig, TaskType, get_peft_model

    records = json.loads(args.data_json.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Data must be a nonempty JSON array of instruction/input/output records")
    examples = []
    for record in records:
        prefix = tokenizer(prompt_text(record["instruction"] + record.get("input", "")), add_special_tokens=False)["input_ids"]
        response = tokenizer(record["output"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        ids = (prefix + response)[:args.max_length]
        labels = ([-100] * len(prefix) + response)[:args.max_length]
        if not any(label != -100 for label in labels):
            raise ValueError("Prompt fills max-length; increase --max-length to retain response tokens")
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"],
        r=8, lora_alpha=16, lora_dropout=0.0,
    ))
    model.enable_input_require_grads()
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=1, max_steps=args.max_steps,
        learning_rate=1e-4, logging_steps=1, save_strategy="no",
        bf16=args.dtype == "bf16", fp16=False, gradient_checkpointing=True,
        report_to="none", dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=examples,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    result = trainer.train()
    check_device()
    if not torch.isfinite(torch.tensor(result.training_loss)).item():
        raise RuntimeError("Nonfinite training loss")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    if not (args.output_dir / "adapter_config.json").is_file():
        raise RuntimeError("Adapter configuration was not saved")
    print(json.dumps({"action": "lora", "status": "PASS", "device": str(next(model.parameters()).device),
                      "steps": result.global_step, "loss": result.training_loss,
                      "adapter_directory": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
