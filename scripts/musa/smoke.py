#!/usr/bin/env python3
"""Offline CUDA-to-MUSA smoke test for torchada, Transformers, and PEFT.

Intended image: registry.mthreads.com/mcctest/training-suite:v2.1.7
The last stdout line is a JSON result; a failure also exits nonzero.
"""

import argparse
import importlib.metadata
import json
import sys
import tempfile
import traceback
from pathlib import Path


def version(module, distribution):
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(args, result):
    # The adapter must patch CUDA APIs before any framework imports.
    import torchada
    import torch_musa
    import torch
    import transformers
    import peft
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    result["versions"] = {
        "python": sys.version.split()[0],
        "torchada": version(torchada, "torchada"),
        "torch_musa": version(torch_musa, "torch_musa"),
        "torch": version(torch, "torch"),
        "transformers": version(transformers, "transformers"),
        "peft": version(peft, "peft"),
    }
    assert hasattr(torch, "musa"), "torch_musa did not register torch.musa"
    assert torch.musa.is_available(), "MUSA is unavailable; CPU fallback is forbidden"
    assert torch.cuda.is_available(), "torchada did not expose CUDA compatibility"
    torch.cuda.set_device(args.device)
    result["device"] = {
        "requested": f"cuda:{args.device}",
        "name": torch.musa.get_device_name(args.device),
    }
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    result["dtype"] = args.dtype

    def musa_tensor(tensor, label):
        assert tensor.device.type == "musa", (
            f"{label} is on {tensor.device}; expected a physical MUSA tensor"
        )
        assert tensor.device.index == args.device, (
            f"{label} is on device {tensor.device.index}; expected {args.device}"
        )

    def musa_model(model, label):
        for name, parameter in model.named_parameters():
            musa_tensor(parameter, f"{label}.{name}")
        for name, buffer in model.named_buffers():
            musa_tensor(buffer, f"{label} buffer {name}")

    def finite(tensor, label):
        musa_tensor(tensor, label)
        assert torch.isfinite(tensor).all().item(), f"{label} contains nonfinite values"

    torch.manual_seed(26)
    # Small random model: no tokenizer, Hub access, or pretrained weights.
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    device = f"cuda:{args.device}"
    input_ids = torch.randint(3, config.vocab_size, (2, 8), device=device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    labels = input_ids.clone()
    for label, tensor in (
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
        ("labels", labels),
    ):
        musa_tensor(tensor, label)
    result["device"]["physical_type"] = input_ids.device.type
    result["device"]["physical_index"] = input_ids.device.index

    model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype)
    musa_model(model, "qwen2")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    finite(output.logits, "base logits")
    finite(output.loss, "base loss")
    output.loss.backward()
    gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            finite(parameter.grad, f"base gradient {name}")
            gradient_count += 1
    assert gradient_count > 0, "Base backward produced no gradients"
    optimizer.step()
    torch.cuda.synchronize()
    musa_model(model, "trained qwen2")
    result["base_loss"] = output.loss.detach().item()
    result["base_gradient_tensors"] = gradient_count
    del optimizer, output
    model.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids[:1],
            attention_mask=attention_mask[:1],
            max_new_tokens=4,
            min_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=config.pad_token_id,
        )
    musa_tensor(generated, "greedy generation")
    assert generated.shape == (1, 12), f"Unexpected generation shape: {generated.shape}"
    assert torch.equal(generated[:, :8], input_ids[:1]), "Generation changed its prompt"
    result["generated_token_ids"] = generated[0].tolist()

    # Reload the adapter onto identical base weights, rather than a new random base.
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    lora_model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
        ),
    )
    musa_model(lora_model, "lora qwen2")
    lora_model.train()
    trainable = {name: p for name, p in lora_model.named_parameters() if p.requires_grad}
    assert trainable, "LoRA has no trainable parameters"
    assert all("lora_" in name for name in trainable), "Unexpected trainable base weights"
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    lora_optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3)
    lora_optimizer.zero_grad(set_to_none=True)
    lora_output = lora_model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    finite(lora_output.loss, "lora loss")
    lora_output.loss.backward()
    nonzero_gradient = False
    for name, parameter in trainable.items():
        assert parameter.grad is not None, f"Missing LoRA gradient: {name}"
        finite(parameter.grad, f"lora gradient {name}")
        nonzero_gradient |= bool(torch.count_nonzero(parameter.grad).item())
    assert nonzero_gradient, "LoRA backward produced only zero gradients"
    lora_optimizer.step()
    torch.cuda.synchronize()
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in trainable.items()
    ), "LoRA optimizer step did not change adapter weights"
    result["lora_loss"] = lora_output.loss.detach().item()
    result["lora_trainable_parameters"] = sum(p.numel() for p in trainable.values())
    musa_model(lora_model, "trained lora qwen2")
    lora_model.eval()
    with torch.no_grad():
        reference_logits = lora_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.detach().clone()
    finite(reference_logits, "saved adapter reference logits")

    def save_and_reload(directory):
        lora_model.save_pretrained(directory)
        assert (directory / "adapter_config.json").is_file(), "Adapter config was not saved"
        assert any(
            (directory / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        ), "Adapter weights were not saved"
        reload_base = Qwen2ForCausalLM(config).to(device=device, dtype=dtype)
        reload_base.load_state_dict(base_state, strict=True)
        reloaded = PeftModel.from_pretrained(reload_base, str(directory), is_trainable=False)
        musa_model(reloaded, "reloaded lora qwen2")
        reloaded_parameters = dict(reloaded.named_parameters())
        for name, parameter in trainable.items():
            assert name in reloaded_parameters, f"Reloaded adapter is missing {name}"
            assert torch.equal(parameter.detach(), reloaded_parameters[name].detach()), (
                f"Saved/reloaded adapter weight mismatch: {name}"
            )
        reloaded.eval()
        with torch.no_grad():
            reload_logits = reloaded(input_ids=input_ids, attention_mask=attention_mask).logits
        finite(reload_logits, "reloaded adapter logits")
        tolerance = 1e-5 if args.dtype == "float32" else 1e-2
        assert torch.allclose(reference_logits, reload_logits, rtol=tolerance, atol=tolerance), (
            "Saved/reloaded adapter changed model logits"
        )
        torch.cuda.synchronize()

    if args.output_dir is not None:
        directory = args.output_dir.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        assert not any(directory.iterdir()), "--output-dir must be empty to avoid overwriting adapters"
        save_and_reload(directory)
        result["adapter_directory"] = str(directory)
    else:
        with tempfile.TemporaryDirectory(prefix="torchada-musa-lora-") as temporary:
            save_and_reload(Path(temporary))
        result["adapter_directory"] = "temporary (removed after successful reload)"
    result["checks"] = {
        "physical_musa": True,
        "cuda_tensor_api": True,
        "base_forward_backward_optimizer": True,
        "greedy_generation": True,
        "lora_training": True,
        "adapter_save_reload_weights_and_logits": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="MUSA device index through CUDA APIs")
    parser.add_argument("--dtype", choices=("float32", "bf16"), default="float32")
    parser.add_argument("--output-dir", type=Path, help="Optional empty directory to retain the adapter")
    args = parser.parse_args()
    result = {"test": "torchada_musa_qwen2_peft_smoke", "status": "FAIL"}
    try:
        if not __debug__:
            raise RuntimeError("Run without Python -O: physical-device assertions must stay enabled")
        assert args.device >= 0, "--device must be nonnegative"
        run(args, result)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc(file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1
    result["status"] = "PASS"
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
