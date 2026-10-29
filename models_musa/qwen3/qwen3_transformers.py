#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-4B 在摩尔线程 MUSA 上的 transformers 推理示例

基础镜像：registry.mthreads.com/mcconline/training-suite:v2.1.7
环境：torch 2.7.1.post1 + torch_musa 2.7.1.post1 / transformers 4.57.0+musa.p34

运行：python3 qwen3_transformers.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_PATH = "Qwen/Qwen3-4B"              # 相对执行目录，按需修改

print("=" * 70)
print(f"torch {torch.__version__} | torch_musa 可见设备数 {torch.musa.device_count()}")
print(f"当前设备 {torch.musa.get_device_name(0)}")
print("=" * 70)

# ---------------- 1. 加载 tokenizer 与模型 ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()
print(f"[模型设备] {model.device} | dtype {next(model.parameters()).dtype}")

# ---------------- 2. 非思考模式（enable_thinking=False） ----------------
messages = [{"role": "user", "content": "用一句话介绍你自己。"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)

print("\n" + "=" * 70)
print("[非思考模式] 回答：")
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# ---------------- 3. 思考模式（enable_thinking=True）+ 流式输出 ----------------
messages = [{"role": "user", "content": "9.11 和 9.8 哪个大？"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n" + "=" * 70)
print("[思考模式] 流式输出：")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
with torch.no_grad():
    model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        streamer=streamer,
    )
