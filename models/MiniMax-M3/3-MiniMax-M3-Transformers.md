# MiniMax-M3 Transformers 部署调用

## MiniMax-M3 简介

[MiniMax-M3](https://github.com/MiniMax-AI/MiniMax-M3) 是 MiniMax 推出的最新一代开源大语言模型。相较于 M2.5，M3 在长上下文、推理质量和多模态能力上均有明显提升：上下文窗口扩展到 **512K**、单次最大输出可达 **128K**，并原生支持**图片输入**（仅图片，视频/音频/文档暂不支持）。本文演示如何通过 Transformers 直接加载并运行 MiniMax-M3 模型。

## 环境准备

基础环境（参考值）：

- OS：Linux
- Python：3.9 - 3.12
- Transformers：>= 4.57.1
- GPU：compute capability 7.0 or higher，显存需求约 220 GB

安装依赖：

> 建议使用虚拟环境（venv / conda / uv）避免依赖冲突

```bash
uv venv
source .venv/bin/activate
uv pip install transformers==4.57.1 torch accelerate --torch-backend=auto
```

## 模型下载

Transformers 会在首次加载时自动从 Hugging Face 下载并缓存模型。若网络受限，可使用 `modelscope` 提前下载：

```python
# model_download.py
from modelscope import snapshot_download

model_dir = snapshot_download('MiniMaxAI/MiniMax-M3', cache_dir='/root/autodl-tmp', revision='master')
print(f"模型下载成功，保存到: {model_dir}")
```

```bash
pip install modelscope
python model_download.py
```

> 注意：使用 `modelscope` 下载模型时无需设置 HF 镜像。若不使用 modelscope，而是直接通过 Transformers 从 Hugging Face 下载，网络受限时可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

## 推理示例

### 基础对话

```python
# test_transformers.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "MiniMaxAI/MiniMax-M3"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

messages = [
    {"role": "user", "content": [{"type": "text", "text": "请介绍一下 MiniMax-M3 模型的主要特点。"}]},
]

model_inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to("cuda")

generated_ids = model.generate(
    model_inputs,
    max_new_tokens=2048,
    generation_config=model.generation_config,
)

response = tokenizer.batch_decode(generated_ids)[0]
print(response)
```

运行：

```bash
python test_transformers.py
```

### 多轮对话

```python
# test_multi_turn.py
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "MiniMaxAI/MiniMax-M3"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

messages = [
    {"role": "user", "content": [{"type": "text", "text": "什么是快速排序？"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "快速排序是一种高效的排序算法，采用分治策略，通过选择一个基准元素将数组分为两部分，然后递归排序。"}]},
    {"role": "user", "content": [{"type": "text", "text": "请用 Python 实现它。"}]},
]

model_inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to("cuda")

generated_ids = model.generate(
    model_inputs,
    max_new_tokens=2048,
    generation_config=model.generation_config,
)

response = tokenizer.batch_decode(generated_ids)[0]
print(response)
```

运行：

```bash
python test_multi_turn.py
```

### 长输出（≤128K）

M3 支持单次最大 128K 的输出长度，按需调整 `max_new_tokens` 即可：

```python
generated_ids = model.generate(
    model_inputs,
    max_new_tokens=131072,   # 128K
    generation_config=model.generation_config,
)
```

> 输出越长，显存与时延越高，建议结合任务实际需求设置。

## 常见问题

### Hugging Face 网络问题

若未使用 `modelscope` 下载模型，而是直接通过 Transformers 从 Hugging Face 下载，可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> 若已通过 `modelscope` 下载模型并指定了本地路径，则无需此设置。

### MiniMax-M3 model is not currently supported

请确认已开启 `trust_remote_code=True`，并升级 Transformers 至 >= 4.57.1：

```bash
pip install -U transformers
```

## 参考链接

- [MiniMax-M3 GitHub](https://github.com/MiniMax-AI/MiniMax-M3)
- [MiniMax-M3 HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M3)
- [MiniMax 平台文档](https://platform.minimax.io/docs/guides/text-generation)
