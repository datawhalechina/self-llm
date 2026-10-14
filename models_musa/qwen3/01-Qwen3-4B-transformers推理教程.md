# Qwen3-4B transformers 推理教程

本文介绍如何在**摩尔线程 MUSA** 上，用 `transformers` 原生 API 推理 `Qwen3-4B`。

相比 vLLM、SGLang 这类推理引擎，`transformers` 直接加载模型、`model.generate()` 出结果，
最适合用来**验证环境是否正常**、以及**调试 prompt 与对话模板**。

## 基础环境

```text
----------------
镜像      registry.mthreads.com/mcconline/training-suite:v2.1.7
操作系统  Ubuntu 22.04.5 LTS
GPU       MTT S5000 (80 GB) × 8
驱动      3.3.5-server
MUSA SDK  4.3.8
Python    3.10.12
torch     2.7.1.post1 + torch_musa 2.7.1.post1
transformers 4.57.0+musa.p34
----------------
```

> 本文基于 `training-suite:v2.1.7` 镜像，容器启动方式见 [models_musa/README.md](../README.md)。
> `transformers` 从 4.51 起原生支持 Qwen3，本镜像自带的 4.57.0 可直接使用，**无需额外安装**。

## 模型准备

Qwen3-4B 权重约 8 GB，推荐用 `modelscope` 下载：

```python
# model_download.py
from modelscope import snapshot_download

# cache_dir='.'：下载到当前目录，得到 ./Qwen/Qwen3-4B
model_dir = snapshot_download('Qwen/Qwen3-4B', cache_dir='.')
print(f"模型下载完成，保存路径为：{model_dir}")
```

```shell
python model_download.py
```

本文后续统一使用**相对路径** `Qwen/Qwen3-4B`（相对**执行目录**），请按自己的实际权重位置修改。
（同样代码可直接跑 Qwen3-0.6B / Qwen3-32B / Qwen3-30B-A3B，只需替换路径。）

## 一、最小可运行示例

新建 `qwen3_transformers.py`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen3-4B"   # 相对执行目录

# 1. 加载 tokenizer 与模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",          # 自动放置到可用的 MUSA 设备
    dtype=torch.bfloat16,       # 半精度，显著降低显存占用
    trust_remote_code=True,
)
model.eval()

# 2. 套用 Qwen3 的对话模板
messages = [{"role": "user", "content": "用一句话介绍你自己。"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,      # 关闭思考模式（Qwen3 是混合推理模型）
)

# 3. 生成
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)

# 4. 解码（去掉输入部分，只保留新生成的 token）
new_tokens = generated[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

```shell
python qwen3_transformers.py
```

运行输出（节选）：

```text
torch 2.7.1.post1 | torch_musa 可见设备数 8
当前设备 MTT S5000
Loading checkpoint shards: 100%|██████████| 3/3
[模型设备] musa:0 | dtype torch.bfloat16

[非思考模式] 回答：
我是通义千问，一个由阿里巴巴集团研发的大型语言模型，能够帮助用户解答问题、创作内容、进行对话等。
```

### 两个关键点

- `device_map="auto"`：由 `accelerate` 自动决定模型放置位置，模型能装进一张卡时就整份落在
  `musa:0`。这段代码与 CUDA 版教程**完全一致**，MUSA 侧不需要改任何参数。
  只想用某几张卡时，用 `MUSA_VISIBLE_DEVICES=0,1` 限制可见设备即可。
- `dtype=torch.bfloat16`：旧版 `transformers` 用 `torch_dtype=`，在 4.57 上会提示
  `` `torch_dtype` is deprecated! Use `dtype` instead! ``，两个参数都能用。

## 二、对话模板与思考模式

Qwen3 是**混合推理模型**，同一份权重可以切换「思考 / 非思考」两种模式，由 `enable_thinking` 控制：

```python
messages = [{"role": "user", "content": "9.11 和 9.8 哪个大？"}]

# 非思考模式：直接回答，快，适合简单问答
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)

# 思考模式：先输出 <think>...</think> 推理过程，再给答案，适合数学/逻辑题
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
```

思考模式下，模型的输出会长这样（节选实录）：

```text
<think>
嗯，用户问的是9.11和9.8哪个大。首先，我需要确认这两个数字的正确数值……
9.11是一个带有两位小数的数，而9.8是一位小数，不过可能写成两位小数的话是9.80。
比较小数的时候，应该从整数部分开始比较，如果整数部分相同，再比较小数部分。
这里两者的整数部分都是9，所以接下来比较小数部分……
</think>

9.8 更大。
```

> 想让模型"永远思考"或"永远不思考"，也可以在加载时直接给出默认值：
> `tokenizer.apply_chat_template(..., enable_thinking=True)` 等价于模型自带的默认行为。

## 三、流式输出

边生成边打印，长回答时体验更好，用 `TextStreamer` 即可：

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
with torch.no_grad():
    model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=1.0,      # 官方推荐采样参数
        top_p=0.95,
        top_k=20,
        streamer=streamer,    # 逐 token 输出
    )
```

> 注意：`do_sample=False`（贪心解码）时，`temperature` / `top_p` / `top_k` 不生效，
> `transformers` 会打印 `The following generation flags are not valid and may be ignored`，属正常现象。

Qwen3 官方推荐的采样参数：

| 场景 | temperature | top_p | top_k |
| --- | --- | --- | --- |
| 思考模式（通用） | 0.6 | 0.95 | 20 |
| 思考模式（数学/代码） | 0.6 | 0.95 | 20 |
| 非思考模式（通用） | 0.7 | 0.8 | 20 |

## 四、常见问题

**1. 报 `Failed to infer device type`**

说明当前 Python 环境里的 `vllm` / `torch_musa` 没有正确加载平台插件。先确认设备可见：

```shell
python3 -c "import torch; print(torch.musa.device_count())"
```

**2. `torch_dtype` 弃用告警**

改用 `dtype=torch.bfloat16` 即可。

## 参考

- [Qwen3 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B)
- [models_musa 环境说明](../README.md)
