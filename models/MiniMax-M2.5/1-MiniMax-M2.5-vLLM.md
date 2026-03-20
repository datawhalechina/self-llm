# MiniMax-M2.5 vLLM 部署调用

## MiniMax-M2.5 简介

[MiniMax-M2.5](https://github.com/MiniMax-AI/MiniMax-M2.5) 是 MiniMax 推出的最新一代开源大语言模型。M2.5 通过在数十万个复杂真实场景中进行强化学习训练，在代码、Agent 工具调用与搜索、办公等多项经济价值任务中达到 SOTA 水平，包括 SWE-Bench Verified 80.2%、Multi-SWE-Bench 51.3%、BrowseComp 76.3% 等优异成绩。

M2.5 兼具高效与低成本——以 100 token/s 的速度连续运行一小时仅需 1 美元。本文以 MiniMax-M2.5 为模型基座，演示如何通过 vLLM 完成模型下载、服务启动与客户端调用。

## vLLM 简介

`vLLM` 是一个面向大语言模型的高性能部署推理框架，提供开箱即用的推理加速与 OpenAI 兼容接口。它支持长上下文推理、流式输出、多卡并行（如张量并行与专家并行）、工具调用与"思考内容"解析等能力，便于将最新模型快速落地到生产环境。

## 环境准备

基础环境（参考值）：

> 可用 `nvidia-smi` 与 `python -c "import torch;print(torch.cuda.is_available())"` 自检 CUDA / PyTorch。

显存与推荐配置（按官方文档）：
- 权重需求约 220 GB 显存；每 1M 上下文 token 约需 240 GB 显存
- 96G × 4 GPU：支持约 40 万 token 总上下文
- 144G × 8 GPU：支持约 300 万 token 总上下文

> **注**：以上数值为硬件支持的最大并发缓存总量，模型单序列（Single Sequence）长度上限为 196k。

安装依赖：

> 建议使用虚拟环境（venv / conda / uv）避免依赖冲突

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

## 模型下载

vLLM 会在首次启动时自动从 Hugging Face 拉取并缓存模型，无需手动下载。若希望提前下载或受网络限制，可选用 `modelscope` 手动下载模型：

```python
# model_download.py
from modelscope import snapshot_download

model_dir = snapshot_download('MiniMaxAI/MiniMax-M2.5', cache_dir='/root/autodl-tmp', revision='master')
print(f"模型下载成功，保存到: {model_dir}")
```

```bash
pip install modelscope
python model_download.py
```

> 注意：使用 `modelscope` 下载模型时无需设置 HF 镜像。若不使用 modelscope，而是让 vLLM 自动从 Hugging Face 拉取，网络受限时可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

## 启动 vLLM 服务

### Python 启动脚本

新建 `start_server.py`：

```python
import torch
from vllm.utils import launch_server_cmd, wait_for_server

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
if gpu_count == 4:
    cmd = (
        "SAFETENSORS_FAST_GPU=1 vllm serve "
        "MiniMaxAI/MiniMax-M2.5 --trust-remote-code "
        "--tensor-parallel-size 4 "
        "--enable-auto-tool-choice --tool-call-parser minimax_m2 "
        "--reasoning-parser minimax_m2_append_think"
    )
elif gpu_count == 8:
    cmd = (
        "SAFETENSORS_FAST_GPU=1 vllm serve "
        "MiniMaxAI/MiniMax-M2.5 --trust-remote-code "
        "--enable_expert_parallel --tensor-parallel-size 8 "
        "--enable-auto-tool-choice --tool-call-parser minimax_m2 "
        "--reasoning-parser minimax_m2_append_think"
    )
else:
    raise RuntimeError(f"建议使用 4 或 8 张 GPU，当前检测到: {gpu_count}")

server_process, port = launch_server_cmd(cmd, port=8000)
wait_for_server(f"http://127.0.0.1:{port}")
print(f"vLLM Server started: http://127.0.0.1:{port}")
```

启动：

```bash
python start_server.py
```

服务启动成功后将监听 `http://127.0.0.1:8000/v1`。

### 命令行直接启动

4 卡部署：

```bash
SAFETENSORS_FAST_GPU=1 vllm serve \
    MiniMaxAI/MiniMax-M2.5 --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
```

8 卡部署：

```bash
SAFETENSORS_FAST_GPU=1 vllm serve \
    MiniMaxAI/MiniMax-M2.5 --trust-remote-code \
    --enable_expert_parallel --tensor-parallel-size 8 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
```

> 由于模型较大，首次加载时间较长，可能需要半小时以上。

> 如遇到 CUDA 内存错误，可在启动参数中添加 `--compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"` 解决。

## curl 测试

使用 curl 调用 OpenAI 兼容接口：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-M2.5",
    "messages": [
      {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
      {"role": "user", "content": [{"type": "text", "text": "请简要介绍 MiniMax-M2.5 模型的特点。"}]}
    ]
  }'
```

## 调用示例

以下示例均使用 OpenAI 官方 Python SDK 调用 vLLM 的 OpenAI 兼容接口。

### 聊天对话（Chat Completions）

```python
# test_chat.py
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M2.5",
    messages=[
        {"role": "user", "content": "请介绍 MiniMax-M2.5 模型相比 M2 有哪些提升？"}
    ],
    max_tokens=8192,
    top_p=0.95,
    temperature=1.0,
)

msg = response.choices[0].message
print("MiniMax-M2.5:", msg.content)
```

运行：

```bash
python test_chat.py
```

### 流式输出（Streaming）

```python
# test_streaming.py
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

stream = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M2.5",
    messages=[{"role": "user", "content": "请用 Python 写一个快速排序算法，并附带详细注释。"}],
    stream=True,
    max_tokens=32768,
    top_p=0.95,
    temperature=1.0,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="", flush=True)
```

运行：

```bash
python test_streaming.py
```

### 工具调用（Tool Calling）

MiniMax-M2.5 在 Agent 和工具调用方面表现出色，能够稳定执行复杂长链条工具调用任务。在 vLLM 部署时通过 `--enable-auto-tool-choice --tool-call-parser minimax_m2` 启用工具调用功能。

```python
# test_tool_calling.py
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."

tool_functions = {"get_weather": get_weather}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "北京今天天气怎么样？请用摄氏度。"}],
    tools=tools,
    tool_choice="auto"
)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")
```

运行：

```bash
python test_tool_calling.py
```

## 参数说明与建议

- `model`：启动时指定的模型名称或本地路径（例：`MiniMaxAI/MiniMax-M2.5`）；OpenAI 请求中的 `model` 需与之对应。
- `--tensor-parallel-size`：张量并行大小，常设为 GPU 数量（4 或 8）。
- `--enable_expert_parallel`：启用专家并行（8 卡示例中开启）。
- `--enable-auto-tool-choice`：自动工具选择（使模型在需要时自主发起工具调用）。
- `--tool-call-parser minimax_m2`：启用 MiniMax 的工具调用解析。
- `--reasoning-parser minimax_m2_append_think`：启用思考内容解析（将 reasoning 以追加方式处理）。
- `max_tokens`：控制生成长度；过大将增加显存和时延。
- `temperature`/`top_p`：控制多样性。官方推荐参数：temperature=1.0, top_p=0.95, top_k=20。

> 显存建议：M2.5 权重约 220 GB；每 1M 上下文约 240 GB。请结合业务并发与上下文需求评估资源。

## 常见问题

### Hugging Face 网络问题

如果遇到网络问题，可设置镜像后再进行拉取：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### MiniMax-M2 model is not currently supported

该 vLLM 版本过旧，请升级到最新版本：

```bash
pip install -U vllm
```

### torch.AcceleratorError: CUDA error

在启动参数添加 `--compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"` 可以解决。

### 模型输出乱码

请升级到最新版本的 vLLM（至少确保版本在提交 [cf3eacfe](https://github.com/vllm-project/vllm/commit/cf3eacfe58fa9e745c2854782ada884a9f992cf7) 之后）。

## 参考链接

- [MiniMax-M2.5 GitHub](https://github.com/MiniMax-AI/MiniMax-M2.5)
- [MiniMax-M2.5 HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMax-M2.5 官方 vLLM 部署指南](https://github.com/MiniMax-AI/MiniMax-M2.5/blob/main/docs/vllm_deploy_guide_cn.md)
- [vLLM 官方文档](https://docs.vllm.ai/en/stable/)
