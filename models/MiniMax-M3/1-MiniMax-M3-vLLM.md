# MiniMax-M3 vLLM 部署调用

## MiniMax-M3 简介

[MiniMax-M3](https://github.com/MiniMax-AI/MiniMax-M3) 是 MiniMax 推出的最新一代开源大语言模型。相较于 M2.5，M3 在长上下文、推理质量和多模态能力上均有明显提升：上下文窗口扩展到 **512K**、单次最大输出可达 **128K**，并原生支持**图片输入**（OpenAI 兼容与 Anthropic 兼容两套接口均可使用，目前仅图片，视频/音频/文档暂不支持）。

本文以 MiniMax-M3 为模型基座，演示如何通过 vLLM 完成模型下载、服务启动与客户端调用（含图片输入）。

## vLLM 简介

`vLLM` 是一个面向大语言模型的高性能部署推理框架，提供开箱即用的推理加速与 OpenAI 兼容接口。它支持长上下文推理、流式输出、多卡并行（如张量并行与专家并行）、工具调用与"思考内容"解析等能力，便于将最新模型快速落地到生产环境。

## 环境准备

基础环境（参考值）：

> 可用 `nvidia-smi` 与 `python -c "import torch;print(torch.cuda.is_available())"` 自检 CUDA / PyTorch。

显存与推荐配置（按官方文档）：
- 权重需求约 220 GB 显存；每 1M 上下文 token 约需 240 GB 显存
- 96G × 4 GPU：支持约 40 万 token 总上下文
- 144G × 8 GPU：支持约 300 万 token 总上下文

> **注**：上下文窗口最大为 512K，单次最大输出为 128K；以上数值为硬件支持的最大并发缓存总量。

安装依赖：

> 建议使用虚拟环境（venv / conda / uv）避免依赖冲突

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

> 请确保 vLLM 版本支持 MiniMax-M3。若启动时报模型不支持，请升级 `pip install -U vllm`。

## 模型下载

vLLM 会在首次启动时自动从 Hugging Face 拉取并缓存模型，无需手动下载。若希望提前下载或受网络限制，可选用 `modelscope` 手动下载模型：

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
        "MiniMaxAI/MiniMax-M3 --trust-remote-code "
        "--tensor-parallel-size 4 "
        "--enable-auto-tool-choice --tool-call-parser minimax_m2 "
        "--reasoning-parser minimax_m2_append_think"
    )
elif gpu_count == 8:
    cmd = (
        "SAFETENSORS_FAST_GPU=1 vllm serve "
        "MiniMaxAI/MiniMax-M3 --trust-remote-code "
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
    MiniMaxAI/MiniMax-M3 --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
```

8 卡部署：

```bash
SAFETENSORS_FAST_GPU=1 vllm serve \
    MiniMaxAI/MiniMax-M3 --trust-remote-code \
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
    "model": "MiniMaxAI/MiniMax-M3",
    "messages": [
      {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
      {"role": "user", "content": [{"type": "text", "text": "请简要介绍 MiniMax-M3 模型的特点。"}]}
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
    model="MiniMaxAI/MiniMax-M3",
    messages=[
        {"role": "user", "content": "请介绍 MiniMax-M3 相比 M2.5 有哪些提升？"}
    ],
    max_tokens=8192,
    top_p=0.95,
    temperature=1.0,
)

msg = response.choices[0].message
print("MiniMax-M3:", msg.content)
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
    model="MiniMaxAI/MiniMax-M3",
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

### 图片输入（Vision）

MiniMax-M3 原生支持图片输入。在 OpenAI 兼容接口中，将 `image_url` 与文本一起放入 `content` 数组即可：

```python
# test_vision.py
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")

response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M3",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片中的内容。"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg"},
                },
            ],
        }
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

> 仅支持图片输入；视频、音频、文档暂不支持。

### 工具调用（Tool Calling）

MiniMax-M3 在 Agent 和工具调用方面继续保持强表现，能够稳定执行复杂长链条工具调用任务。在 vLLM 部署时通过 `--enable-auto-tool-choice --tool-call-parser minimax_m2` 启用工具调用功能。

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

- `model`：启动时指定的模型名称或本地路径（例：`MiniMaxAI/MiniMax-M3`）；OpenAI 请求中的 `model` 需与之对应。
- `--tensor-parallel-size`：张量并行大小，常设为 GPU 数量（4 或 8）。
- `--enable_expert_parallel`：启用专家并行（8 卡示例中开启）。
- `--enable-auto-tool-choice`：自动工具选择（使模型在需要时自主发起工具调用）。
- `--tool-call-parser minimax_m2`：启用 MiniMax 的工具调用解析。
- `--reasoning-parser minimax_m2_append_think`：启用思考内容解析（将 reasoning 以追加方式处理）。
- `max_tokens`：控制生成长度，M3 支持最大 128K 输出；过大将增加显存和时延。
- `temperature`/`top_p`：控制多样性。官方推荐参数：temperature=1.0, top_p=0.95, top_k=20。

> 显存建议：M3 权重约 220 GB；每 1M 上下文约 240 GB。请结合业务并发与上下文需求评估资源。

## 常见问题

### Hugging Face 网络问题

如果遇到网络问题，可设置镜像后再进行拉取：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### MiniMax-M3 model is not currently supported

该 vLLM 版本过旧，请升级到最新版本：

```bash
pip install -U vllm
```

### torch.AcceleratorError: CUDA error

在启动参数添加 `--compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"` 可以解决。

### 模型输出乱码

请升级到最新版本的 vLLM，并确保使用与 MiniMax-M3 适配的版本。

## 参考链接

- [MiniMax-M3 GitHub](https://github.com/MiniMax-AI/MiniMax-M3)
- [MiniMax-M3 HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M3)
- [MiniMax 平台文档](https://platform.minimax.io/docs/guides/text-generation)
- [vLLM 官方文档](https://docs.vllm.ai/en/stable/)
