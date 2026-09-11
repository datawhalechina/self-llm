# 01-DeepSeek-V4.1-Flash-vLLM docker 部署调用

## vLLM 简介

`vLLM` 是一个高性能的大语言模型推理与服务框架，具备以下特点：

- 高效的 KV 缓存与内存管理：基于 `PagedAttention` 显著降低显存浪费，提升长文本与高并发场景下的吞吐。
- 兼容 OpenAI 接口：可直接以 `OpenAI API` 形式对外提供 `completions` 与 `chat completions` 能力，便于与现有生态集成。
- 多 GPU 并行与易扩展：支持 Tensor Parallel、Expert Parallel 等策略，参数简单、易于横向扩展吞吐与上下文长度上限。
- 生态良好：与 `HuggingFace`/`ModelScope` 模型仓库无缝衔接，支持多种推理优化与特性（如推理/思考内容解析、工具调用）。

> `DeepSeek-V4.1-Flash` 是 DeepSeek 推出的 Flash 系列语言模型，官方权重已上线 Hugging Face 与 ModelScope（`deepseek-ai/DeepSeek-V4.1-Flash`）。本教程基于官方 vLLM Docker 镜像部署，免去了本地配置 Python/CUDA 环境的繁琐。

## 环境准备

本教程采用 **官方 vLLM Docker 镜像** 部署，无需在宿主机安装 Python/CUDA 等依赖，只需宿主机满足以下条件：

- 一台带有 NVIDIA GPU 的 Linux 服务器（本教程测试机器 8 × 96GB）
- 已安装 [Docker](https://docs.docker.com/engine/install/) 与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)（使容器可调用 GPU）
- 宿主机已配置好 NVIDIA 驱动（`nvidia-smi` 可正常显示）

### 拉取 vLLM 镜像

由于 `DeepSeek-V4.1-Flash` 是较新的模型，官方提供了预构建的专属镜像，我们直接使用该镜像即可，无需自行编译 vLLM。

```bash
# 拉取 DeepSeek-V4.1-Flash 专属镜像
docker pull vllm/vllm-openai:deepseekv41-flash-0909
```

> 提示：国内用户若拉取官方镜像较慢，可配置 Docker 镜像加速器（如阿里云、DaoCloud 等），或改用等效的国内镜像源。

### 验证 GPU 可用

在宿主机执行以下命令，确认 Docker 能正确识别 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

能正常打印 GPU 信息即说明环境就绪。

## 模型下载

`DeepSeek-V4.1-Flash` 已在 Hugging Face（`deepseek-ai/DeepSeek-V4.1-Flash`）与 ModelScope 上线。由于本教程使用 Docker 部署，推荐**先在宿主机下载模型**，再将模型目录挂载进容器，避免重复下载。

新建 `model_download.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件。

```python
# model_download.py
# 注意修改 cache_dir 为保存的路径
from modelscope import snapshot_download

model_dir = snapshot_download(
    'deepseek-ai/DeepSeek-V4.1-Flash',
    cache_dir='请修改我！！！',
    revision='master',
)

print(f"模型下载完成，保存路径为：{model_dir}")
```

然后在终端中输入 `python model_download.py` 执行下载，这里需要耐心等待一段时间直到模型下载完成。

> 注意：记得修改 `cache_dir` 为你的模型下载路径哦~ 记下该路径（下文记为 `/host/model/path/DeepSeek-V4.1-Flash`），启动容器时会通过 `-v` 挂载进去。

## vLLM Serving

### 启动服务（Docker）

以下为 8 卡 96GB 部署示例：

```bash
docker run --gpus all --ipc=host --name deepseek-v41-flash-vllm \
  -v /host/model/path/DeepSeek-V4.1-Flash:/model \
  -p 8000:8000 \
  vllm/vllm-openai:deepseekv41-flash-0909 \
  --model /model \
  --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size 8 \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.91 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --enable-chunked-prefill \
  --host 0.0.0.0 \
  --port 8000
```

> 常用 `docker run` 参数说明：
> - `--gpus all`：将全部 GPU 透传给容器（也可指定 `--gpus "device=0,1,2,3,4,5,6,7"`）。
> - `--ipc=host`：共享宿主机 IPC 命名空间，提升多进程通信与显存共享效率。
> - `-v /host/model/path:/model`：将宿主机模型目录挂载到容器内 `/model`。
> - `-p 8000:8000`：将容器 8000 端口映射到宿主机，便于外部访问。
> - 若需后台运行，可在 `docker run` 后加 `-d`；查看日志用 `docker logs -f deepseek-v41-flash-vllm`。

成功启动后，你将看到 `Application startup complete` 的输出。

我们通过上述 vLLM 启动的服务兼容 OpenAI 接口，因此可以通过 Python 的 OpenAI 库进行调用。

### 关键参数说明

| 参数 | 作用 |
| --- | --- |
| `--served-model-name deepseek-v4.1-flash` | 对外暴露的模型名，客户端需用同名 `model` 调用 |
| `--tensor-parallel-size 8` | 张量并行划分数，将模型部署到 8 张 GPU |
| `--tokenizer-mode deepseek_v41` | 使用 DeepSeek-V4.1 专用分词方式 |
| `--tool-call-parser deepseek_v41` | 解析 DeepSeek-V4.1 工具调用结果 |
| `--enable-auto-tool-choice` | 开启工具调用自动路由 |
| `--reasoning-parser deepseek_v41` | 解析思考内容与最终回答，返回 `reasoning_content` 字段 |
| `--kv-cache-dtype fp8` | 使用 FP8 量化 KV Cache，降低长上下文显存占用 |
| `--gpu-memory-utilization 0.91` | vLLM 目标可用显存比例（0~1），OOM 时可尝试调低 |
| `--max-model-len 262144` | 单请求最大上下文长度（约 256K），越大显存占用越高 |
| `--max-num-seqs 128` | 单个批次最多同时处理的序列数量 |
| `--enable-chunked-prefill` | 开启分块预填充，降低长输入首 token 时延与显存峰值 |

> 显存预算建议：FP8 KV Cache 配合 `--enable-chunked-prefill` 可显著降低显存需求，8 × 96GB 可流畅运行 256K 上下文。

### 健康检查

```bash
curl http://127.0.0.1:8000/v1/models
```

### 基础调用

在一般测试代码中，我们可以通过 OpenAI 库调用刚刚部署好的模型服务：

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"  # 使用正确的端口

prompt_daily_chat = "你好，你是谁"
prompt_math_reasoning = "Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$."
prompt_coding = "写一个python程序，实现快速排序"
prompts = [prompt_daily_chat, prompt_math_reasoning, prompt_coding]

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

for i in range(len(prompts)):
    response = client.chat.completions.create(
        model="deepseek-v4.1-flash",
        messages=[{"role": "user", "content": prompts[i]}],
        temperature=0,
        max_tokens=8000
    )

    print(f"问题 {i+1}: {prompts[i]}")
    print(f"DeepSeek-V4.1-Flash 回复 {i+1}: {response.choices[0].message.content}")
    print("-" * 100)
```

### 工具调用

工具调用是大语言模型的一项重要能力，`DeepSeek-V4.1-Flash` 同样支持工具调用。在部署时需加入 `--tool-call-parser deepseek_v41` 与 `--enable-auto-tool-choice` 参数。

我们以天气查询为例测试模型的工具调用能力（模拟一个查询天气的函数，真实使用需外接 API）：

```python
from openai import OpenAI
import json

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)


def get_weather(location: str, date: str = "今天", unit: str = "celsius") -> str:
    """示例工具：查询天气（示例中返回模拟结果）。"""
    normalized_unit = "°C" if unit == "celsius" else "°F"
    return f"{location}{date}多云，气温 28{normalized_unit}，湿度 70%，东北风 3 级。"


tool_messages = [
    {"role": "system", "content": "你可以调用工具来获取实时天气。"},
    {"role": "user", "content": "帮我查一下深圳今天的天气，用摄氏度。"},
]

weather_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "根据地点与日期查询天气，返回简要的天气描述。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名，如北京、深圳"},
                    "date": {"type": "string", "description": "日期，YYYY-MM-DD 或 今天/明天", "default": "今天"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                },
                "required": ["location"],
            },
        },
    }
]

first_tool_response = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=tool_messages,
    temperature=0,
    max_tokens=7500,
    tools=weather_tools,
    tool_choice="auto",
)

assistant_msg = first_tool_response.choices[0].message
tool_messages.append({
    "role": "assistant",
    "content": assistant_msg.content or "",
    "tool_calls": assistant_msg.tool_calls,
})

tool_calls = assistant_msg.tool_calls or []
for tool_call in tool_calls:
    function_name = tool_call.function.name
    try:
        function_args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        function_args = {}

    if function_name == "get_weather":
        tool_result = get_weather(
            location=function_args.get("location", "未知城市"),
            date=function_args.get("date", "今天"),
            unit=function_args.get("unit", "celsius"),
        )
    else:
        tool_result = f"不支持的工具: {function_name}"

    tool_messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result,
    })

final_tool_response = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=tool_messages,
    temperature=0,
    max_tokens=8000,
)

print(f"DeepSeek-V4.1-Flash 工具调用回复: {first_tool_response.choices[0].message.content}")
print(f"DeepSeek-V4.1-Flash 回复 4: {final_tool_response.choices[0].message.content}")
```

在工具调用中，`DeepSeek-V4.1-Flash` 能够适时调用工具并最终返回准确的天气查询信息。
