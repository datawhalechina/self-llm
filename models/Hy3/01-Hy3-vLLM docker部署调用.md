# 01-Hy3-vLLM 部署调用

## vLLM 简介

`vLLM` 是一个高性能的大语言模型推理与服务框架，具备以下特点：

- 高效的 KV 缓存与内存管理：基于 `PagedAttention` 显著降低显存浪费，提升长文本与高并发场景下的吞吐。
- 兼容 OpenAI 接口：可直接以 `OpenAI API` 形式对外提供 `completions` 与 `chat completions` 能力，便于与现有生态集成。
- 多 GPU 并行与易扩展：支持 Tensor Parallel 等策略，参数简单、易于横向扩展吞吐与上下文长度上限。
- 生态良好：与 `HuggingFace`/`ModelScope` 模型仓库无缝衔接，支持多种推理优化与特性（如推理/思考内容解析、工具调用）。

> `Hy3` 是腾讯混元于 2026 年 7 月 6 日正式发布的旗舰级 MoE 语言模型，采用 Apache 2.0 协议开源，已 "day 0" 接入 Hugging Face 与 ModelScope。本教程基于 官方 vLLM Docker 镜像 部署 FP8 权重，免去了本地配置 Python/CUDA 环境的繁琐。

## 关于 Hy3 架构

`Hy3` 是一个融合快慢思考机制（fast/slow thinking）的 **MoE（混合专家）** 语言模型，关键参数如下：

| 属性                | 数值                                    |
| :------------------ | :-------------------------------------- |
| 架构                | 混合专家（MoE）                         |
| 总参数量            | 2950 亿                                 |
| 激活参数量          | 210 亿                                  |
| MTP 层参数量        | 38 亿                                   |
| 层数（不含 MTP 层） | 80                                      |
| MTP 层数量          | 1                                       |
| 注意力头数          | 64（GQA，8 个 KV 头，每个头维度为 128） |
| 隐藏层维度          | 4096                                    |
| 中间层维度          | 13312                                   |
| 上下文长度          | 256K                                    |
| 词表大小            | 120832                                  |
| 专家数量            | 192 位专家，top-8 激活                  |
| 支持精度            | BF16 / FP8                              |

模型在复杂推理、指令遵循、上下文学习、代码生成与智能体（Agent）能力上表现突出。其思考模式默认开启，可通过请求级参数关闭。

## 环境准备

本教程采用 **官方 vLLM Docker 镜像** 部署，无需在宿主机安装 Python/CUDA 等依赖，只需宿主机满足以下条件：

- 一台带有 NVIDIA GPU 的 Linux 服务器（测试机器 4 × 96GB H20）
- 已安装 [Docker](https://docs.docker.com/engine/install/) 与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)（使容器可调用 GPU）
- 宿主机已配置好 NVIDIA 驱动（`nvidia-smi` 可正常显示）

### 拉取 vLLM 镜像

由于 `Hy3` 是较新的模型，官方提供了专门预构建的 `vllm/vllm-openai:hy3` 镜像（测试中发现 v0.25.1 也可以支持部署），我们直接使用该镜像即可，无需自行编译 vLLM。

```bash
# 拉取 Hy3 专属镜像
docker pull vllm/vllm-openai:hy3
```

> 提示：国内用户若拉取官方镜像较慢，可配置 Docker 镜像加速器（如阿里云、DaoCloud 等），或改用等效的国内镜像源。

### 验证 GPU 可用

在宿主机执行以下命令，确认 Docker 能正确识别 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

能正常打印 GPU 信息即说明环境就绪。

## 模型下载

`Hy3` 已在 Hugging Face（`tencent/Hy3`）与 ModelScope（`Tencent-Hunyuan/Hy3`）上线。由于本教程使用 Docker 部署，推荐**先在宿主机下载模型**，再将模型目录挂载进容器，避免重复下载。

新建 `model_download.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件。

```python
# model_download.py
# 注意修改 cache_dir 为保存的路径
from modelscope import snapshot_download

model_dir = snapshot_download(
    'Tencent-Hunyuan/Hy3',
    cache_dir='请修改我！！！',
    revision='master',
)

print(f"模型下载完成，保存路径为：{model_dir}")
```

然后在终端中输入 `python model_download.py` 执行下载，这里需要耐心等待一段时间直到模型下载完成。

> 注意：记得修改 `cache_dir` 为你的模型下载路径哦~ 记下该路径（下文记为 `/host/model/path/Tencent-Hunyuan/Hy3`），启动容器时会通过 `-v` 挂载进去。


## vLLM Serving

### 启动服务（Docker）

FP8 权重（约 300GB）可根据显存情况灵活选择 GPU 数量。以下为参考配置：

**4 卡部署示例**（测试环境 4 × 96GB H20）：

```bash
docker run --gpus all --ipc=host --name hy3-vllm \
  -v ./Tencent-Hunyuan/Hy3-FP8:/model \
  -p 40061:40061 \
  vllm/vllm-openai:hy3 \
  /model \
  --tensor-parallel-size 4 \
  --disable-access-log-for-endpoints /metrics \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --max-model-len 262144 \
  --tool-call-parser hy_v3 \
  --kv-cache-dtype fp8 \
  --reasoning-parser hy_v3 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 40061 \
  --served-model-name hy3-fp8
```

> 常用 `docker run` 参数说明：
> - `--gpus all`：将全部 GPU 透传给容器（也可指定 `--gpus "device=0,1,2,3"`）。
> - `--ipc=host`：共享宿主机 IPC 命名空间，提升多进程通信与显存共享效率。
> - `-v /host/model/path:/model`：将宿主机模型目录挂载到容器内 `/model`。
> - `-p 40061:40061`：将容器 40061 端口映射到宿主机，便于外部访问。
> - `--tensor-parallel-size 4`：4 卡张量并行。
> - `--max-model-len 262144`：最大上下文长度约 256K。
> - `--kv-cache-dtype fp8`：使用 FP8 量化 KV Cache，降低显存占用。
> - `--reasoning-parser hy_v3` + `--enable-reasoning`：开启推理内容解析，返回 `reasoning_content` 字段。
> - `--tool-call-parser hy_v3` + `--enable-auto-tool-choice`：开启工具调用自动路由。
> - 若需后台运行，可在 `docker run` 后加 `-d`；查看日志用 `docker logs -f hy3-vllm`。

成功启动后，你将看到 `Application startup complete` 的输出。

我们通过上述 vLLM 启动的服务兼容 OpenAI 接口，因此可以通过 Python 的 OpenAI 库进行调用。下面展示日常问答、数学推理、代码写作和工具调用的实际例子来测试 `Hy3` 的能力。

### 示例测试

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:40061/v1"  # 使用正确的端口

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
        model="hy3",
        messages=[{"role": "user", "content": prompts[i]}],
        temperature=0,
        max_tokens=8000
    )

    print(f"问题 {i+1}: {prompts[i]}")
    print(f"Hy3 回复 {i+1}: {response.choices[0].message.content}")
    print("-" * 100)
```


### 关键参数说明

- `--tensor-parallel-size`：张量并行划分数。等于所用 GPU 数时较常见；多卡可提升吞吐和可用上下文长度上限。
- `--max-model-len`：单请求最大上下文长度（输入+输出）。越大显存占用越高，易触发 OOM。可按显存情况下调，如 8192。
- `--gpu_memory_utilization`：vLLM 目标可用显存比例（0~1）。OOM 可尝试调低，如 0.8/0.7。
- `--served-model-name`：对外暴露的模型名。客户端需用同名 `model` 调用。
- `--port`/`--host`：服务监听端口/地址。云主机需放通端口安全组。
- `--trust_remote_code`：允许加载仓库中的自定义代码（必需，否则部分模型无法正确初始化）。
- `--kv-cache-dtype fp8`：使用 FP8 量化 KV Cache，显著降低显存占用，可支持更长上下文。
- `--reasoning-parser hy_v3`：开启推理内容解析，返回 `reasoning_content` 字段，便于展示"思考"过程。
- `--tool-call-parser hy_v3` + `--enable-auto-tool-choice`：开启工具调用自动路由。

> 显存预算建议：FP8 版本配合 `--kv-cache-dtype fp8` 可显著降低显存需求，4 × 96GB H20 可流畅运行 256K 上下文。

### 健康检查

```bash
curl http://127.0.0.1:40061/v1/models
```

### 工具调用

工具调用是大语言模型的一项重要能力，`Hy3` 同样支持工具调用（见[官方说明](https://huggingface.co/tencent/Hy3)）。我们以天气查询为例测试模型的工具调用能力：在部署时需加入 `--tool-call-parser hy_v3` 与 `--enable-auto-tool-choice` 参数。

在一般测试代码基础上加上工具调用的示例如下（模拟一个查询天气的函数，真实使用需外接 API）：

```python
from openai import OpenAI
import json

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8085/v1"
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
    model="hy3",
    messages=tool_messages,
    temperature=0,
    max_tokens=7500,
    tools=weather_tools,
    tool_choice="auto",
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
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
    model="hy3",
    messages=tool_messages,
    temperature=0,
    max_tokens=8000,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

print(f"Hy3 工具调用回复: {first_tool_response.choices[0].message.content}")
print(f"Hy3 回复 4: {final_tool_response.choices[0].message.content}")
```

在工具调用中，`Hy3` 能够适时调用工具并最终返回准确的天气查询信息。
