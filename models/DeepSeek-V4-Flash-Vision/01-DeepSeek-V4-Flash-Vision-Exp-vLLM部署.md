# 01-DeepSeek-V4-Flash-Vision-Exp vLLM docker部署调用

## vLLM 简介

`vLLM` 是一个高性能的大语言模型推理与服务框架，具备以下特点：

- 高效的 KV 缓存与内存管理：基于 `PagedAttention` 显著降低显存浪费，提升长文本与高并发场景下的吞吐。
- 兼容 OpenAI 接口：可直接以 `OpenAI API` 形式对外提供 `completions` 与 `chat completions` 能力，便于与现有生态集成。
- 多 GPU 并行与易扩展：支持 Tensor Parallel 等策略，参数简单、易于横向扩展吞吐与上下文长度上限。
- 生态良好：与 `HuggingFace`/`ModelScope` 模型仓库无缝衔接，支持多种推理优化与特性（如推理/思考内容解析、工具调用）。

## 关于 DeepSeek-V4-Flash-Vision-Exp
`DeepSeek-V4-Flash-Vision-Exp` 是 DeepSeek 开源的首个 V4 系列多模态模型。它在 `DeepSeek-V4-Flash` 文本底座上新增了视觉理解能力，支持对图片进行描述、截图文字识别、图表分析等任务，同时保持了与纯文本 `V4-Flash` 一致甚至更强的 Agent、推理与世界知识能力。

主要特性：

- **统一视觉 Token 上限**：每张图片最多消耗 384 个 Token，长图与短图在缩放后 Token 成本一致；
- **大容量多图输入**：单请求最多可输入 600 张图片，适合 `Screenshot Agent`、`Computer Use`、多模态 RAG 等场景；
- **MIT 协议全开放**：完整开放模型权重、视觉编码器、Aligner、DFlash Attention、MoE、DSpark 等核心模块，可商用、可修改；

## 环境准备
本教程采用 **官方 vLLM Docker 镜像** 部署，无需在宿主机安装 Python/CUDA 等依赖，只需宿主机满足以下条件：

- 一台带有 NVIDIA GPU 的 Linux 服务器（测试机器 4 × 96GB H20）
- 已安装 [Docker](https://docs.docker.com/engine/install/) 与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)（使容器可调用 GPU）
- 宿主机已配置好 NVIDIA 驱动（`nvidia-smi` 可正常显示）

### 拉取 vLLM 镜像

Vision-Exp 是 8 月底才开源的实验性模型，涉及 DFlash Attention、Hyper-Connections、DSpark 自投机 等一整套新结构，普通 vLLM 版本跑不起来，必须用 DeepSeek 官方维护的专用镜像 **`vllm/vllm-openai:deepseekv4-flash-vision`**

```bash
# 拉取 Vision-Exp 专属镜像
docker pull vllm/vllm-openai:deepseekv4-flash-vision
```

> 提示：国内用户若拉取官方镜像较慢，可配置 Docker 镜像加速器（如阿里云、DaoCloud 等），或改用等效的国内镜像源。

### 验证 GPU 可用

在宿主机执行以下命令，确认 Docker 能正确识别 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

能正常打印 GPU 信息即说明环境就绪。

## 模型下载
`deepseekv4-flash-vision-exp` 已在 Hugging Face（`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`）与 ModelScope（`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`）上线。由于本教程使用 Docker 部署，推荐**先在宿主机下载模型**，再将模型目录挂载进容器，避免重复下载。

使用 `modelscope` 中的 `snapshot_download` 函数下载模型。新建 `model_download.py` 文件并输入以下内容：

```python
# model_download.py
from modelscope import snapshot_download

model_dir = snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp',
    cache_dir='your_model_dir',   # 请修改为你自己的模型下载路径
    revision='master'
)
print(f"模型下载完成，保存路径为：{model_dir}")
```

在终端中执行下载：

```bash
python model_download.py
```

> 注意：请记得修改 `cache_dir` 为你自己的模型下载路径哦~

如果网络环境可以访问 Hugging Face，也可以直接使用：

```bash
pip install -U huggingface_hub
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir /path/to/DeepSeek-V4-Flash-Vision-Exp
```

两种方式任选其一即可，不要重复下载。

## vLLM Serving

`DeepSeek-V4-Flash-Vision-Exp` 是视觉模型，需要在加载时指定视觉相关参数。推荐使用 `vLLM` 官方 Docker 镜像或 `vllm serve` 方式启动。

4 卡部署示例**
>测试环境 4 × 96GB H20 
Driver Version: 570.172.08     CUDA Version: 12.8

```bash
docker run --gpus all --ipc=host --name DeepSeek-V4-Flash-Vision \
    --shm-size 32g \
    -p 8000:8000 \
    -v /models/DeepSeek-V4-Flash-Vision-Exp:/models \
    vllm/vllm-openai:deepseekv4-flash-vision \
    --model /models \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --served-model-name deepseek-v4-flash-vision-exp \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
    --gpu-memory-utilization=0.95 \
    --max-model-len 262144
```

## 5. 图片理解接口调用

服务启动后，可以通过兼容 `OpenAI` 的接口进行多模态推理。图片支持以下两种传入方式。

### 使用 Base64 内联图片

对于本地图片，最简便的方式是将图片进行 `base64` 编码后，以内联的 `data:` URL 形式传入请求，也就是把 `content` 从纯字符串改成由文本块和图片块组成的数组：

```python
# test_vision_base64.py
import base64

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # 本地 vLLM 服务地址
    api_key="EMPTY",                       # 本地服务不需要真实 key，随便填写即可
)

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="deepseek-v4-flash-vision-exp",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片的内容。"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

运行：

```bash
python test_vision_base64.py
```


