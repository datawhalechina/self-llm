## vLLM 零基础大模型部署（Ubuntu 24.04 + ROCm 7+）

> 内容参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm) `docs/zh/01-deploy`，适配 self-llm 目录结构。完整与最新版请以 hello-rocm 为准。

本节介绍在 Ubuntu 24.04 + ROCm 7+ 环境下，使用 **vLLM** 部署和调用 **Gemma 4** 模型。

当前 vLLM 官方文档推荐优先使用官方 ROCm Docker 镜像：

```text
vllm/vllm-openai-rocm:latest
```

旧的 `rocm/vllm`、`rocm/vllm-dev` 镜像已不再作为 vLLM 官方新文档的首选路径。若只想快速跑通推理服务，Docker 方式比手动编译 Triton / FlashAttention / vLLM 更简单、可复现。

> 前置条件：已完成 [Ubuntu 24.04 + ROCm 7 环境准备](./1-env-prepare-ubuntu24-rocm7.md)。Gemma 模型通常需要先在 Hugging Face 模型页点击 **Agree & Access**，并准备具备 `read` 权限的 `HF_TOKEN`。

---

## 一、方式一：官方 vLLM ROCm Docker 镜像（推荐）

### 1. 路线 A：ROCm 7.13 官方验证镜像（gfx1151）

ROCm 7.13 官方文档提供了针对 `gfx1151` 的 vLLM 0.19.1 Docker 镜像：

```bash
docker pull rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

> 注意：该镜像内置 PyTorch 2.10.0 + vLLM 0.19.1；PyTorch 2.11.0 属于 ROCm 7.13 pip 安装路线，不要把两条路线的版本混写。

启动容器并进入 shell：

```bash
docker run -it --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --network=host \
  --ipc=host \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ~/models:/app/models \
  -e HF_HOME="/app/models" \
  rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  bash
```

容器内可以继续执行 `vllm serve`。

---

### 2. 路线 B：vLLM upstream 通用镜像

vLLM 官方 upstream 文档推荐使用 `vllm/vllm-openai-rocm` 镜像。该路线适合希望跟随 vLLM 官方最新发布的用户。

```bash
docker pull vllm/vllm-openai-rocm:latest
```

如果需要尝试最新预览版本，可使用：

```bash
docker pull vllm/vllm-openai-rocm:nightly
```

日常教程建议使用 `latest`，避免 nightly 版本变化导致行为不稳定。

---

### 3. 启动 Gemma 4 服务

以下示例以 `google/gemma-4-E4B-it` 为例。若使用其他 Gemma 4 版本，请替换 `--model` 与 `--served-model-name`。

```bash
export HF_TOKEN="hf_***"

docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model google/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

参数说明：

- `--device /dev/kfd --device /dev/dri`：将 AMD GPU 设备暴露给容器。
- `-v ~/.cache/huggingface:/root/.cache/huggingface`：复用宿主机 Hugging Face 缓存，避免重复下载模型。
- `--env "HF_TOKEN=$HF_TOKEN"`：传入 Hugging Face token，Gemma 等受限模型需要。
- `--max-model-len 4096`：控制最大上下文长度；显存不足时可降低到 2048。
- `--gpu-memory-utilization 0.9`：控制 vLLM 可使用的显存比例。

服务启动后，使用以下命令检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

---

### 4. 使用本地模型目录启动

如果已经提前下载好模型，可将本地目录挂载到容器中：

```bash
docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/models:/app/models \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:latest \
  --model /app/models/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --max-model-len 4096
```

本地模型目录需要包含 `config.json`、tokenizer 文件和权重文件。

---

## 二、OpenAI-compatible API 调用

vLLM 默认提供 OpenAI-compatible API。

### 1. curl 调用

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it",
    "messages": [
      {"role": "user", "content": "用一句话介绍 ROCm。"}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 256
  }'
```

### 2. Python 调用

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=[{"role": "user", "content": "用一句话介绍深度学习。"}],
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

---

## 三、方式二：ROCm wheel 安装（可选）

如果不使用 Docker，也可以安装 vLLM ROCm wheel。该方式对 Python / ROCm / glibc 版本要求更严格。

官方当前要求重点：

- Python 3.12
- ROCm 7.0 或 ROCm 7.2.1 对应 wheel
- glibc >= 2.35

推荐使用 `uv` 创建 Python 3.12 环境：

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

> 注意：如果 Python 版本不匹配，安装器可能回退到 CUDA wheel，随后在 AMD GPU 上出现 `libcudart.so` 相关错误。遇到该问题时，优先检查 Python 是否为 3.12，以及是否使用了 ROCm wheel 索引。

安装完成后启动服务：

```bash
vllm serve google/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --max-model-len 4096
```

---

## 四、方式三：源码编译（进阶）

源码编译适合需要修改 vLLM、调试算子或适配特殊硬件的场景。普通部署不建议优先选择该方式。

官方 ROCm 构建文档：

- [https://docs.vllm.ai/en/stable/getting_started/installation/gpu/](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

编译前通常需要确认 GPU 架构：

```bash
rocminfo | grep gfx
```

然后设置：

```bash
export PYTORCH_ROCM_ARCH="gfx1151"
```

具体依赖和构建步骤请以 vLLM 官方文档为准。

---

## 五、常见问题

### 1. Gemma 模型下载失败或 403

先在 Hugging Face 模型页点击 **Agree & Access**，然后生成具备 `read` 权限的 token，并通过 `HF_TOKEN` 传入容器。

### 2. 容器里看不到 GPU

确认宿主机存在 `/dev/kfd` 和 `/dev/dri`，Docker 命令中包含对应 `--device` 参数，并且当前用户具备运行 Docker 的权限。

### 3. 显存不足

降低 `--max-model-len`，例如从 4096 降到 2048；也可以降低并发或选择更小模型。

### 4. 应该使用 AMD 的 `rocm/vllm` 镜像吗？

新文档优先使用 vLLM 官方镜像 `vllm/vllm-openai-rocm`。AMD 的固定版本镜像更适合特定硬件/版本验证，不作为本教程主路径。

---

## 延伸学习

更完整的 ROCm 部署、微调与基础设施教程，请前往 Datawhale 项目 [hello-rocm](https://github.com/datawhalechina/hello-rocm)：

- 仓库：[https://github.com/datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm)
- 在线文档：[https://datawhalechina.github.io/hello-rocm/](https://datawhalechina.github.io/hello-rocm/)
- 部署专区原文：[https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy](https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy)

> 本文内容参考 hello-rocm `01-deploy` 教程，并按 self-llm `models_amd` 目录结构做了路径适配。

