## Qwen3.5-4B vLLM 部署调用（Ubuntu 24.04 + ROCm 7+）

> 内容参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm) `docs/zh/01-deploy`，适配 self-llm 目录结构。完整与最新版请以 hello-rocm 为准。

本节介绍在 Ubuntu 24.04 + ROCm 7+ 环境下，使用 **vLLM** 部署并调用 **Qwen3.5-4B**。

当前 vLLM 官方文档推荐优先使用官方 ROCm Docker 镜像：

```text
vllm/vllm-openai-rocm:latest
```

旧的 `rocm/vllm`、`rocm/vllm-dev` 镜像已不再作为 vLLM 官方新文档的首选路径。Qwen3.5 架构较新，推荐使用官方镜像或较新的 vLLM 版本，避免旧版本不识别 `qwen3_5` 模型类型。

> 前置条件：已完成 [Ubuntu 24.04 + ROCm 7 环境准备](./1-env-prepare-ubuntu24-rocm7.md)。

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

---

### 3. 启动 Qwen3.5-4B 服务

```bash
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
  --model Qwen/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

服务启动后检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

---

### 4. 使用本地模型目录启动

如果已经提前下载好模型，可挂载本地模型目录：

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
  --model /app/models/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --max-model-len 4096 \
  --trust-remote-code
```

---

## 二、OpenAI-compatible API 调用

vLLM 默认提供 OpenAI-compatible API：

```text
http://127.0.0.1:8000/v1
```

### 1. curl 调用

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-4B",
    "messages": [
      {"role": "user", "content": "用一句话介绍 ROCm。"}
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 256,
    "extra_body": {
      "chat_template_kwargs": {"enable_thinking": false}
    }
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
    model="Qwen3.5-4B",
    messages=[{"role": "user", "content": "用一句话介绍深度学习。"}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(response.choices[0].message.content)
```

---

## 三、thinking mode 说明

Qwen3.5 默认支持 thinking mode。不同任务可使用不同设置：


| 场景               | 建议设置                           |
| ---------------- | ------------------------------ |
| 通用推理 / 数学 / 复杂问题 | `enable_thinking=true`         |
| 角色扮演 / 简短问答 / 演示 | `enable_thinking=false`        |
| API 对齐测试         | 固定采样参数，并显式传入 `enable_thinking` |


非 thinking 模式的常用采样参数：

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "max_tokens": 512
}
```

---

## 四、性能测试脚本（tokens/s）

```bash
RAND_PROMPT="随机码$(date +%N): 请详细介绍 ROCm 的用途，要求内容丰富，不要重复。"
start=$(date +%s.%N)

response=$(curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen3.5-4B\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$RAND_PROMPT\"}],
    \"max_tokens\": 512,
    \"temperature\": 0.7,
    \"extra_body\": {\"chat_template_kwargs\": {\"enable_thinking\": false}}
  }")

end=$(date +%s.%N)
content=$(echo "$response" | jq -r '.choices[0].message.content')
tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
duration=$(echo "$end - $start" | bc)

echo "==================== 原始内容 ===================="
echo "$content"
echo "=================================================="

if (( $(echo "$duration < 0.05" | bc -l) )); then
  echo "检测到异常极速响应 ($duration 秒)，可能命中了缓存。"
else
  tps=$(echo "scale=2; $tokens / $duration" | bc)
  echo "生成 Token 数: $tokens"
  echo "实际总耗时: $duration 秒"
  echo "真实推理速度: $tps tokens/s"
fi
```

---

## 五、方式二：ROCm wheel 安装（可选）

如果不使用 Docker，也可以安装 vLLM ROCm wheel。该方式对 Python / ROCm / glibc 版本要求更严格。

官方当前要求重点：

- Python 3.12
- ROCm 7.0 或 ROCm 7.2.1 对应 wheel
- glibc >= 2.35

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

安装完成后启动服务：

```bash
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --max-model-len 4096 \
  --trust-remote-code
```

> 注意：如果 Python 版本不匹配，安装器可能回退到 CUDA wheel，随后在 AMD GPU 上出现 `libcudart.so` 相关错误。

---

## 六、方式三：源码编译（进阶）

源码编译适合需要修改 vLLM、调试算子或适配特殊硬件的场景。普通部署不建议优先选择该方式。

官方 ROCm 构建文档：

- [https://docs.vllm.ai/en/stable/getting_started/installation/gpu/](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

查询 GPU 架构：

```bash
rocminfo | grep gfx
```

设置架构示例：

```bash
export PYTORCH_ROCM_ARCH="gfx1151"
```

具体依赖和构建步骤请以 vLLM 官方文档为准。

---

## 七、常见问题

### 1. 模型类型不识别

升级 vLLM / Transformers，或使用官方 `vllm/vllm-openai-rocm:latest` 镜像。Qwen3.5 架构较新，旧版本可能无法识别 `qwen3_5` 模型类型。

### 2. API 输出包含 thinking 内容

检查请求体中是否显式传入：

```json
{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}
```

### 3. 显存不足

降低 `--max-model-len`，例如从 4096 降到 2048；或减少并发请求。

### 4. 应该继续使用 `rocm/vllm-dev:nightly` 吗？

新文档优先使用 vLLM 官方镜像 `vllm/vllm-openai-rocm:latest`。旧 AMD 镜像只适合特定版本验证，不作为本教程主路径。

---

## 延伸学习

更完整的 ROCm 部署、微调与基础设施教程，请前往 Datawhale 项目 [hello-rocm](https://github.com/datawhalechina/hello-rocm)：

- 仓库：[https://github.com/datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm)
- 在线文档：[https://datawhalechina.github.io/hello-rocm/](https://datawhalechina.github.io/hello-rocm/)
- 部署专区原文：[https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy](https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy)

> 本文内容参考 hello-rocm `01-deploy` 教程，并按 self-llm `models_amd` 目录结构做了路径适配。

