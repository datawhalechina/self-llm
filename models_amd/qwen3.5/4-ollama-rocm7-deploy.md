## Ollama 零基础环境部署（Ubuntu 24.04 + ROCm 7+）

> 内容参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm) `docs/zh/01-deploy`，适配 self-llm 目录结构。完整与最新版请以 hello-rocm 为准。

本节介绍在 Ubuntu 24.04 + ROCm 7+ 环境下，如何安装和使用 **Ollama（ROCm 版 llama.cpp 后端）**运行 Qwen3.5 系列模型。

> 前置条件：已完成 [Ubuntu 24.04 环境准备](./1-env-prepare-ubuntu24-rocm7.md)。

---

### 1. 安装 Ollama（系统服务）

一键安装到 `systemctl` 中管理，默认占用本地 `11434` 端口启动服务。

更多信息参考官方文档：[https://docs.ollama.com/linux](https://docs.ollama.com/linux)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

### 2. 验证服务连通性

安装完成后，可以用以下命令验证服务是否正常运行：

```bash
curl http://localhost:11434
```

若返回版本号或服务信息，说明 Ollama 服务已启动。

---

### 3. 准备 Qwen3.5 模型

Ollama 的可用模型标签会随官方模型库更新而变化。若 Ollama 官方库已经提供 Qwen3.5 标签，可直接使用对应标签：

```bash
# 示例：请以 ollama 官方模型库实际标签为准 https://ollama.com/library/qwen3.5
ollama pull qwen3.5:4b
ollama run qwen3.5:4b
```

如果官方库尚未提供合适标签，可基于本地 GGUF 文件创建模型：

```bash
mkdir -p ~/models/qwen3.5
# 将 Qwen3.5 GGUF 文件放入 ~/models/qwen3.5/
```

创建 `Modelfile`：

```text
FROM ./qwen3.5-4b-q4_k_m.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.8
```

创建并运行本地模型：

```bash
cd ~/models/qwen3.5
ollama create qwen3.5-4b-local -f Modelfile
ollama run qwen3.5-4b-local
```

---

### 4. 使用 curl 测速（计算 tokens/s）

下面示例调用 Ollama REST 接口，并使用 `jq` 与 `awk` 解析返回中的评估信息，计算推理速度。

```bash
curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen3.5-4b-local",
  "prompt": "用一句话解释什么是大语言模型",
  "stream": false
}' | jq '.eval_count, .eval_duration' | \
awk 'NR==1{count=$1} NR==2{duration=$1/1e9} END{printf "tokens/s: %.2f\n", count/duration}'
```

- `eval_count`：推理产生的 token 数量
- `eval_duration`：推理耗时，单位为纳秒
- `tokens/s`：通过 `count / (duration / 1e9)` 计算得到

---

---

## 延伸学习

更完整的 ROCm 部署、微调与基础设施教程，请前往 Datawhale 项目 [hello-rocm](https://github.com/datawhalechina/hello-rocm)：

- 仓库：[https://github.com/datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm)
- 在线文档：[https://datawhalechina.github.io/hello-rocm/](https://datawhalechina.github.io/hello-rocm/)
- 部署专区原文：[https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy](https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy)

> 本文内容参考 hello-rocm `01-deploy` 教程，并按 self-llm `models_amd` 目录结构做了路径适配。

