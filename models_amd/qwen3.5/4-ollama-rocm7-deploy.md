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

### 5. 可选：用公开网页来源回答问题

本地模型自带的知识有时间限制。涉及公开资料时，可以让已部署的模型通过 Ollama 的工具调用发起一次网页搜索，再根据返回的摘要回答。示例会列出搜索结果的原始链接，方便核对。

保持 Ollama 服务运行。在本教程目录下，用环境准备章节安装的 `uv` 和 Python 3.13 为 [示例脚本](./ollama_web_search.py) 创建独立的虚拟环境：

```bash
uv venv --python 3.13 .venv-search
source .venv-search/bin/activate
uv pip install 'mcp==1.30.0' 'httpx==0.28.1'
ollama list
read -r -p '本地模型名称：' MODEL
python ollama_web_search.py --model "$MODEL" --web-search \
  '请搜索 NASA 官网，找出欧罗巴快船（Europa Clipper）的发射日期，并注明来源。'
```

请选择第 3 节下载的本地模型；`ollama list` 也可能列出云端模型。启用 `--web-search` 后，模型生成的搜索目标和关键词会发送到 [Parallel MCP](https://docs.parallel.ai/integrations/mcp/installation)，无需 API 密钥。请只输入可以公开搜索的问题。去掉这个参数并使用本地模型，推理就留在本机。需要在 Ollama 服务上禁用云端功能时，请参考[官方设置](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features)。

可用 NASA 的[任务公告](https://www.nasa.gov/news-release/liftoff-nasas-europa-clipper-sails-toward-ocean-moon-of-jupiter)核对发射日期（2024 年 10 月 14 日）。脚本通过本地 `/api/chat` 接收工具调用，把摘要交回模型，并单独显示原始链接。

---

## 延伸学习

更完整的 ROCm 部署、微调与基础设施教程，请前往 Datawhale 项目 [hello-rocm](https://github.com/datawhalechina/hello-rocm)：

- 仓库：[https://github.com/datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm)
- 在线文档：[https://datawhalechina.github.io/hello-rocm/](https://datawhalechina.github.io/hello-rocm/)
- 部署专区原文：[https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy](https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy)

> 本文内容参考 hello-rocm `01-deploy` 教程，并按 self-llm `models_amd` 目录结构做了路径适配。

