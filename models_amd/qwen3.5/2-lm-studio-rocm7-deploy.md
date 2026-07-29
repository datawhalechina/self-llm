## LM Studio 零基础大模型部署（Ubuntu 24.04 + ROCm 7+）

> 内容参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm) `docs/zh/01-deploy`，适配 self-llm 目录结构。完整与最新版请以 hello-rocm 为准。

本节介绍如何在 Ubuntu 24.04 上，基于 **ROCm 7+** 使用 **LM Studio + ROCm 版 llama.cpp 后端**部署 Qwen3.5 系列模型。

> 前置条件：已完成 [Ubuntu 24.04 环境准备](./1-env-prepare-ubuntu24-rocm7.md)，并确认 ROCm 与 GPU 可正常使用。

---

### 1. 使用 LM Studio（选择 ROCm 版本 llama.cpp 后端推理）

#### 1.1 下载 LM Studio AppImage

首先从官网下载安装包：

```bash
https://lmstudio.ai/
```

下载最新的 `.AppImage` 文件到本地。

示意图（LM Studio 官网下载页面，可复用 Qwen3 文档截图）：



---

#### 1.2 解压 AppImage

提取 AppImage 内容并解压到 `squashfs-root` 目录：

```bash
chmod u+x LM-Studio-*.AppImage
./LM-Studio-*.AppImage --appimage-extract
```

---

#### 1.3 修复 chrome-sandbox 权限

进入 `squashfs-root` 目录，并为 `chrome-sandbox` 设置权限：

```bash
cd squashfs-root
sudo chown root:root chrome-sandbox
sudo chmod 4755 chrome-sandbox
```

---

#### 1.4 启动 LM Studio

```bash
./lm-studio
```

---

### 2. 安装 ROCm 版本 llama.cpp 后端推理

在 LM Studio 中选择 **ROCm 版本的 llama.cpp 后端**安装：



LM Studio 所提供的 ROCm 版本 llama.cpp 后端会列出不同 GPU 架构支持情况。下方截图为通用后端说明，可复用：



---

### 3. 加载 Qwen3.5 模型

LM Studio 通常使用 GGUF 格式模型。请在 LM Studio 的模型搜索或本地导入页面中选择适合当前硬件的 Qwen3.5 GGUF 模型。

建议：

- 显存较小：优先选择 Q4_K_M 或相近量化版本。
- 显存充足：可尝试更高精度量化版本。
- 如需关闭 thinking 输出，请在提示词或 API 调用侧显式说明输出要求。



---

### 4. Qwen3.5 性能示例

Qwen3.5-4B 的性能会受到模型量化格式、上下文长度、GPU 架构、ROCm 后端版本等因素影响。建议记录以下信息：

- 模型名称与量化格式
- 上下文长度
- GPU 型号与架构（如 `gfx1151`）
- tokens/s



---

## 延伸学习

更完整的 ROCm 部署、微调与基础设施教程，请前往 Datawhale 项目 [hello-rocm](https://github.com/datawhalechina/hello-rocm)：

- 仓库：[https://github.com/datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm)
- 在线文档：[https://datawhalechina.github.io/hello-rocm/](https://datawhalechina.github.io/hello-rocm/)
- 部署专区原文：[https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy](https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy)

> 本文内容参考 hello-rocm `01-deploy` 教程，并按 self-llm `models_amd` 目录结构做了路径适配。

