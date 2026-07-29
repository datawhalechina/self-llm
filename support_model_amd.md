# AMD GPU 平台支持模型列表

<p align="center">
  <img src="./images/rocm_logo.png" height="56" alt="ROCm"/>
  &nbsp;&nbsp;&nbsp;
  <img src="./images/aup_logo.png" height="56" alt="AMD University Program"/>
</p>

<p align="center">
  <em>感谢 AMD University Program（AMD 大学合作部）对本项目的支持</em>
</p>

> 本页面收录在 AMD GPU（ROCm）平台上验证的大语言模型部署教程。教程基于 Ubuntu 24.04 + ROCm 7 实践整理，覆盖环境准备、LM Studio、vLLM、Ollama、llama.cpp 等常见部署路径。所有内容参考 Datawhale [hello-rocm](https://github.com/datawhalechina/hello-rocm) 的 `01-deploy` 专区并适配本仓库目录。

> 💡 **想学更多？** 完整的 ROCm 部署、微调与基础设施教程见：[hello-rocm](https://github.com/datawhalechina/hello-rocm) · [在线文档](https://datawhalechina.github.io/hello-rocm/)

## AMD 硬件环境支持

目前教程主要面向以下 AMD 硬件（需支持 ROCm）：

- **AMD Ryzen AI MAX / AI 300 系列**：如 AI Max+ 395（gfx1151）等
- **AMD Radeon RX 系列**：RX 7000 / 9000 系列等
- **AMD Instinct 计算卡**：MI 系列等

具体型号请以 [ROCm 官方支持列表](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) 为准。

## 实践算力支持（AMD 云平台）

感谢 **AMD University Program（AMD 大学合作部）** 提供的云算力支持。无本地 AMD GPU 时，可优先使用以下免费 / 学习用云平台完成 ROCm 环境验证与本仓库模型部署实践：

| 平台 | 定位 | 入口 |
|---|---|---|
| **AUP Learning Cloud（推荐）** | AMD 大学合作部学习云，GitHub 授权登录，适合正式训练与 Notebook 调试 | [tpe.aupcloud.io](https://tpe.aupcloud.io) |
| **AMD Radeon Cloud（备用）** | AMD AI 开发者计划中文站云算力，适合快速启动现成 ROCm 模板做验证 | [AMD 开发者云](https://developer.amd.com.cn/login?source=91kadjjnI) |

📖 **完整使用指南：** [AMD ROCm 云平台使用指南（AUP Learning Cloud 优先）](./models_amd/README_00_AMD_AUP免费云平台使用指南.md)

> 额度、镜像、硬件与授权方式以各平台当前页面及管理员通知为准。建议先走 AUP Learning Cloud 完成主流程，再用 Radeon Cloud 做快速验证。

## 目录

- [实践算力支持（AMD 云平台）](#实践算力支持amd-云平台)
- [谷歌 Gemma4](#谷歌-gemma4)
- [Qwen3.5](#qwen35)

## 已支持模型列表

### 谷歌 Gemma4

[Gemma 4](https://huggingface.co/collections/google/gemma-4)（示例以 `gemma-4-E4B-it` 为主）

- [Ubuntu 24.04 + ROCm 7 环境准备](./models_amd/gemma4/1-env-prepare-ubuntu24-rocm7.md)
- [Gemma 4 模型介绍](./models_amd/gemma4/2-gemma4_model.md)
- [LM Studio 部署](./models_amd/gemma4/3-lm-studio-rocm7-deploy.md)
- [vLLM 部署](./models_amd/gemma4/4-vllm-rocm7-deploy.md)
- [Ollama 部署](./models_amd/gemma4/5-ollama-rocm7-deploy.md)
- [llama.cpp 部署](./models_amd/gemma4/6-llamacpp-rocm7-deploy.md)

### Qwen3.5

[Qwen3.5](https://github.com/QwenLM/Qwen3.5)

- [Ubuntu 24.04 + ROCm 7 环境准备](./models_amd/qwen3.5/1-env-prepare-ubuntu24-rocm7.md)
- [LM Studio 部署](./models_amd/qwen3.5/2-lm-studio-rocm7-deploy.md)
- [vLLM 部署](./models_amd/qwen3.5/3-vllm-rocm7-deploy.md)
- [Ollama 部署](./models_amd/qwen3.5/4-ollama-rocm7-deploy.md)
- [llama.cpp 部署](./models_amd/qwen3.5/5-llamacpp-rocm7-deploy.md)

## AMD / ROCm 环境配置通用指南

### 1. 系统要求

**操作系统：**

- Linux Ubuntu 24.04（推荐，教程主路径）
- Windows 11（部分框架 / pip 路线可用，详见各篇环境准备）

**硬件要求：**

- 支持 ROCm 的 AMD GPU（建议显存 8GB+；更大模型请参考对应教程）
- 最低 16GB 系统内存，推荐 32GB+
- 存储：至少 50GB 可用空间

### 2. 驱动与 ROCm

- 安装 / 升级至 **ROCm 7.x**（教程以 ROCm 7.13 / TheRock 体系为参考）
- 使用 `rocminfo` / `rocm-smi` / `amd-smi` 验证 GPU 与驱动状态
- 详细步骤见各模型目录下的「环境准备」文档

### 3. 软件环境

**Python 环境（示例）：**

```bash
# 推荐 Python 3.10+（部分路线使用 3.13）
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate

# 或使用 conda / venv
# conda create -n amd_llm python=3.13
# conda activate amd_llm
```

**常见推理栈：**

- PyTorch（ROCm 版本）
- vLLM（ROCm Docker / 源码）
- Ollama / llama.cpp（ROCm 后端）
- LM Studio（ROCm 版 llama.cpp 后端）

## 推荐学习路径

1. 完成对应模型的 **环境准备**
2. 零基础可先走 **LM Studio** 或 **Ollama**
3. 需要高吞吐服务再看 **vLLM**
4. 需要命令行 / REST 精细控制再看 **llama.cpp**
5. 进阶内容（微调、基础设施、更多模型）前往 [hello-rocm](https://github.com/datawhalechina/hello-rocm)

## 常见问题

### Q: 如何确认 AMD GPU 是否支持 ROCm？

A: 参考 [ROCm 官方支持列表](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)，并在本机执行：

```bash
rocminfo
rocm-smi
# 或
amd-smi
```

### Q: 部署时遇到 HIP error 怎么办？

A:

1. 确认 ROCm 已正确安装
2. 检查 `PATH` / `LD_LIBRARY_PATH` / `ROCM_PATH` 等环境变量
3. 确认用户已加入 `render` / `video` 组后重新登录或重启

### Q: 如何贡献新的 AMD 模型教程？

A: 欢迎向本仓库提交 PR；更系统的 ROCm 教程也欢迎贡献到 [hello-rocm](https://github.com/datawhalechina/hello-rocm)。特别期待：

- 更多 AMD GPU 型号的实测记录
- 性能优化与基准测试结果
- 新模型在 ROCm 上的部署路径

> 💡 **提示：** AMD / ROCm 生态演进较快，若本仓库文档与最新实践不一致，请优先对照 [hello-rocm](https://github.com/datawhalechina/hello-rocm) 与 [ROCm 官方文档](https://rocm.docs.amd.com/)。

