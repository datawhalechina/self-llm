# AMD GPU 平台支持模型列表

> 本页面专门收录了在 AMD GPU 平台上经过测试和验证的大语言模型部署教程。我们为每个支持的模型提供了详细的 AMD 环境配置指南、部署步骤和优化建议。所有教程均在实际 AMD 硬件环境中验证通过，确保能够在 AMD 平台上顺利运行。

## AMD 硬件环境支持

目前教程主要支持以下 AMD 硬件平台：
- **AMD Ryzen AI 300 系列**：AI Max+ 395、AI Max 370、AI Max 385
- **AMD Radeon RX 系列**：RX 7900 XTX、RX 7900 XT、RX 6900 XT 等
- **AMD Instinct 计算卡**：MI210、MI250、MI300 系列

## 目录

- [谷歌 Gemma3](#谷歌-gemma3)
- [Qwen3](#qwen3)

## 已支持模型列表

### 谷歌 Gemma3

[Gemma3](https://huggingface.co/google/gemma-3-4b-it)

- [x] [gemma3-4b-it AMD 环境准备](./models_amd/gemma3/1-gemma3-4b-it%20AMD环境准备.md) @陈榆
- [x] [gemma3-4b-it AMD 模型服务部署](./models_amd/gemma3/2-gemma3-4b-it%20模型服务部署.md) @陈榆

### Qwen3

[Qwen3](https://github.com/QwenLM/Qwen3)

- [x] [Qwen3-8B AMD部署调用](./models_amd/qwen3/1-Qwen3-8B-AMD部署调用.md) @陈榆

## AMD 环境配置通用指南

### 1. 系统要求

**操作系统：**
- Windows 11 64-bit（推荐）
- Linux Ubuntu 20.04+（部分支持）

**硬件要求：**
- AMD Ryzen AI 300 系列或更新处理器
- 最低 16GB 内存，推荐 32GB+
- 存储：至少 50GB 可用空间

### 2. 驱动安装

**AMD Ryzen AI NPU 驱动：**
- 下载并安装最新的 AMD Ryzen AI 软件包
- 确保 NPU 驱动正确安装和识别

**AMD GPU 驱动：**
- 安装 AMD Software: Adrenalin Edition
- 安装 ROCm 平台（Linux 环境）

### 3. 软件环境

**Python 环境：**
```bash
# 推荐使用 Python 3.9+
conda create -n amd_llm python=3.9
conda activate amd_llm

# 更换 pypi 源加速安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**核心依赖：**
- torch (AMD ROCm 版本)
- transformers
- huggingface_hub
- lemonade-server (Ryzen AI 平台专用)

## 性能优化建议

### 1. 内存优化
- 利用统一内存架构，合理分配系统内存和显存
- 对于大模型，建议使用 32GB+ 内存配置

### 2. NPU 加速
- 在支持的硬件上启用 NPU 推理加速
- 使用 lemonade-server SDK 获得最佳性能

### 3. 模型量化
- 使用 INT4/INT8 量化减少内存占用
- 在保证精度的前提下提升推理速度

## 常见问题

### Q: 如何检查我的 AMD 设备是否被正确识别？
A: 可以使用以下命令检查硬件支持情况：
```bash
# 检查 NPU 设备
python -c "import lemonade; print(lemonade.get_device_info())"

# 检查 GPU 设备（ROCm）
rocm-smi
```

### Q: 如何贡献新的 AMD 模型教程？
A: 欢迎提交 PR 到本仓库，我们特别期待：
- 更多 AMD GPU 型号的支持教程
- Linux ROCm 环境的部署指南
- 性能优化和基准测试结果

> 💡 **提示：** 本教程系列正在持续更新中，如果您有特定 AMD 平台的模型部署需求或建议，欢迎通过 Issue 或 PR 与我们联系。