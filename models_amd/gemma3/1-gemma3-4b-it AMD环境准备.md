# gemma-3-4b-it AMD环境准备

## 环境准备

本文基础环境如下：

```
----------------
Windows11
CPU AI 395
内存 128G
----------------
```

> 非常感谢 AMD University Program 对本开源项目的大力支持，本项目的环境都在此主机下完成

## 芯片介绍

AMD Strix Halo处理器可以说是一款划时代的产品，尤其是旗舰型号锐龙 AI Max+ 395，拥有史上最强集显，可以轻松媲美桌面级RTX 4060独立显卡。全新"Zen5"架构 CPU、RDNA3.5 架构 GPU、XDNA2架构 NPU，其中NPU AI引擎拥有高达50 TOPS的算力。锐龙 AI Max系列可以搭配最多128GB LPDDR5X-8000统一内存，带宽高达256GB/s，分配最多96GB作为专属显存，从而能在本地直接运行例如GPT-OSS-120B这种千亿参数的完整大模型，尤其是对于MoE专家模型可谓得天独厚。

## NPU 安装与配置

### 先决条件

Ryzen AI 软件支持带有神经网络处理单元（NPU）的 AMD 处理器。请参考发布说明以获取完整的支持配置列表。

在安装 Ryzen AI 软件之前，系统必须安装以下依赖项：

| 依赖项 | 版本要求 |
|--------|---------|
| Windows 11 | build >= 22621.3527 |
| Visual Studio | 2022 |
| cmake | version >= 3.26 |
| Python 发行版（推荐 Miniforge） | 最新版本 |

⚠ **重要提示**：

* Visual Studio 2022 Community：确保安装了"使用 C++ 的桌面开发"工作负载
* Miniforge：确保在系统 PATH 环境变量中设置以下路径之一：
  - `path\to\miniforge3\condabin`
  - `path\to\miniforge3\Scripts\`
  - `path\to\miniforge3\`
  
  （系统 PATH 变量应在"环境变量"窗口的"系统变量"部分设置），安装程序请都是用管理员权限打开使用！！！！

### 安装 NPU 驱动程序

1. **下载 NPU 驱动程序**
   - 下载并安装 NPU 驱动程序版本：32.0.203.280 或更新版本
   - 下载链接：
     * [NPU Driver (Version 32.0.203.280)](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)
     * [NPU Driver (Version 32.0.203.304)](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)

2. **安装步骤**
   - 解压下载的 ZIP 文件
   - 以管理员模式打开终端
   - 执行 `.\npu_sw_installer.exe` 文件

3. **验证安装**
   - 打开任务管理器 -> 性能 -> NPU0
   - 确保 NPU MCDM 驱动程序已正确安装：
     * 版本：32.0.203.280，日期：5/16/2025
     * 或版本：32.0.203.304，日期：10/07/2025

### 安装 Ryzen AI 软件

1. **下载安装程序**
   - 下载 Ryzen AI 软件安装程序：`ryzenai-lt-1.6.1.exe`

2. **运行安装向导**
   - 启动 EXE 安装程序并按照安装向导的说明操作：
     * 接受许可协议条款
     * 提供 Ryzen AI 安装的目标文件夹（默认：`C:\Program Files\RyzenAI\1.6.1`）
     * 指定 conda 环境的名称（默认：`ryzen-ai-1.6.1`）

3. **完成安装**
   - Ryzen AI 软件包现在已安装在安装程序创建的 conda 环境中

> **注意**：NuGet 包可在 [ryzen-ai-1.6.1-nuget.zip](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers) 下载

### 测试安装

Ryzen AI 软件安装文件夹包含用于验证软件是否正确安装的测试。此安装测试位于 `quicktest` 子文件夹中。

1. **打开 Conda 命令提示符**
   - 在 Windows 开始菜单中搜索"Miniforge Prompt"

2. **激活 Conda 环境**
   ```bash
   conda activate <env_name>
   ```
   其中 `<env_name>` 是安装程序创建的 conda 环境名称（默认为 `ryzen-ai-1.6.1`）

3. **运行测试**
   ```bash
   cd %RYZEN_AI_INSTALLATION_PATH%/quicktest
   python quicktest.py
   ```

4. **验证结果**
   - `quicktest.py` 脚本会设置环境并运行一个简单的 CNN 模型
   - 成功运行时，您将看到类似以下的输出，这表明模型正在 NPU 上运行，并且 Ryzen AI 软件的安装成功：
   ```
   [Vitis AI EP] No. of Operators :   NPU   398 VITIS_EP_CPU     2
   [Vitis AI EP] No. of Subgraphs :   NPU     1 Actually running on NPU     1
   Test Passed
   ```

> **注意**：Ryzen AI 软件安装文件夹的完整路径存储在 `RYZEN_AI_INSTALLATION_PATH` 环境变量中。

---
