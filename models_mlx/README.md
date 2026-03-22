
本教程旨在帮助用户基于苹果原生框架[MLX-LM](https://github.com/ml-explore/mlx-lm) 部署和使用大模型，以便于充分利用 Apple M 系列芯片的性能进行本地推理。

## 🔧 环境配置

```bash
# 创建 Conda 虚拟环境
conda create -n mlx-lm python=3.11
conda activate mlx-lm
# 安装依赖
pip install -r requirements.txt
```

## 📁 项目结构

```
models_mlx/
├── run_app_gradio.py          # Gradio 交互式应用（模型下载 + 对话）
├── requirements.txt           # Python 依赖
├── configs/                   # 模型配置（JSON 格式，支持热加载）
│   └── model_info/
│       ├── mlx.json           #   MLX 量化模型列表
│       └── original.json      #   原始 HuggingFace 模型列表
├── modules/                   # 功能模块
│   └── download_model.py      #   模型下载模块（可独立运行）
├── models/                    # 下载的模型存放目录
├── notebooks/                 # Jupyter Notebook 教程
│   ├── Qwen3_MLX_部署与交互.ipynb
│   └── Qwen3_Transformers_部署与交互.ipynb
└── docs/                      # 文档
    └── MLX-LM_Intro.md        #   MLX 框架简介
```

## 📖 教程内容

### 理论部分

<table align="center">
  <tr>
    <td valign="top" width="25%">
      • <a href="./docs/MLX-LM_Intro.md">MLX框架简介</a><br>
    </td>
  </tr>
</table>

### Notebook 教程

| Notebook | 说明 |
|----------|------|
| [Qwen3_MLX_部署与交互](./notebooks/Qwen3_MLX_部署与交互.ipynb) | 使用 MLX 框架部署 Qwen3（Apple Silicon 推荐） |
| [Qwen3_Transformers_部署与交互](./notebooks/Qwen3_Transformers_部署与交互.ipynb) | 使用 Transformers 框架部署 Qwen3（通用兼容） |

### Gradio 交互应用

集模型下载与对话为一体的 Web 界面，支持：

- **模型下载**：按 公司 → 系列 → 模型 三级选择，自动检测本地是否已下载
- **模型对话**：支持 MLX / Transformers 双框架，MLX 支持流式输出
- **参数调节**：Temperature、Top-p、Max Tokens、思考模式
- **热加载**：修改 `configs/` 下的 JSON 配置后，刷新页面或点击刷新按钮即可生效

```bash
python run_app_gradio.py
```

### 命令行下载模型

也可以通过命令行交互式下载模型（无需启动 Gradio）：

```bash
python -m modules.download_model
```

## 🚀 支持模型

模型列表通过 `configs/` 目录下的 JSON 文件配置。

| 公司 | 系列 | 模型列表 |
|------|------|-------------------|
| Alibaba | QwQ | `QwQ-0.5B-4bit` |
| Alibaba | Qwen1.5 | `Qwen1.5-0.5B-Chat-4bit`<br>`Qwen1.5-1.8B-Chat-4bit`<br>`Qwen1.5-MoE-A2.7B-4bit`<br>`Qwen1.5-MoE-A2.7B-Chat-4bit` |
| Alibaba | Qwen2 | `Qwen2-0.5B-Instruct-4bit`<br>`Qwen2-1.5B-4bit`<br>`Qwen2-1.5B-Instruct-4bit` |
| Alibaba | Qwen2-Math | `Qwen2-Math-1.5B-Instruct-4bit` |
| Alibaba | Qwen2.5 | `Qwen2.5-0.5B-4bit`<br>`Qwen2.5-0.5B-Instruct-4bit`<br>`Qwen2.5-1.5B-4bit`<br>`Qwen2.5-1.5B-Instruct-4bit`<br>`Qwen2.5-3B-4bit`<br>`Qwen2.5-3B-Instruct-4bit` |
| Alibaba | Qwen2.5-Coder | `Qwen2.5-Coder-0.5B-4bit`<br>`Qwen2.5-Coder-0.5B-Instruct-4bit`<br>`Qwen2.5-Coder-1.5B-4bit`<br>`Qwen2.5-Coder-1.5B-Instruct-4bit`<br>`Qwen2.5-Coder-3B-4bit`<br>`Qwen2.5-Coder-3B-Instruct-4bit` |
| Alibaba | Qwen2.5-Math | `Qwen2.5-Math-1.5B-4bit`<br>`Qwen2.5-Math-1.5B-Instruct-4bit` |
| Alibaba | Qwen3 | `Qwen3-0.6B-4bit`<br>`Qwen3-0.6B-Base-4bit`<br>`Qwen3-1.7B-4bit` |
| Alibaba | Qwen3.5 | `Qwen3.5-0.8B-4bit`<br>`Qwen3.5-2B-4bit` |
| DeepSeek | DeepSeek-R1 | `DeepSeek-R1-Distill-Qwen-1.5B-4bit` |
| DeepSeek | DeepSeek-V3 | - |
| Google | Gemma-2 | `gemma-2-2b-4bit`<br>`gemma-2-2b-it-4bit`<br>`gemma-2-2b-jpn-it-4bit`<br>`gemma-2-baku-2b-it-4bit` |
| Google | Gemma-3 | `gemma-3-1b-it-4bit`<br>`gemma-3-1b-pt-4bit`<br>`gemma-3-270m-4bit`<br>`gemma-3-270m-it-4bit` |
| Meta | Llama-3.1 | - |
| Meta | Llama-3.2 | `Llama-3.2-1B-Instruct-4bit`<br>`Llama-3.2-3B-Instruct-4bit` |
| Meta | Llama-4 | - |
| Microsoft | Phi-2 | `phi-2-super-4bit` |
| Microsoft | Phi-4 | - |
| Mistral | Mistral | `Ministral-3-3B-Instruct-2512-4bit`<br>`Ministral-3-3B-Reasoning-2512-4bit` |
| Moonshot | Kimi | - |

如需添加新模型，编辑 `configs/model_info/mlx.json` 或 `configs/model_info/original.json` 即可。
