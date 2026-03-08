
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
│   ├── models_info_mlx.json   #   MLX 量化模型列表
│   └── models_info.json       #   原始 HuggingFace 模型列表
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

模型列表通过 `configs/` 目录下的 JSON 文件配置，当前支持：

| 公司 | 系列 |
|------|------|
| Alibaba | Qwen3、Qwen2.5、Qwen2.5-Coder |
| DeepSeek | DeepSeek-R1、DeepSeek-V3 |
| Meta | Llama-3.1、Llama-3.2、Llama-4 |
| Google | Gemma-2、Gemma-3 |
| Mistral | Mistral |
| Microsoft | Phi-4 |

如需添加新模型，编辑 `configs/models_info_mlx.json` 或 `configs/models_info.json` 即可。
