
This tutorial helps users deploy and run large language models locally on Apple Silicon Macs using the native [MLX-LM](https://github.com/ml-explore/mlx-lm) framework.

## 🔧 Environment Setup

```bash
# Create Conda virtual environment
conda create -n mlx-lm python=3.11
conda activate mlx-lm
# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
models_mlx/
├── run_app_gradio.py          # Gradio interactive app (download + chat)
├── requirements.txt           # Python dependencies
├── configs/                   # Model configs (JSON, hot-reloadable)
│   ├── models_info_mlx.json   #   MLX quantized model list
│   └── models_info.json       #   Original HuggingFace model list
├── modules/                   # Functional modules
│   └── download_model.py      #   Model download module (standalone runnable)
├── models/                    # Downloaded model storage
├── notebooks/                 # Jupyter Notebook tutorials
│   ├── Qwen3_MLX_部署与交互.ipynb
│   └── Qwen3_Transformers_部署与交互.ipynb
└── docs/                      # Documentation
    └── MLX-LM_Intro.md        #   MLX framework introduction
```

## 📖 Tutorials

### Theory

<table align="center">
  <tr>
    <td valign="top" width="25%">
      • <a href="./docs/MLX-LM_Intro.md">MLX Framework Introduction</a><br>
    </td>
  </tr>
</table>

### Notebook Tutorials

| Notebook | Description |
|----------|-------------|
| [Qwen3_MLX_部署与交互](./notebooks/Qwen3_MLX_部署与交互.ipynb) | Deploy Qwen3 with MLX (recommended for Apple Silicon) |
| [Qwen3_Transformers_部署与交互](./notebooks/Qwen3_Transformers_部署与交互.ipynb) | Deploy Qwen3 with Transformers (universal compatibility) |

### Gradio Interactive App

An all-in-one web interface for model downloading and chatting:

- **Model Download**: Three-level selection (Company → Series → Model) with local existence detection
- **Model Chat**: Supports both MLX and Transformers backends; MLX supports streaming output
- **Parameter Tuning**: Temperature, Top-p, Max Tokens, Thinking mode
- **Hot Reload**: Edit JSON configs in `configs/`, then refresh the page or click the refresh button

```bash
python run_app_gradio.py
```

### CLI Model Download

You can also download models via an interactive command-line interface (no Gradio needed):

```bash
python -m modules.download_model
```

## 🚀 Supported Models

Model lists are configured via JSON files in the `configs/` directory. Currently supported:

| Company | Series |
|---------|--------|
| Alibaba | Qwen3, Qwen2.5, Qwen2.5-Coder |
| DeepSeek | DeepSeek-R1, DeepSeek-V3 |
| Meta | Llama-3.1, Llama-3.2, Llama-4 |
| Google | Gemma-2, Gemma-3 |
| Mistral | Mistral |
| Microsoft | Phi-4 |

To add new models, simply edit `configs/models_info_mlx.json` or `configs/models_info.json`.
