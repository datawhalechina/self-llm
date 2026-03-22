
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
│   └── model_info/
│       ├── mlx.json           #   MLX quantized model list
│       └── original.json      #   Original HuggingFace model list
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

Model lists are configured via JSON files in the `configs/` directory.

| Company | Series | Models |
|---------|--------|-----------------------|
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

To add new models, simply edit `configs/model_info/mlx.json` or `configs/model_info/original.json`.
