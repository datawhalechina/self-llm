#!/usr/bin/env bash
set -euo pipefail

export LAB_ROOT="${LAB_ROOT:-/root/autodl-fs/experiments/minimax-m3-8gpu}"
export MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/MiniMax-M3-BF16}"
export CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/root/miniconda3/envs/minimax-m3-bf16-lora}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/.cache/conda/pkgs}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/miniconda3/envs}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.cache/uv}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [ -f "$LAB_ROOT/secrets/.env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$LAB_ROOT/secrets/.env.local"
  set +a
fi

if [ ! -d "$CONDA_ENV_PREFIX" ]; then
  echo "Conda environment not found: $CONDA_ENV_PREFIX" >&2
  return 1 2>/dev/null || exit 1
fi

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  . "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_PREFIX"
else
  export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
fi
