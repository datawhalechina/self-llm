#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="${LAB_ROOT:-/root/autodl-fs/experiments/minimax-m3-8gpu}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-256}"
SESSION="${TMUX_SESSION:-minimax-m3-bf16-${RUN_ID}}"
ENV_PREFIX="${CONDA_ENV_PREFIX:-/root/miniconda3/envs/minimax-m3-bf16-lora}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/MiniMax-M3-BF16}"
DATA_PATH="${DATA_PATH:-$LAB_ROOT/data/tiny_qa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAB_ROOT/output/minimax-m3-bf16-zero3-lora-${RUN_ID}}"
LOG="$LAB_ROOT/logs/minimax-m3-bf16-zero3-lora-${RUN_ID}.log"
TERMINAL_STATUS="$OUTPUT_DIR/.terminal_status"
NOTIFY_ENV="$LAB_ROOT/secrets/serverchan.env"
NOTIFY_SCRIPT="$LAB_ROOT/bin/send_serverchan.py"

if [ "${TMUX_LAUNCHED:-0}" != "1" ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 3
  fi
  tmux new-session -d -s "$SESSION" \
    "TMUX_LAUNCHED=1 LAB_ROOT=$LAB_ROOT RUN_ID=$RUN_ID MAX_STEPS=$MAX_STEPS NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS MAX_LENGTH=$MAX_LENGTH TMUX_SESSION=$SESSION CONDA_ENV_PREFIX=$ENV_PREFIX MODEL_DIR=$MODEL_DIR DATA_PATH=$DATA_PATH OUTPUT_DIR=$OUTPUT_DIR bash $LAB_ROOT/train/run_m3_bf16_zero3_tmux.sh"
  echo "session=$SESSION"
  echo "log=$LOG"
  echo "output_dir=$OUTPUT_DIR"
  exit 0
fi

. "$LAB_ROOT/bin/source_env.sh"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LAB_ROOT/logs" "$OUTPUT_DIR"
rm -f "$TERMINAL_STATUS"
set +e
torchrun --standalone --nproc_per_node=8 \
  "$LAB_ROOT/train/finetune_m3_bf16_zero3.py" \
  --model "$MODEL_DIR" \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --deepspeed "$LAB_ROOT/train/deepspeed_zero3_cpu_offload.json" \
  --max-length "$MAX_LENGTH" \
  --max-steps "$MAX_STEPS" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --project "${FT_SWANLAB_PROJECT:-minimax-m3-prep}" \
  --experiment "minimax-m3-bf16-zero3-lora-${RUN_ID}" \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$TERMINAL_STATUS"

if [ -f "$NOTIFY_ENV" ] && [ -f "$NOTIFY_SCRIPT" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$NOTIFY_ENV"
  set +a
  if [ -n "${SERVERCHAN_SENDKEY:-}" ]; then
    if [ "$status" -eq 0 ]; then
      title="MiniMax M3 ZeRO-3 微调成功"
      desp="run=${RUN_ID}\noutput=${OUTPUT_DIR}\nlog=${LOG}"
    else
      title="MiniMax M3 ZeRO-3 微调失败"
      desp="run=${RUN_ID}\nexit_code=${status}\nlog=${LOG}\n\n最后日志：\n\n$(tail -n 30 "$LOG")"
    fi
    python "$NOTIFY_SCRIPT" --title "$title" --desp "$desp" --noip || true
  fi
fi
exit "$status"
