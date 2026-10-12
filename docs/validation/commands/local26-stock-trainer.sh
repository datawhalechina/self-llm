#!/usr/bin/env bash
set -uo pipefail
cd /home/mccxadmin/self-llm-local26
GPU_ALLOW_UNLEASED=1 GPU_INDICES=0 MUSA_MODELS_ROOT=/data/models bash scripts/musa/container.sh bash -lc '
  set -euo pipefail
  bash scripts/musa/install.sh
  python scripts/musa/run.py scripts/musa/tests/stock_trainer_gpu.py
' 2>&1 | tee /tmp/local26-stock-trainer.log
code=${PIPESTATUS[0]}
echo "STOCK_TRAINER_CHECK_EXIT=$code" | tee -a /tmp/local26-stock-trainer.log
exit "$code"
