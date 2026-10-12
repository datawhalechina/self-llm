#!/usr/bin/env bash
set -uo pipefail
repo=/home/mccxadmin/self-llm-local26
log=/tmp/local26-gpu-smoke.log
docker run --rm --name self-llm-local26-validation \
  --runtime=mthreads -e MTHREADS_VISIBLE_DEVICES=0 \
  --label self-llm.task=LOCAL-26 --label self-llm.owner=dongyang \
  --shm-size=8g -v "$repo:/workspace/self-llm" \
  -w /workspace/self-llm \
  registry.mthreads.com/mcctest/training-suite:v2.1.7 \
  bash -lc 'set -euo pipefail
    bash scripts/musa/install.sh
    python scripts/musa/run.py scripts/musa/smoke.py --dtype float32
    python scripts/musa/run.py scripts/musa/smoke.py --dtype bf16
  ' 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
echo "GPU_CHECK_EXIT=$code" | tee -a "$log"
exit "$code"
