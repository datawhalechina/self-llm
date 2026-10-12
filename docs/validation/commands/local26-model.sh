#!/usr/bin/env bash
set -uo pipefail
repo=/home/mccxadmin/self-llm-local26
out=/home/mccxadmin/self-llm-local26-results
mkdir -p "$out"
printf '%s\n' '[{"instruction":"计算 1+1。","input":"","output":"1+1=2。"},{"instruction":"介绍 Python。","input":"","output":"Python 是一种常用的编程语言。"}]' > "$out/data.json"
docker run --rm --name self-llm-local26-model \
  --runtime=mthreads -e MTHREADS_VISIBLE_DEVICES=0 \
  --label self-llm.task=LOCAL-26 --label self-llm.owner=dongyang \
  --shm-size=8g -v "$repo:/workspace/self-llm" \
  -v /data/models:/models:ro -v "$out:/results" \
  -w /workspace/self-llm \
  registry.mthreads.com/mcctest/training-suite:v2.1.7 \
  bash -lc 'set -euo pipefail
    bash scripts/musa/install.sh
    python scripts/musa/run.py scripts/musa/qwen.py infer --model-dir /models/Qwen3-0.6B --dtype bf16 --max-new-tokens 32
    python scripts/musa/run.py scripts/musa/qwen.py lora --model-dir /models/Qwen3-0.6B --dtype bf16 --data-json /results/data.json --output-dir /results/qwen3-lora --max-steps 2
    python scripts/musa/run.py scripts/musa/qwen.py infer --model-dir /models/Qwen2-0.5B-Instruct --dtype bf16 --max-new-tokens 32
    python scripts/musa/run.py scripts/musa/qwen.py lora --model-dir /models/Qwen2-0.5B-Instruct --dtype bf16 --data-json /results/data.json --output-dir /results/qwen2-lora --max-steps 2
  ' 2>&1 | tee /tmp/local26-model-check.log
code=${PIPESTATUS[0]}
echo "MODEL_CHECK_EXIT=$code" | tee -a /tmp/local26-model-check.log
exit "$code"
