#!/usr/bin/env bash
set -euo pipefail
out=/home/mccxadmin/self-llm-local26-results
docker run --rm --runtime=mthreads -e MTHREADS_VISIBLE_DEVICES=0 \
 -e "TASK_UID=$(id -u)" -e "TASK_GID=$(id -g)" -v "$out:/results" \
 registry.mthreads.com/mcctest/training-suite:v2.1.7 \
 bash -lc 'chown -R "$TASK_UID:$TASK_GID" /results; python /results/verify_artifacts.py' \
 2>&1 | tee /tmp/local26-artifact-check.log
