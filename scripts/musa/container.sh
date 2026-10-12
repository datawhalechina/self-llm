#!/usr/bin/env bash
set -euo pipefail

: "${GPU_LEASE_ID:?Set the active GPU lease id before starting the container}"
: "${GPU_INDICES:?Set physical GPU indices returned by gpu_acquire}"
if [[ ! "$GPU_INDICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo 'GPU_INDICES must be comma-separated physical GPU indices' >&2
    exit 2
fi
if [[ ! "$GPU_LEASE_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'GPU_LEASE_ID contains invalid characters' >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    set -- bash
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
name="agent-self-llm-LOCAL-26-${GPU_LEASE_ID:0:8}"
exec docker run --rm -i \
    --name "$name" \
    --runtime=mthreads \
    --env "MTHREADS_VISIBLE_DEVICES=$GPU_INDICES" \
    --label serverlist.managed=1 \
    --label "serverlist.lease_id=$GPU_LEASE_ID" \
    --label "serverlist.owner=${GPU_OWNER:-dongyang}" \
    --label serverlist.ticket=LOCAL-26 \
    --shm-size=8g \
    --mount "type=bind,src=$repo_root,dst=/workspace/self-llm" \
    --workdir /workspace/self-llm \
    registry.mthreads.com/mcctest/training-suite:v2.1.7 \
    "$@"
