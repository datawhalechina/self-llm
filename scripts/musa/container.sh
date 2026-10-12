#!/usr/bin/env bash
set -euo pipefail

: "${GPU_INDICES:?Set physical GPU indices for this run}"
if [[ ! "$GPU_INDICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo 'GPU_INDICES must be comma-separated physical GPU indices' >&2
    exit 2
fi

allow_unleased="${GPU_ALLOW_UNLEASED:-0}"
if [[ "$allow_unleased" != 0 && "$allow_unleased" != 1 ]]; then
    echo 'GPU_ALLOW_UNLEASED must be 0 or 1' >&2
    exit 2
fi

if [[ "$allow_unleased" == 1 ]]; then
    container_suffix=unleased
else
    : "${GPU_LEASE_ID:?Set the active GPU lease id before starting the container}"
    if [[ ! "$GPU_LEASE_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo 'GPU_LEASE_ID contains invalid characters' >&2
        exit 2
    fi
    container_suffix="${GPU_LEASE_ID:0:8}"
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
name="agent-self-llm-LOCAL-26-$container_suffix"
docker_args=(
    docker run --rm -i
    --name "$name"
    --runtime=mthreads
    --env "MTHREADS_VISIBLE_DEVICES=$GPU_INDICES"
    --label "self-llm.owner=${GPU_OWNER:-dongyang}"
    --label self-llm.task=LOCAL-26
)
if [[ "$allow_unleased" == 1 ]]; then
    echo 'GPU lease checks bypassed by explicit GPU_ALLOW_UNLEASED=1' >&2
else
    docker_args+=(
        --label serverlist.managed=1
        --label "serverlist.lease_id=$GPU_LEASE_ID"
        --label "serverlist.owner=${GPU_OWNER:-dongyang}"
        --label serverlist.ticket=LOCAL-26
    )
fi
if [[ -n "${MUSA_MODELS_ROOT:-}" ]]; then
    if [[ ! -d "$MUSA_MODELS_ROOT" ]]; then
        echo 'MUSA_MODELS_ROOT must be an existing host directory' >&2
        exit 2
    fi
    docker_args+=(--mount "type=bind,src=$MUSA_MODELS_ROOT,dst=/models,readonly")
fi
docker_args+=(
    --shm-size=8g
    --mount "type=bind,src=$repo_root,dst=/workspace/self-llm"
    --workdir /workspace/self-llm
    registry.mthreads.com/mcctest/training-suite:v2.1.7
    "$@"
)
exec "${docker_args[@]}"
