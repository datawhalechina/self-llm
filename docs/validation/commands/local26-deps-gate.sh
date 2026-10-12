#!/usr/bin/env bash
set -o pipefail
docker run --rm --runtime=runc -e MTHREADS_VISIBLE_DEVICES=none \
  --label serverlist.owner=dongyang --label serverlist.ticket=LOCAL-26 \
  --mount type=bind,src=/home/mccxadmin/self-llm-local26,dst=/workspace/self-llm \
  --workdir /workspace/self-llm \
  registry.mthreads.com/mcctest/training-suite:v2.1.7 bash -lc '
    set -e
    python scripts/musa/check_dependencies.py snapshot /tmp/pip-baseline.json
    python -m pip install --no-deps torchada
    python scripts/musa/check_dependencies.py compare /tmp/pip-baseline.json
    echo REAL_IMAGE_GATE_EXIT=0
    mkdir -p /tmp/conflict-fixture/local26_conflict_fixture-1.0.dist-info
    printf "Metadata-Version: 2.1\nName: local26-conflict-fixture\nVersion: 1.0\nRequires-Dist: local26-definitely-missing-package\n" > /tmp/conflict-fixture/local26_conflict_fixture-1.0.dist-info/METADATA
    set +e
    PYTHONPATH=/tmp/conflict-fixture python scripts/musa/check_dependencies.py compare /tmp/pip-baseline.json
    rejected=$?
    set -e
    test "$rejected" = 1
    echo NEW_CONFLICT_REJECTED=1
    mkdir -p /tmp/malformed-fixture/local26_bad_metadata-1.0.dist-info
    printf "Metadata-Version: 2.1\nName: local26-bad-metadata\nVersion: 1.0\nRequires-Dist: ???\n" > /tmp/malformed-fixture/local26_bad_metadata-1.0.dist-info/METADATA
    set +e
    PYTHONPATH=/tmp/malformed-fixture python scripts/musa/check_dependencies.py compare /tmp/pip-baseline.json
    rejected=$?
    set -e
    test "$rejected" = 1
    echo MALFORMED_METADATA_REJECTED=1
    python -m compileall -q scripts/musa
    bash -n scripts/musa/install.sh scripts/musa/container.sh
  ' 2>&1 | tee /tmp/local26-dependency-gate-check.log
result=${PIPESTATUS[0]}
printf 'DEPENDENCY_GATE_EXIT=%s\n' "$result" | tee -a /tmp/local26-dependency-gate-check.log
exit "$result"
