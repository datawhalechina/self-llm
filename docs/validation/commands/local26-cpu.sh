#!/usr/bin/env bash
set -o pipefail
python3 - <<'PY'
from pathlib import Path
for p in Path('/home/mccxadmin/self-llm-local26/scripts/musa').rglob('._*'):
    if p.is_file():
        p.unlink()
PY
docker run --rm --runtime=runc -e MTHREADS_VISIBLE_DEVICES=none \
  --label serverlist.owner=dongyang --label serverlist.ticket=LOCAL-26 \
  --mount type=bind,src=/home/mccxadmin/self-llm-local26,dst=/workspace/self-llm \
  --workdir /workspace/self-llm \
  registry.mthreads.com/mcctest/training-suite:v2.1.7 bash -lc '
    python -m pip install --no-deps torchada &&
    python -c "import importlib.metadata as m; print(\"TORCHADA_VERSION=\" + m.version(\"torchada\"))" &&
    python -m unittest discover -s scripts/musa/tests -v &&
    python -m compileall -q scripts/musa &&
    bash -n scripts/musa/install.sh scripts/musa/container.sh
  ' 2>&1 | tee /tmp/local26-cpu-check.log
printf 'CPU_CHECK_EXIT=%s\n' "${PIPESTATUS[0]}"
