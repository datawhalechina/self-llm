#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
baseline=$(mktemp)
trap 'rm -f "$baseline"' EXIT
python "$script_dir/check_dependencies.py" snapshot "$baseline"

# Keep the image's PyTorch/MUSA stack. An absent torch_musa is an image problem.
python -c 'import torch; import torch_musa; print("torch:", torch.__version__, "torch_musa:", getattr(torch_musa, "__version__", "unknown"))'
if ! python -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("torchada") is None)'; then
    python -m pip install --no-deps torchada
fi
python -c 'import torchada; import importlib.metadata as m; print("torchada:", m.version("torchada"))'
# PEFT 0.20 rejects torchao 0.9 even for ordinary, unquantized LoRA layers.
# Remove this incompatible optional package only in the task's isolated container.
if python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
import sys

try:
    incompatible = (Version(version("peft")) >= Version("0.20.0")
                    and Version(version("torchao")) < Version("0.16.0"))
except PackageNotFoundError:
    incompatible = False
sys.exit(0 if incompatible else 1)
PY
then
    echo 'Removing incompatible optional torchao for unquantized PEFT LoRA'
    python -m pip uninstall -y torchao
fi
python "$script_dir/check_dependencies.py" compare "$baseline"
