#!/usr/bin/env bash
set -euo pipefail

# Keep the image's PyTorch/MUSA stack. An absent torch_musa is an image problem.
python -c 'import torch; import torch_musa; print("torch:", torch.__version__, "torch_musa:", getattr(torch_musa, "__version__", "unknown"))'
if ! python -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("torchada") is None)'; then
    python -m pip install --no-deps torchada
fi
python -c 'import torchada; import importlib.metadata as m; print("torchada:", m.version("torchada"))'
python -m pip check
