"""Opt-in startup hook, inherited by Python workers started by musa/run.py."""

import os
import sys

if os.environ.get("SELF_LLM_MUSA") == "1":
    try:
        os.environ["TORCHADA_FORCE_CUDA_AVAILABLE"] = "1"
        # torchada may compile extensions and launch Python-based build tools.
        # Those tools must not import torchada again while its build lock is held.
        os.environ["SELF_LLM_MUSA"] = "0"
        try:
            import torchada  # noqa: F401
            import torch_musa  # noqa: F401
            import torch
        finally:
            os.environ["SELF_LLM_MUSA"] = "1"

        if not torch.musa.is_available():
            raise RuntimeError("MUSA GPU unavailable; check the driver and visible devices")
        if not torch.cuda.is_available():
            raise RuntimeError("torchada did not enable CUDA API compatibility")
    except Exception as exc:
        # Python normally ignores sitecustomize failures and runs the target anyway.
        print(f"self-llm MUSA startup failed: {exc}", file=sys.stderr, flush=True)
        os._exit(1)
