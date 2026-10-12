"""Keep CPU index sampling compatible with torchada's MUSA Generator alias."""

import builtins
import functools
import sys
import types


class _SamplingTorch:
    def __init__(self, torch):
        self._torch = torch
        self.Generator = torch._C.Generator

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def get_default_device(self):
        # Accelerate consults this getter when creating an index generator.
        # torchada reports MUSA here while randperm's index tensor stays on CPU.
        return self.Generator().device


def _cpu_function(function, view):
    if not isinstance(function, types.FunctionType):
        return function
    if getattr(function, "_musa_cpu_sampling", False):
        return function
    if function.__globals__.get("torch") is not view._torch:
        return function
    namespace = dict(function.__globals__, torch=view)
    result = types.FunctionType(function.__code__, namespace, function.__name__,
                                function.__defaults__, function.__closure__)
    functools.update_wrapper(result, function)
    result.__kwdefaults__ = function.__kwdefaults__
    result._musa_cpu_sampling = True
    return result


def activate(torch):
    # The runner's process tests use a minimal framework stub without data APIs.
    if getattr(getattr(torch, "_C", None), "Generator", None) is None:
        return
    from torch.utils.data import RandomSampler, DistributedSampler

    view = _SamplingTorch(torch)
    for sampler in (RandomSampler, DistributedSampler):
        sampler.__iter__ = _cpu_function(sampler.__iter__, view)

    def patch_accelerate():
        loader = sys.modules.get("accelerate.data_loader")
        if loader is None:
            return
        sampler = getattr(loader, "SeedableRandomSampler", None)
        if sampler is not None:
            for name in ("__init__", "__iter__"):
                setattr(sampler, name, _cpu_function(getattr(sampler, name), view))
        original = getattr(loader, "prepare_data_loader", None)
        if original is not None:
            patched = _cpu_function(original, view)
            loader.prepare_data_loader = patched
            accelerator = sys.modules.get("accelerate.accelerator")
            if accelerator is not None and getattr(accelerator, "prepare_data_loader", None) is original:
                accelerator.prepare_data_loader = patched

    patch_accelerate()
    original_import = builtins.__import__

    def import_with_sampling(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        package = (globals or {}).get("__package__", "") or ""
        if name == "accelerate" or name.startswith("accelerate.") or package.startswith("accelerate"):
            patch_accelerate()
        return module

    builtins.__import__ = import_with_sampling
