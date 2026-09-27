import importlib
import torch
from . import env


def import_class(cls_string):
    if not isinstance(cls_string, str):
        return cls_string
    module, cls = cls_string.rsplit(".", 1)
    module = importlib.import_module(module)
    return getattr(module, cls)


def compile_wrapper(func):
    """Decorator: compile the function once with torch.compile when TORCH_COMPILE=True."""
    _compiled = None

    def wrapper(*args, **kwargs):
        nonlocal _compiled
        if env.TORCH_COMPILE:
            if _compiled is None:
                _compiled = torch.compile(func, **env.COMPILE_KWARG)
            return _compiled(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper
