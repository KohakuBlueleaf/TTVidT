import os
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "1") == "1"
COMPILE_KWARG = {"mode": "default", "dynamic": False}
