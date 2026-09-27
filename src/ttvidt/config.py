"""Python config files and the ``ttvidt-run`` launcher.

A config is a plain Python file. Its UPPERCASE module-level names form the run
configuration, captured by ``config_gen()``::

    from ttvidt.config import Config

    LEARNING_RATE = 5e-4
    EPOCH = 8

    def config_gen():
        return Config.from_globals()

``config_gen()`` may also be a generator that yields one ``Config`` per run.

Configs can build on shared bases stored under ``configs/_base``: the loader
puts the enclosing ``configs/`` directory on ``sys.path`` while the file runs,
so a config may ``from _base.pretrain import *`` and then override names.

Running a script with a config::

    ttvidt-run <script.py> -c <config.py> [KEY=VALUE ...] [--entrypoint FUNC]
    # equivalently: python -m ttvidt.config run <script.py> -c <config.py> ...

The launcher

1. executes the config and obtains its ``Config`` (or each yielded ``Config``);
2. imports the script as a module (its ``if __name__ == "__main__"`` block does
   not run), so its module-level defaults exist first;
3. sets every config entry, then every ``KEY=VALUE`` override, on the script
   module, replacing the defaults;
4. calls the function invoked in the script's ``__main__`` guard (``main()``).

``KEY=VALUE`` values go through ``ast.literal_eval`` and fall back to a plain
string, so ``BATCH_SIZE=16``, ``GPUS=[0,1]``, ``DEVICE=cpu`` and
``CHECKPOINT_PATH=ttvidt/abc123/checkpoints/epoch=7.ckpt`` all work. A leading
``--`` on the key is accepted.

The contract is the same as the KohakuEngine ``kogine run`` command the
experiments were developed with, so either launcher runs these configs.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator

if __name__ == "__main__":  # `python -m ttvidt.config`: share one Config class
    sys.modules.setdefault("ttvidt.config", sys.modules[__name__])

_PROTECTED = frozenset({"__name__", "__file__", "__package__", "__loader__", "__spec__",
                        "__cached__", "__builtins__", "__doc__"})
_CONVENTIONAL_ENTRYPOINTS = ("main", "run")


# ----------------------------------------------------------------------------- Config
def _filter_globals(namespace: dict[str, Any], module_name: str) -> dict[str, Any]:
    """Config entries of a module: public names that are not modules and not
    callables/classes imported from elsewhere."""
    out: dict[str, Any] = {}
    for name, value in namespace.items():
        if name.startswith("_") or isinstance(value, ModuleType):
            continue
        if isinstance(value, type) or callable(value):
            if getattr(value, "__module__", None) == module_name:
                out[name] = value
            continue
        out[name] = value
    return out


@dataclass
class Config:
    """One script execution: globals to inject, plus entrypoint args / kwargs."""

    globals_dict: dict[str, Any] = field(default_factory=dict)
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.globals_dict, dict):
            raise TypeError("globals_dict must be a dict")
        self.args = list(self.args)

    @classmethod
    def from_globals(cls) -> "Config":
        """Capture the calling module's globals (call from inside a config file)."""
        frame = inspect.currentframe().f_back
        g = frame.f_globals if frame is not None else {}
        return cls(globals_dict=_filter_globals(g, str(g.get("__name__", "<unknown>"))))

    def __getitem__(self, k: str) -> Any:
        return self.globals_dict[k]

    def __contains__(self, k: str) -> bool:
        return k in self.globals_dict

    def get(self, k: str, default: Any = None) -> Any:
        return self.globals_dict.get(k, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.globals_dict)

    def copy(self) -> "Config":
        return Config(dict(self.globals_dict), list(self.args), dict(self.kwargs), dict(self.metadata))


def _is_config(c: Any) -> bool:
    return isinstance(c, Config) or (type(c).__name__ == "Config" and hasattr(c, "globals_dict"))


# ----------------------------------------------------------------------------- loading
def _exec_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _configs_root(path: Path) -> Path:
    """Nearest ancestor directory named ``configs`` (else the file's own dir)."""
    for parent in path.parents:
        if parent.name == "configs":
            return parent
    return path.parent


def load_config(path: str | Path) -> Config | Iterator[Config]:
    """Execute a config file; return its ``Config`` or an iterator of them.

    Base modules imported from ``configs/_base`` are dropped from ``sys.modules``
    afterwards, so loading several configs in one process never shares (or
    leaks mutations of) the base's dicts.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    root = str(_configs_root(path))
    before = set(sys.modules)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        name = "ttvidt_cfg_" + re.sub(r"\W", "_", path.stem)
        mod = _exec_module(path, name)
        if _is_config(getattr(mod, "CONFIG", None)):
            return mod.CONFIG
        if hasattr(mod, "config_gen"):
            result = mod.config_gen()
            if _is_config(result):
                return result
            if isinstance(result, Iterable):
                return _checked(iter(result))
            raise TypeError(f"config_gen() in {path.name} must return a Config or yield Configs")
        return Config(globals_dict=_filter_globals(vars(mod), name))
    finally:
        if added and root in sys.path:
            sys.path.remove(root)
        for m in set(sys.modules) - before:
            f = getattr(sys.modules[m], "__file__", None) or ""
            if f.startswith(root) or m.startswith("ttvidt_cfg_"):
                sys.modules.pop(m, None)


def _checked(it: Iterator[Any]) -> Iterator[Config]:
    for i, c in enumerate(it):
        if not _is_config(c):
            raise TypeError(f"config_gen() yield #{i} is {type(c).__name__}, not Config")
        yield c


def parse_override(kv: str) -> tuple[str, Any]:
    """``KEY=VALUE`` -> (key, value), literal-parsed with a string fallback."""
    if "=" not in kv:
        raise ValueError(f"override must be KEY=VALUE, got {kv!r}")
    k, v = kv.split("=", 1)
    k = k.lstrip("-").strip()
    if not k.isidentifier():
        raise ValueError(f"bad override name {k!r}")
    try:
        return k, ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return k, v


# ----------------------------------------------------------------------------- running
def _import_script(script: Path) -> ModuleType:
    d = str(script.parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    name = re.sub(r"\W", "_", script.stem)
    if not name.isidentifier() or name in sys.modules:
        name = f"ttvidt_script_{name}"
    return _exec_module(script, name)


def _is_main_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__" and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _main_block_function(script: Path) -> str | None:
    """Name of the function called inside ``if __name__ == "__main__":``."""
    tree = ast.parse(script.read_text(), filename=str(script))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if isinstance(call.func, ast.Name):
                        return call.func.id
                    if (isinstance(call.func, ast.Attribute) and call.func.attr == "run"
                            and isinstance(call.func.value, ast.Name)
                            and call.func.value.id == "asyncio" and call.args
                            and isinstance(call.args[0], ast.Call)
                            and isinstance(call.args[0].func, ast.Name)):
                        return call.args[0].func.id
    return None


def find_entrypoint(module: ModuleType, script: Path, explicit: str | None = None) -> Callable:
    names = [explicit] if explicit else [_main_block_function(script), *_CONVENTIONAL_ENTRYPOINTS]
    for n in names:
        fn = getattr(module, n, None) if n else None
        if callable(fn):
            return fn
    raise RuntimeError(f"no entrypoint found in {script} (looked for {[n for n in names if n]}); "
                       "pass --entrypoint <function>")


def inject(module: ModuleType, globals_dict: dict[str, Any]) -> None:
    for k, v in globals_dict.items():
        if k in _PROTECTED:
            raise ValueError(f"cannot override protected module attribute {k}")
        setattr(module, k, v)


def run(script: str | Path, config: str | Path | None = None,
        overrides: Iterable[str] | None = None, entrypoint: str | None = None) -> Any:
    """Run ``script`` under ``config``; returns the last entrypoint result."""
    script = Path(script).resolve()
    if not script.exists():
        raise FileNotFoundError(f"script not found: {script}")
    ov = dict(parse_override(kv) for kv in (overrides or []))
    if config is None:
        cfgs: Iterable[Config] = [Config()]
    else:
        loaded = load_config(config)
        cfgs = [loaded] if _is_config(loaded) else loaded
    module = _import_script(script)
    fn = find_entrypoint(module, script, entrypoint)
    result = None
    for cfg in cfgs:
        inject(module, {**cfg.globals_dict, **ov})
        result = fn(*cfg.args, **cfg.kwargs)
        if inspect.iscoroutine(result):
            result = asyncio.run(result)
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ttvidt-run",
                                description="Run a script with a Python config file.")
    p.add_argument("script", help="script to run; `run` may precede it")
    p.add_argument("-c", "--config", default=None)
    p.add_argument("--entrypoint", default=None,
                   help="function to call (default: the one in the __main__ guard)")
    p.add_argument("overrides", nargs="*", help="KEY=VALUE overrides")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":  # `python -m ttvidt.config run ...`
        argv = argv[1:]
    a, rest = p.parse_known_args(argv)
    extra = [r for r in rest if r.startswith("-") and "=" in r]
    if len(extra) != len(rest):
        p.error(f"unrecognized arguments: {' '.join(r for r in rest if r not in extra)}")
    run(a.script, a.config, [*a.overrides, *extra], a.entrypoint)


if __name__ == "__main__":
    main()
