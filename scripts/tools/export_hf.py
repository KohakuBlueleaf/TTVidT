"""Make an exported model directory loadable with ``transformers`` (``trust_remote_code``).

    python scripts/tools/export_hf.py release/ttvidt-tt3d <out dir>

Input: a directory written by ``export_release`` (config.json + model.safetensors).
Output: the same weights and config, with the ``transformers`` fields added to
config.json (``model_type``, ``architectures``, ``auto_map``) and the encoder source
files copied flat next to them (package imports rewritten to relative ones). The result
loads both ways:

    AutoModel.from_pretrained(out_dir, trust_remote_code=True)     # transformers only
    ttvidt.hub.load_model(out_dir)                                 # this repository

Upload the directory as a Hugging Face model repo (e.g. ``hf upload <repo> <out dir>``).
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import ttvidt
from ttvidt.hub import CONFIG_NAME, RELEASE_FORMAT, WEIGHTS_NAME

SRC = Path(ttvidt.__file__).parent
# bundled file name -> source file
FILES = {
    "modeling_ttvidt.py": "model/hf.py",
    "dinov3_vit.py": "model/dinov3_vit.py",
    "tt3d.py": "modules/tt3d.py",
    "tt.py": "modules/tt.py",
    "layers.py": "modules/layers.py",
    "patch.py": "modules/patch.py",
    "pos_embed.py": "modules/pos_embed.py",
    "utils.py": "utils.py",
    "env.py": "env.py",
}
# optimfactory's muP init (only used for random init), so the bundle needs torch +
# transformers only
MUP = '''"""muP initialisation (same as ``optimfactory.mup_init`` / ``mup_init_output``)."""

import math

import torch


def mup_init(params, is_output: bool = False) -> None:
    for param in params:
        if param.ndim == 1:
            continue
        fan_in = math.prod(param.shape[1:])
        std = (1 / fan_in) ** (1 if is_output else 0.5)
        torch.nn.init.normal_(param, mean=0.0, std=std)


def mup_init_output(param: torch.Tensor) -> None:
    mup_init([param], is_output=True)
'''
HF_FIELDS = {
    "model_type": "ttvidt",
    "architectures": ["TTVidTModel"],
    "auto_map": {
        "AutoConfig": "modeling_ttvidt.TTVidTConfig",
        "AutoModel": "modeling_ttvidt.TTVidTModel",
    },
}


def bundle_source(text: str) -> str:
    text = re.sub(
        r"from ttvidt\.(?:model|modules)\.(\w+) import", r"from .\1 import", text
    )
    text = text.replace("from ttvidt.utils import", "from .utils import")
    text = text.replace("from optimfactory import", "from .mup import")
    # transformers only bundles files named in ``from .<module> import`` lines
    text = text.replace(
        "from . import env\n",
        "from . import env\nfrom .env import TORCH_COMPILE as _  # noqa: F401 (bundles env.py)\n",
    )
    if "ttvidt." in re.sub(r"#.*|\"\"\"[\s\S]*?\"\"\"", "", text):
        raise ValueError("unrewritten package reference in bundled source")
    return text


def main(release_dir: str, out_dir: str) -> None:
    src, out = Path(release_dir), Path(out_dir)
    cfg = json.loads((src / CONFIG_NAME).read_text())
    if cfg.get("format") != RELEASE_FORMAT:
        raise ValueError(f"{src / CONFIG_NAME}: not a {RELEASE_FORMAT} directory")
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in FILES.items():
        (out / name).write_text(bundle_source((SRC / rel).read_text()))
    (out / "mup.py").write_text(MUP)
    (out / CONFIG_NAME).write_text(json.dumps({**cfg, **HF_FIELDS}, indent=2))
    if src.resolve() != out.resolve():
        shutil.copy(src / WEIGHTS_NAME, out / WEIGHTS_NAME)
        for extra in ("README.md",):
            if (src / extra).exists():
                shutil.copy(src / extra, out / extra)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("release_dir")
    ap.add_argument("output_dir")
    a = ap.parse_args()
    main(a.release_dir, a.output_dir)
