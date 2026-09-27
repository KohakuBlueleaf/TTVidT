"""Convert a training checkpoint into a self-contained model directory.

    python scripts/tools/export_release.py ttvidt/<run_id>/checkpoints/epoch=7.ckpt release/ttvidt-tt3d

writes ``config.json`` + ``model.safetensors`` (encoder and decoder; the frozen
frame VAE is not included). Load the result with ``ttvidt.hub.load_model``.
"""

import argparse

from ttvidt.hub import export_release

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("output_dir")
    ap.add_argument("--name", default=None, help="model name recorded in config.json")
    a = ap.parse_args()
    extra = {"name": a.name} if a.name else None
    print("wrote", export_release(a.checkpoint, a.output_dir, extra=extra))
