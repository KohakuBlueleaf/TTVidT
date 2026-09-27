"""Export pretrained DiT decoder checkpoints for downstream use.

Strips all input projections (img_proj, motion_proj, context_proj, cond_proj)
— only xt_proj is kept. Downstream loads with strict=False to re-init skipped layers.

Saves:
  - decoder state_dict (filtered) in safetensors format
  - config metadata as JSON sidecar

Usage:
  python scripts/train/export_decoder.py <lightning_ckpt_path> [--output <output_path>]
  python scripts/train/export_decoder.py ttvidt-decoder/<run_id>/checkpoints/epoch=*-step=100000.ckpt
"""

import argparse
import json
import torch
from pathlib import Path
from safetensors.torch import save_file


SKIP_PREFIXES = (
    "decoder.img_proj.",
    "decoder.motion_proj.",
    "decoder.context_proj.",
    "decoder.cond_proj.",
)


def export(ckpt_path: str, output_path: str | None = None):
    ckpt_path = Path(ckpt_path)
    if output_path is None:
        output_path = ckpt_path.with_suffix(".exported.pt")
    else:
        output_path = Path(output_path)

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Lightning wraps state_dict under "state_dict" key
    if "state_dict" in ckpt:
        full_sd = ckpt["state_dict"]
    else:
        full_sd = ckpt

    # Extract decoder keys, strip "decoder." prefix, skip encoder-dependent projections
    decoder_sd = {}
    skipped = []
    for k, v in full_sd.items():
        if not k.startswith("decoder."):
            continue
        if any(k.startswith(prefix) for prefix in SKIP_PREFIXES):
            skipped.append(k)
            continue
        # Strip "decoder." prefix
        decoder_sd[k[len("decoder."):]] = v

    print(f"Exported: {len(decoder_sd)} params")
    print(f"Skipped:  {len(skipped)} params ({', '.join(s.split('.')[1] for s in skipped)})")

    # Extract config from hparams (decoder_config is saved as full copy before .pop())
    hparams = ckpt.get("hyper_parameters", {})
    decoder_config = hparams.get("decoder_config", {})

    # MotionDecoder init args: everything needed to reconstruct the decoder
    export_meta = {
        "decoder_config": decoder_config,
        "encoder_hidden_size": hparams.get("encoder_hidden_size"),
        "latent_h": hparams.get("latent_h"),
        "latent_w": hparams.get("latent_w"),
        "latent_mean": hparams.get("latent_mean"),
        "latent_std": hparams.get("latent_std"),
        "pretrain_mode": hparams.get("pretrain_mode"),
        "source_ckpt": str(ckpt_path),
    }

    # Save weights as safetensors
    st_path = output_path.with_suffix(".safetensors")
    save_file(decoder_sd, str(st_path))
    print(f"Saved weights: {st_path} ({st_path.stat().st_size / 1e6:.1f} MB)")

    # Save config as JSON sidecar
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(export_meta, f, indent=2)
    print(f"Saved config:  {json_path}")

    print(f"\nTo load downstream:")
    print(f"  from safetensors.torch import load_file")
    print(f"  sd = load_file('{st_path.name}')")
    print(f"  decoder.load_state_dict(sd, strict=False)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export pretrained DiT decoder checkpoint")
    parser.add_argument("ckpt_path", help="Path to Lightning checkpoint")
    parser.add_argument("--output", "-o", help="Output path (default: <ckpt>.exported.pt)")
    args = parser.parse_args()
    export(args.ckpt_path, args.output)
