"""Load TT-VidT models from training checkpoints or exported model directories.

Exported models (``export_release``) are a directory (local, or a Hugging Face repo) with

    config.json         architecture + the keyword arguments of TTVidTrainer
    model.safetensors   encoder (+ decoder) weights

and load without network access to the DINOv3 base model: the architecture is
rebuilt from ``config.json`` and every weight comes from ``model.safetensors``.

    from ttvidt.hub import load_model
    model = load_model("ttvidt/<run_id>/checkpoints/epoch=7.ckpt", device="cuda")   # or an exported dir
    out = model.encoder(video)          # video: [B, T, 3, 256, 256] in [-1, 1]
    motion = out.motion_output          # [B, T, K, D] motion tokens per frame

Training checkpoints (``ttvidt/<run_id>/checkpoints/epoch=N.ckpt``) load the same
way; their architecture comes from the hyper-parameters stored in the file.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from ttvidt.trainer import TTVidTrainer

RELEASE_FORMAT = "ttvidt-release-v1"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
# frozen frame VAE: not part of the encoder, shipped separately
_SKIP_PREFIXES = ("decoder_ae_enc.", "decoder_ae_dec.", "latent_enc.", "latent_dec.")


def _init_kwargs(hp: dict) -> dict:
    """Constructor kwargs from stored hyper-parameters, for evaluation use:
    no frame VAE, no pretrained-decoder download (weights come from the file)."""
    accepted = set(inspect.signature(TTVidTrainer.__init__).parameters) - {"self"}
    kw = {k: v for k, v in hp.items() if k in accepted}
    kw.update(decoder_ae=None, encoder_model=None, decoder_model=None, decoder_pretrained=None)
    return kw


def _infer_tt3d_variant(hp: dict, state_dict: dict) -> dict:
    """Checkpoints trained before ``tt_spatial_depthwise`` existed do not record it;
    read the variant off the weights (the depth-separable resample has a
    ``spatial_mix`` parameter, the full-linear one does not)."""
    me = dict(hp.get("motion_encoder_config") or {})
    if (hp.get("backbone_arch", "ttvidt") == "ttvidt" and me.get("tt_mode") == "3d"
            and "tt_spatial_depthwise" not in me):
        me["tt_spatial_depthwise"] = any(k.endswith("spatial_down.spatial_mix") for k in state_dict)
        hp = {**hp, "motion_encoder_config": me}
    return hp


def _load_weights(model: TTVidTrainer, state_dict: dict, with_decoder: bool) -> None:
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(_SKIP_PREFIXES)}
    if not with_decoder:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder.")}
    model.on_load_checkpoint({"state_dict": state_dict})  # key remap for older checkpoints
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith(_SKIP_PREFIXES)
               and (with_decoder or not k.startswith("decoder."))]
    if missing or unexpected:
        raise RuntimeError(f"weights do not match the model: {len(missing)} missing "
                           f"(e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if str(path).count("/") == 1:  # Hugging Face repo id
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(path), allow_patterns=[CONFIG_NAME, WEIGHTS_NAME]))
    raise FileNotFoundError(f"no checkpoint at {path}")


def load_model(path: str | Path, device: str | torch.device = "cpu",
               with_decoder: bool = True, use_ema: bool = False) -> TTVidTrainer:
    """Load a model in eval mode from a release dir / repo id or a ``.ckpt`` file.

    ``use_ema`` (training checkpoints only) loads the exponential moving average of
    the weights kept during training instead of the raw weights.
    Exported models hold the raw weights.
    """
    p = _resolve(path)
    if p.suffix == ".ckpt":
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        hp = _infer_tt3d_variant(ck["hyper_parameters"], ck["state_dict"])
        model = TTVidTrainer(**_init_kwargs(hp))
        _load_weights(model, ck["state_dict"], with_decoder)
        if use_ema:
            if "ema_state" not in ck:
                raise ValueError(f"{p} has no EMA weights")
            ema = dict(ck["ema_state"])
            model.on_load_checkpoint({"state_dict": ema})
            missing, unexpected = model.load_state_dict(ema, strict=False)
            if unexpected:
                raise RuntimeError(f"EMA weights do not match the model: {unexpected[:3]}")
    else:
        from safetensors.torch import load_file

        d = p if p.is_dir() else p.parent
        cfg = json.loads((d / CONFIG_NAME).read_text())
        if cfg.get("format") != RELEASE_FORMAT:
            raise ValueError(f"{d / CONFIG_NAME}: unknown format {cfg.get('format')!r}")
        if use_ema:
            raise ValueError("exported models have no separate EMA copy")
        model = TTVidTrainer(**_init_kwargs(cfg["model"]))
        _load_weights(model, load_file(str(d / WEIGHTS_NAME)), with_decoder)
    return model.to(device).eval().requires_grad_(False)


def export_release(ckpt_path: str | Path, out_dir: str | Path, extra: dict | None = None) -> Path:
    """Convert a training checkpoint into the release format (see module docstring).

    The TT-VidT encoder is stored with its full DINOv3VidTConfig so that loading
    never needs the (gated) DINOv3 base weights.
    """
    from safetensors.torch import save_file

    model = load_model(ckpt_path, device="cpu", with_decoder=True)
    hp = dict(model.hparams)
    kw = {k: v for k, v in _init_kwargs(hp).items()
          if k not in ("decoder_ae", "encoder_model", "decoder_model", "decoder_pretrained")}
    if kw.get("backbone_arch") == "ttvidt":
        kw["backbone_config"] = model.encoder.config.to_dict()
        kw["base_model_name"] = None
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()
             if not k.startswith(_SKIP_PREFIXES)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out / WEIGHTS_NAME))
    cfg = {"format": RELEASE_FORMAT, "model": kw, **(extra or {})}
    (out / CONFIG_NAME).write_text(json.dumps(cfg, indent=2, default=str))
    return out
