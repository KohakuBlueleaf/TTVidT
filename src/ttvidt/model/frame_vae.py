"""Frozen frame VAE: reconstruction targets live in its latent space.

Any VAE with the diffusers ``AutoencoderKL`` API can be used:

* a local folder or Hugging Face repo in diffusers format, e.g. the default
  ``KBlueLeaf/latentmaid-vae`` (custom code, loaded with ``trust_remote_code``) or
  ``stabilityai/sdxl-vae`` / ``KBlueLeaf/EQ-SDXL-VAE``;
* ``"<repo>:<subfolder>"`` for a VAE inside a pipeline repo,
  e.g. ``"black-forest-labs/FLUX.1-dev:vae"``;
* a legacy ``.safetensors`` / ``.pt`` state dict of ``ttvidt.model.image_vae``.

``load_frame_vae`` returns an ``(encoder, decoder)`` pair of modules with the
interface the trainer uses: ``encoder(x) -> (mean, logvar)`` for pixels in
[-1, 1], and ``decoder(z) -> x``. It also returns the latent channel count (the
decoder's ``image_dim`` must match) and the per-channel latent statistics stored
in the VAE config, if any (``latents_mean`` / ``latents_std``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


class _Encoder(nn.Module):
    """``x -> (mean, logvar)`` over a diffusers VAE (owns the VAE module)."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        dist = self.vae.encode(x).latent_dist
        return dist.mean, dist.logvar


class _Decoder(nn.Module):
    """``z -> x`` over the same VAE. The VAE is held outside the module registry
    so its weights are not stored twice; ``_Encoder`` owns and moves them."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self._vae = [vae]

    def forward(self, z):
        return self._vae[0].decode(z).sample


@dataclass
class FrameVAE:
    encoder: nn.Module
    decoder: nn.Module
    latent_channels: int
    latents_mean: list | None = None
    latents_std: list | None = None


def _read_config(repo: str, subfolder: str = "") -> dict:
    import json

    local = Path(repo) / subfolder / "config.json"
    if local.is_file():
        return json.loads(local.read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(repo, "config.json", subfolder=subfolder or None)).read_text())


def load_frame_vae(spec: str, device: str = "cpu") -> FrameVAE:
    """Load a frozen frame VAE from a diffusers folder / repo id or a legacy file."""
    p = Path(spec)
    if p.is_file() and p.suffix in (".safetensors", ".pt", ".pth"):
        from ttvidt.model.image_vae import load_autoencoder

        vae = load_autoencoder(str(p), device=device)
        return FrameVAE(vae.encoder.eval().requires_grad_(False),
                        vae.decoder.eval().requires_grad_(False), latent_channels=4)

    from diffusers import AutoModel

    repo, _, subfolder = str(spec).partition(":") if not p.exists() else (str(spec), "", "")
    kwargs = {"subfolder": subfolder} if subfolder else {}
    # custom-code VAEs (e.g. KBlueLeaf/latentmaid-vae) declare an auto_map in config.json
    kwargs["trust_remote_code"] = "auto_map" in _read_config(repo, subfolder)
    vae = AutoModel.from_pretrained(repo, **kwargs)
    vae = vae.to(device).eval().requires_grad_(False)
    cfg = vae.config
    mean, std = cfg.get("latents_mean"), cfg.get("latents_std")
    return FrameVAE(_Encoder(vae), _Decoder(vae), latent_channels=int(cfg["latent_channels"]),
                    latents_mean=list(mean) if mean is not None else None,
                    latents_std=list(std) if std is not None else None)


@torch.no_grad()
def check_latent_channels(vae: FrameVAE, decoder_config: dict) -> None:
    want = decoder_config.get("image_dim")
    if want is not None and want != vae.latent_channels:
        raise ValueError(f"frame VAE has {vae.latent_channels} latent channels but the decoder "
                         f"expects image_dim={want}; set MOTION_DECODER_CONFIG['image_dim']")
