import copy
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from anyschedule import AnySchedule

from transformers import DINOv3ViTModel
from ttvidt.model import DINOv3VidTModel, MotionDecoder
from ttvidt.model import VideoMAE3DViTModel, DisMo2DPlus3DModel
from ttvidt.data.video import save_video


class DeltaTEmbedding(nn.Module):
    """
    Maps integer shift k to a single [D] token via sincos encoding + MLP.

    Used by adaptive_ar to tell the decoder how far ahead to predict.
    The output token is concatenated to the motion sequence (M+1 tokens).
    """

    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        half_dim = hidden_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self, k: int, batch_size: int, num_pairs: int, device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            k: integer shift value
            batch_size: B
            num_pairs: N (number of temporal pairs)
            device: torch device

        Returns:
            [B, N, 1, D] delta_t token to concatenate to motion_emb along dim=2
        """
        k_t = torch.tensor([k], device=device, dtype=torch.float32)
        args = k_t.unsqueeze(-1) * self.freqs  # [1, D/2]
        sincos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [1, D]
        token = self.proj(sincos)  # [1, D]
        return token.view(1, 1, 1, -1).expand(batch_size, num_pairs, 1, -1)


def mup_param_group(
    m, learning_rate, base_dim, base_weight_decay=0.001, input_layer=False
):
    pg = {}
    for param in m.parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1:
            fan_in = -1
        elif param.ndim == 2:
            fan_in = param.shape[1]
        else:
            fan_ins = param.shape[1:]
            fan_in = 1
            for s in fan_ins:
                fan_in *= s

        if input_layer or fan_in == -1:
            lr = learning_rate
            wd = base_weight_decay
        else:
            lr = learning_rate * (base_dim / fan_in)
            wd = base_weight_decay * (fan_in / base_dim)
        if lr in pg:
            pg[lr].append(param)
        else:
            pg[lr] = [param]
    param_groups = [{"params": v, "lr": k, "weight_decay": wd} for k, v in pg.items()]
    return param_groups


def _auto_scale_decoder_dims(
    motion_decoder_config: dict,
    encoder_hidden_size: int,
    encoder_intermediate_size: int,
    encoder_num_heads: int,
):
    """
    When the decoder hidden_size differs from the encoder, auto-scale
    intermediate_size and num_heads proportionally -- unless the user already
    set them explicitly in the config.

    This is called AFTER the default fallback logic has filled in
    intermediate_size / num_heads from the encoder.  If hidden_size ==
    encoder_hidden_size nothing changes (the encoder defaults are correct).

    Scaling rules:
      intermediate_size: scale by (decoder_hidden / encoder_hidden) ratio,
                         preserving the encoder's expansion factor.
      num_heads:         keep the encoder's per-head dimension
                         (head_dim = encoder_hidden / encoder_num_heads),
                         giving num_heads = decoder_hidden / head_dim.
    """
    dec_hidden = motion_decoder_config.get("hidden_size", encoder_hidden_size)
    if dec_hidden == encoder_hidden_size:
        return  # no scaling needed

    # Track which keys were set by fallback (not by the user).
    # The fallback code uses "not in" checks before setting, so if they
    # are present at this point AND equal to the encoder value, they came
    # from fallback.  But we can't distinguish user-set == encoder-value
    # from fallback.  Convention: only auto-scale when the current value
    # exactly matches the encoder value (i.e., likely came from fallback).

    cur_inter = motion_decoder_config.get("intermediate_size", encoder_intermediate_size)
    cur_heads = motion_decoder_config.get("num_heads", encoder_num_heads)

    if cur_inter == encoder_intermediate_size:
        ratio = dec_hidden / encoder_hidden_size
        motion_decoder_config["intermediate_size"] = int(encoder_intermediate_size * ratio)

    if cur_heads == encoder_num_heads:
        head_dim = encoder_hidden_size // encoder_num_heads
        new_heads = dec_hidden // head_dim
        if new_heads >= 1 and dec_hidden % head_dim == 0:
            motion_decoder_config["num_heads"] = new_heads
        # else: keep encoder default -- user should set num_heads explicitly


class TTVidTrainer(pl.LightningModule):
    def __init__(
        self,
        base_model_name=None,
        motion_encoder_config=None,
        motion_decoder_config=None,
        encoder_model=None,
        encoder_ae_mode="video",  # "video" (temporal) or "image" (per-frame)
        decoder_model=None,
        decoder_ae=None,  # Optional separate decoder AE (e.g. image VAE F8C4)
        decoder_ae_mode="image",  # "video" (temporal) or "image" (per-frame)
        latent_mean=None,  # Per-channel mean for latent normalization, e.g. [c1, c2, c3, c4]
        latent_std=None,  # Per-channel std for latent normalization
        patch_size=None,
        image_dim=None,
        token_counts=None,
        gradient_checkpointing=False,
        train_mode="regression",  # regression/autoregressive/adaptive_ar/twojump_ar/mae
        mae_mask_ratio=0.75,  # encoder-level tube masking ratio for mae mode
        ar_shift_range=None,  # (min_k, max_k) latent token shift range for adaptive_ar/twojump_ar
        ar_shift_sampling="gamma",  # "uniform" or "gamma" (DisMo-style Gamma(3,12) favoring small k)
        source_frame_count=None,  # raw frame count for standard video (for logging truncation)
        use_ref=False,
        unfreeze_backbone=False,
        decoder_mask_ratio=0.0,
        mask_patch_size=4,  # pixio style MAE
        diffusion_timestep_sampling="uniform",  # "uniform" or "logit_normal"
        backbone_arch="ttvidt",  # "ttvidt", "videomae_3d", "dismo_2d3d"
        backbone_config=None,  # dict of arch-specific config for non-ttvidt backbones
        decoder_pretrained=None,  # "<hf_repo>/<name>" of a pretrained DiT decoder, e.g. "KBlueLeaf/TTVidT-decoders/pretrain_video_S_qknorm"
        decoder_lr_multiplier=1.0,  # Decoder LR = learning_rate * decoder_lr_multiplier
        # --- optional speed knobs (off by default) ---
        fused_adam=False,  # AdamW(fused=True): single-kernel optimizer step
        ema_foreach=False,  # EMA update via torch._foreach_lerp_ instead of per-param loop
        nan_guard_interval=1,  # loss.item() NaN-guard/ema_loss sync every N steps (1 = every step)
        name="test",
        log_interval=100,
        learning_rate=0.1,
        base_dim=256,
        decoder_base_dim=None,  # muP base dim for decoder; defaults to base_dim if None
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        scheduler_config={
            "lr": {
                "mode": "cosine",
                "end": -1,
                "min_value": 0.05,
                "warmup": 0,
            }
        },
    ):
        super().__init__()
        # frozen VAE modules are passed in at construction, not stored as hparams
        self.save_hyperparameters(ignore=["decoder_ae", "encoder_model", "decoder_model"])
        self.backbone_arch = backbone_arch

        if motion_encoder_config is None:
            motion_encoder_config = {}
        if motion_decoder_config is None:
            motion_decoder_config = {}

        # Pixel input mode: when encoder_model is None, backbone receives raw
        # pixel frames directly (no latent encoding).  Frame counts stay as-is
        # (no AE temporal downsampling factor).
        self._pixel_input = encoder_model is None
        self._encoder_ae_mode = encoder_ae_mode  # "video" or "image"
        self._decoder_ae_mode = decoder_ae_mode  # "video" or "image"

        if encoder_model is not None:
            self.latent_enc = encoder_model.eval().requires_grad_(False)
            if decoder_model is not None:
                self.latent_dec = decoder_model.eval().requires_grad_(False)
            else:
                self.latent_dec = None
        else:
            self.latent_enc = None
            self.latent_dec = None

        # Optional separate decoder AE (e.g. image VAE F8C4).
        # Accepts either a single nn.Module (decoder only) or a tuple
        # (encoder, decoder) where the encoder is used to encode targets
        # to the decoder's latent space for loss computation.
        if decoder_ae is not None:
            if isinstance(decoder_ae, (tuple, list)):
                self.decoder_ae_enc = decoder_ae[0].eval().requires_grad_(False)
                self.decoder_ae_dec = decoder_ae[1].eval().requires_grad_(False)
            else:
                self.decoder_ae_enc = None
                self.decoder_ae_dec = decoder_ae.eval().requires_grad_(False)
        else:
            self.decoder_ae_enc = None
            self.decoder_ae_dec = None

        # Latent normalization: per-channel mean/std for scaling decoder AE latents.
        # Applied after encoding targets, reversed before decoding outputs.
        if latent_mean is not None:
            self.register_buffer("latent_mean", torch.tensor(latent_mean, dtype=torch.float32).view(1, 1, -1, 1, 1))
            self.register_buffer("latent_std", torch.tensor(latent_std, dtype=torch.float32).view(1, 1, -1, 1, 1))
        else:
            self.latent_mean = None
            self.latent_std = None

        # =====================================================================
        # Backbone selection
        # =====================================================================
        if backbone_arch == "ttvidt":
            from ttvidt.model.dinov3_vit import DINOv3VidTConfig

            if base_model_name is not None:
                # Init from pretrained DINOv3 weights
                dino = DINOv3ViTModel.from_pretrained(
                    base_model_name, attn_implementation="sdpa"
                )
                self.encoder: DINOv3VidTModel = DINOv3VidTModel.from_dino_v3(
                    dino, **motion_encoder_config
                )
            else:
                # Random init: build from backbone_config
                assert (
                    backbone_config is not None
                ), "backbone_config is required for ttvidt random init (base_model_name=None)"
                backbone_config = dict(backbone_config)
                # Merge motion_encoder_config into backbone_config
                for k, v in motion_encoder_config.items():
                    if k not in backbone_config:
                        backbone_config[k] = v
                cfg = DINOv3VidTConfig(**backbone_config)
                self.encoder = DINOv3VidTModel(cfg)

            if patch_size is not None or image_dim is not None:
                self.encoder.setup_patch(
                    image_dim or self.encoder.config.num_channels,
                    patch_size or self.encoder.config.patch_size,
                )
            # freeze/unfreeze pretrained backbone part
            self.encoder.freeze_pretrained(requires_grad=unfreeze_backbone)

            if use_ref:
                self.ref = (
                    DINOv3VidTModel.from_dino_v3(dino).requires_grad_(False).eval()
                )
                if "image_dim" not in motion_decoder_config:
                    motion_decoder_config["image_dim"] = self.ref.config.hidden_size
                if "patch_size" not in motion_decoder_config:
                    motion_decoder_config["patch_size"] = 1
                if "temporal_patch_size" not in motion_decoder_config:
                    motion_decoder_config["temporal_patch_size"] = 1
            else:
                self.ref = None

            # Set encoder hidden size for decoder projection if using different hidden sizes
            encoder_hidden_size = self.encoder.config.hidden_size
            if "encoder_hidden_size" not in motion_decoder_config:
                motion_decoder_config["encoder_hidden_size"] = encoder_hidden_size

            if "hidden_size" not in motion_decoder_config:
                motion_decoder_config["hidden_size"] = encoder_hidden_size
            if "intermediate_size" not in motion_decoder_config:
                motion_decoder_config["intermediate_size"] = (
                    self.encoder.config.intermediate_size
                )
            if "num_heads" not in motion_decoder_config:
                motion_decoder_config["num_heads"] = (
                    self.encoder.config.num_attention_heads
                )
            if "patch_size" not in motion_decoder_config:
                motion_decoder_config["patch_size"] = self.encoder.config.patch_size
            if "temporal_patch_size" not in motion_decoder_config:
                motion_decoder_config["temporal_patch_size"] = (
                    self.encoder.config.temporal_patch_size
                )

            # Scale intermediate_size and num_heads to match decoder hidden_size
            # when only hidden_size was explicitly set
            _auto_scale_decoder_dims(
                motion_decoder_config,
                encoder_hidden_size,
                self.encoder.config.intermediate_size,
                self.encoder.config.num_attention_heads,
            )

        elif backbone_arch == "videomae_3d":
            assert (
                VideoMAE3DViTModel is not None
            ), "VideoMAE3DViTModel could not be imported"
            assert (
                backbone_config is not None
            ), "backbone_config is required for backbone_arch='videomae_3d'"
            from ttvidt.model.videomae_3d import VideoMAE3DConfig

            cfg = VideoMAE3DConfig(**backbone_config)
            self.encoder = VideoMAE3DViTModel(cfg)
            self.ref = None

            encoder_hidden_size = cfg.hidden_size
            if "encoder_hidden_size" not in motion_decoder_config:
                motion_decoder_config["encoder_hidden_size"] = encoder_hidden_size
            if "hidden_size" not in motion_decoder_config:
                motion_decoder_config["hidden_size"] = encoder_hidden_size
            if "intermediate_size" not in motion_decoder_config:
                motion_decoder_config["intermediate_size"] = cfg.intermediate_size
            if "num_heads" not in motion_decoder_config:
                motion_decoder_config["num_heads"] = cfg.num_attention_heads
            if "patch_size" not in motion_decoder_config:
                motion_decoder_config["patch_size"] = cfg.patch_size
            if "temporal_patch_size" not in motion_decoder_config:
                motion_decoder_config["temporal_patch_size"] = cfg.temporal_patch_size

            _auto_scale_decoder_dims(
                motion_decoder_config,
                encoder_hidden_size,
                cfg.intermediate_size,
                cfg.num_attention_heads,
            )

        elif backbone_arch == "dismo_2d3d":
            assert (
                DisMo2DPlus3DModel is not None
            ), "DisMo2DPlus3DModel could not be imported"
            assert (
                backbone_config is not None
            ), "backbone_config is required for backbone_arch='dismo_2d3d'"
            from ttvidt.model.dismo_2d3d import DisMo2DPlus3DConfig

            # Optionally init Phase 1 from DINOv3 pretrained weights
            # Copy to avoid mutating the original dict (saved in hparams)
            backbone_config = dict(backbone_config)
            init_from_dino = backbone_config.pop("init_from_dino", None)
            if init_from_dino:
                dino = DINOv3ViTModel.from_pretrained(
                    init_from_dino, attn_implementation="sdpa"
                )
                self.encoder = DisMo2DPlus3DModel.from_dino_v3(
                    dino,
                    num_spatial_layers=backbone_config.get("num_spatial_layers", 8),
                    num_temporal_layers=backbone_config.get("num_temporal_layers", 12),
                    temporal_patch_size=backbone_config.get("temporal_patch_size", 4),
                    temporal_patch_overlap=backbone_config.get(
                        "temporal_patch_overlap", 1
                    ),
                    num_motion_tokens=backbone_config.get("num_motion_tokens", 16),
                    temporal_ffn_type=backbone_config.get("temporal_ffn_type", "gelu"),
                    temporal_intermediate_size=backbone_config.get("temporal_intermediate_size", None),
                )
                # Re-initialize patch embedding if input channels differ from DINOv3
                # (e.g. latent space with 64 channels vs. DINOv3's 3 RGB channels)
                target_in_channels = backbone_config.get("in_channels", None)
                target_spatial_patch = backbone_config.get("spatial_patch_size", None)
                if target_in_channels is not None or target_spatial_patch is not None:
                    from ttvidt.modules.layers import LatentPatchEmbed

                    in_ch = target_in_channels or self.encoder.config.in_channels
                    sp = target_spatial_patch or self.encoder.config.spatial_patch_size
                    self.encoder.patch_embed = LatentPatchEmbed(
                        in_channels=in_ch,
                        hidden_size=self.encoder.config.hidden_size,
                        spatial_patch_size=sp,
                    )
                    self.encoder.config.in_channels = in_ch
                    self.encoder.config.spatial_patch_size = sp
            else:
                cfg = DisMo2DPlus3DConfig(**backbone_config)
                self.encoder = DisMo2DPlus3DModel(cfg)
            self.ref = None

            dismo_config = self.encoder.config
            encoder_hidden_size = dismo_config.hidden_size
            if "encoder_hidden_size" not in motion_decoder_config:
                motion_decoder_config["encoder_hidden_size"] = encoder_hidden_size
            if "hidden_size" not in motion_decoder_config:
                motion_decoder_config["hidden_size"] = encoder_hidden_size
            if "intermediate_size" not in motion_decoder_config:
                motion_decoder_config["intermediate_size"] = (
                    dismo_config.intermediate_size
                )
            if "num_heads" not in motion_decoder_config:
                motion_decoder_config["num_heads"] = dismo_config.num_attention_heads
            if "patch_size" not in motion_decoder_config:
                motion_decoder_config["patch_size"] = dismo_config.patch_size
            if "temporal_patch_size" not in motion_decoder_config:
                motion_decoder_config["temporal_patch_size"] = (
                    dismo_config.temporal_patch_size
                )

            _auto_scale_decoder_dims(
                motion_decoder_config,
                encoder_hidden_size,
                dismo_config.intermediate_size,
                dismo_config.num_attention_heads,
            )

        else:
            raise ValueError(
                f"Unknown backbone_arch: {backbone_arch}. "
                f"Must be one of: 'ttvidt', 'videomae_3d', 'dismo_2d3d'"
            )

        # All encoder configs expose temporal_patch_size uniformly
        self.temporal_patch_size = self.encoder.config.temporal_patch_size

        self.decoder = MotionDecoder(**motion_decoder_config)
        self._decoder_is_dit = self.decoder.decoder_style == "dit"
        self._decoder_has_source_encoder = (
            hasattr(self.decoder, 'source_encoder') and self.decoder.source_encoder is not None
        )

        # Load pretrained decoder weights if specified
        if decoder_pretrained is not None:
            self._load_pretrained_decoder(decoder_pretrained)

        # All encoder configs expose patch_size uniformly
        # (new configs provide patch_size as a property alias for spatial_patch_size)
        self._encoder_patch_size = self.encoder.config.patch_size

        self.encoder.gradient_checkpointing = gradient_checkpointing
        self.decoder.gradient_checkpointing = gradient_checkpointing

        self.train_mode = train_mode
        assert self.train_mode in [
            "regression",
            "autoregressive",
            "adaptive_ar",
            "twojump_ar",
            "mae",
            "mae_diffusion",
        ], f"Unknown train_mode: {self.train_mode}"

        # MAE mode: encoder-level tube masking with token removal
        self.mae_mask_ratio = mae_mask_ratio
        if self.train_mode in ("mae", "mae_diffusion"):
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, encoder_hidden_size))

        # AR shift range: (min_k, max_k) in latent token space
        # Used by adaptive_ar and twojump_ar to randomly shift targets
        # For k_max=3 with ae_temporal_factor=4: dataset must provide FRAME_COUNT + k_max * 4 raw frames
        self.ar_shift_range = ar_shift_range
        self.ar_shift_sampling = ar_shift_sampling
        if self.train_mode in ("adaptive_ar", "twojump_ar"):
            assert (
                ar_shift_range is not None
            ), f"{self.train_mode} requires ar_shift_range=(min_k, max_k)"
            # Precompute shift sampling weights
            if ar_shift_sampling == "gamma":
                # DisMo-style: Gamma(concentration=3.0, rate=12.0) evaluated on seconds.
                # Scale integer k by 1/6 to match DisMo's 6fps temporal domain.
                dist = torch.distributions.Gamma(3.0, 12.0)
                ks = torch.arange(
                    ar_shift_range[0], ar_shift_range[1] + 1, dtype=torch.float32
                )
                log_probs = dist.log_prob(ks / 6.0)
                self._ar_shift_weights = (
                    (log_probs - log_probs.logsumexp(0)).exp().tolist()
                )
            else:
                self._ar_shift_weights = None

        # Delta-t embedding: MLP projector on sampled k → single [D] token
        # Tells the decoder how far ahead to predict (adaptive_ar only)
        # twojump_ar doesn't need this — near-target motion tokens implicitly encode the jump
        if self.train_mode == "adaptive_ar":
            self.delta_t_emb = DeltaTEmbedding(encoder_hidden_size)
        else:
            self.delta_t_emb = None

        # Number of raw frames for the standard (non-buffer) video
        # When dataset provides extra buffer frames for AR shift, this truncates raw_video for logging
        self.source_frame_count = source_frame_count

        if self._pixel_input:
            print("Input mode: pixel (no AE encoding)")
        else:
            print("Input mode: latent (with AE encoding)")
        if self.decoder_ae_dec is not None:
            print("Output decoding: separate decoder AE")
        elif self.latent_dec is not None:
            print("Output decoding: video AE decoder (latent_dec)")
        else:
            print("Output decoding: none (pixel output)")
        print(
            f"Using train mode: {self.train_mode} with Temporal Patch Size: {self.temporal_patch_size}"
        )
        # Log encoder/decoder dimension summary
        dec_hidden = self.decoder.hidden_size
        dec_enc_hidden = self.decoder.encoder_hidden_size
        if dec_hidden != dec_enc_hidden:
            print(
                f"Decoder hidden_size={dec_hidden} (encoder={dec_enc_hidden}), "
                f"using projection layers"
            )
        if self.ar_shift_range is not None:
            print(f"  AR shift range: {self.ar_shift_range} (latent token space)")
            print(f"  AR shift sampling: {self.ar_shift_sampling}", end="")
            if self._ar_shift_weights is not None:
                w = {
                    k: f"{p:.3f}"
                    for k, p in zip(
                        range(ar_shift_range[0], ar_shift_range[1] + 1),
                        self._ar_shift_weights,
                    )
                }
                print(f" → weights: {w}")
            else:
                print()

        assert 0 <= decoder_mask_ratio <= 1
        assert 1 <= mask_patch_size
        self.decoder_mask_ratio = decoder_mask_ratio
        self.mask_patch_size = mask_patch_size
        if self.decoder_mask_ratio > 0:
            print(f"Using decoder mask ratio: {self.decoder_mask_ratio}")
            print(f"Using mask patch size: {self.mask_patch_size}")

        self.diffusion_timestep_sampling = diffusion_timestep_sampling
        if self.decoder.decode_mode == "diffusion":
            print(
                f"Using diffusion mode with timestep sampling: {diffusion_timestep_sampling}"
            )

        self.token_counts = token_counts
        self.learning_rate = learning_rate
        self.decoder_lr_multiplier = decoder_lr_multiplier
        self.base_dim = base_dim
        self.decoder_base_dim = decoder_base_dim if decoder_base_dim is not None else base_dim
        self.weight_decay = weight_decay
        self.betas = betas
        self.scheduler_config = scheduler_config
        self.ema_fake_acc = -1
        self.ema_loss = -1
        self.ema_psnr = -1
        self.previous_global_step = -1
        self.start_global_step = -1
        self.log_interval = log_interval
        self.name = name

        # Optional speed knobs (all off by default)
        self.fused_adam = bool(fused_adam)
        self.ema_foreach = bool(ema_foreach)
        self.nan_guard_interval = max(int(nan_guard_interval or 1), 1)

        if unfreeze_backbone:
            self.encoder.requires_grad_(True)

        # EMA: maintain exponential moving average of trainable weights
        # Saved to checkpoint only, does not affect training
        self._ema_decay = 0.9999
        self._ema_state: dict[str, torch.Tensor] = {}
        self._ema_initialized = False
        # Cached (ema_list, param_list) groups for the foreach EMA path
        self._ema_foreach_groups = None

    def _load_pretrained_decoder(self, decoder_pretrained: str):
        """Load pretrained decoder weights.

        Args:
            decoder_pretrained: a local ``.safetensors`` file (with or without the
                extension), or "<hf_repo>/<name>" for ``<name>.safetensors`` in a
                Hugging Face repo, e.g. "KBlueLeaf/TTVidT-decoders/pretrain_video_S_qknorm".
        """
        from huggingface_hub import hf_hub_download
        import safetensors.torch as st

        local = [decoder_pretrained, f"{decoder_pretrained}.safetensors"]
        weights_path = next((f for f in local if os.path.isfile(f)), None)
        if weights_path is None:  # "<hf_repo>/<name>" -> <name>.safetensors in that repo
            parts = decoder_pretrained.split("/")
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            weights_path = hf_hub_download(repo_id, f"{filename}.safetensors")
        sd = st.load_file(weights_path)
        result = self.decoder.load_state_dict(sd, strict=False)
        print(
            f"[Decoder pretrained] Loaded from {decoder_pretrained}: "
            f"{len(sd) - len(result.unexpected_keys)} matched, "
            f"{len(result.missing_keys)} missing, "
            f"{len(result.unexpected_keys)} unexpected"
        )
        if result.missing_keys:
            print(f"  Missing: {result.missing_keys}")

    def _init_ema(self):
        """Initialize EMA state from current trainable parameters."""
        self._ema_state = {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        self._ema_initialized = True
        self._ema_foreach_groups = None
        self._ema_device_synced = True  # freshly cloned on the param device

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA weights after each optimizer step."""
        if not self._ema_initialized:
            self._init_ema()
            return
        decay = self._ema_decay
        # After a weights-only continuation (CKPT_PATH set, TRAINER_RESUME=False)
        # the EMA state is restored from a CPU checkpoint while params live on
        # GPU -> lerp_/foreach would hit a cross-device error. Migrate each EMA
        # buffer onto its param's device once, then let the groups rebuild.
        if not getattr(self, "_ema_device_synced", False):
            for name, param in self.named_parameters():
                e = self._ema_state.get(name)
                if e is not None and e.device != param.device:
                    self._ema_state[name] = e.to(param.device)
            self._ema_foreach_groups = None
            self._ema_device_synced = True
        if self.ema_foreach:
            # Fused multi-tensor path: a handful of kernels instead of one
            # lerp_ launch per parameter tensor.
            if self._ema_foreach_groups is None:
                groups: dict[tuple, tuple[list, list]] = {}
                for name, param in self.named_parameters():
                    if param.requires_grad and name in self._ema_state:
                        ema = self._ema_state[name]
                        key = (param.device, param.dtype, ema.device, ema.dtype)
                        ema_list, param_list = groups.setdefault(key, ([], []))
                        ema_list.append(ema)
                        param_list.append(param.data)
                self._ema_foreach_groups = list(groups.values())
            for ema_list, param_list in self._ema_foreach_groups:
                torch._foreach_lerp_(ema_list, param_list, 1 - decay)
            return
        for name, param in self.named_parameters():
            if param.requires_grad and name in self._ema_state:
                self._ema_state[name].lerp_(param.data, 1 - decay)

    def on_before_zero_grad(self, optimizer):
        """Called after optimizer.step(), before zero_grad(). Update EMA here."""
        self._update_ema()

    def on_save_checkpoint(self, checkpoint):
        """Save EMA weights into the checkpoint."""
        if self._ema_initialized:
            checkpoint["ema_state"] = self._ema_state

    def on_load_checkpoint(self, checkpoint):
        """Restore EMA weights from checkpoint."""
        if "ema_state" in checkpoint:
            self._ema_state = checkpoint["ema_state"]
            self._ema_initialized = True
            self._ema_foreach_groups = None
            self._ema_device_synced = False  # restored on CPU; migrate on 1st update
        # Key remap for checkpoints saved with older `transformers`: newer releases
        # nest the DINOv3 internals under `.model`, so keys such as
        # "encoder.layer.*" / "encoder.embeddings.*" would no longer match and be
        # skipped silently by strict=False, leaving the spatial path at its DINOv3
        # initialisation. Rewrite each stale "encoder.<x>" key to
        # "encoder.model.<x>" when that is what this model expects.
        own = set(self.state_dict().keys())

        def _remap(dct, tag):
            if dct is None:
                return
            n = 0
            for k in [k for k in dct.keys() if k.startswith("encoder.") and k not in own]:
                nk = "encoder.model." + k[len("encoder."):]
                if nk in own:
                    dct[nk] = dct.pop(k)
                    n += 1
            if n:
                print(f"[TTVidTrainer] remapped {n} "
                      f"encoder.* keys -> encoder.model.* ({tag})", flush=True)

        _remap(checkpoint.get("state_dict"), "state_dict")
        # keep EMA buffers aligned with the (remapped) param names so encoder
        # EMA is restored and tracked, not silently dropped on a name mismatch.
        _remap(getattr(self, "_ema_state", None), "ema_state")

    @torch.no_grad()
    def _dump_instability_diagnostics(
        self, loss_val, target_v, pred, frames_output, motion_output, encoder_result, raw_input,
    ):
        """Dump comprehensive diagnostics when loss spikes."""
        import json
        log_path = f"instability_dump_{self.name}_step{self.global_step}.log"
        lines = []
        lines.append(f"{'='*80}")
        lines.append(f"INSTABILITY DETECTED at step {self.global_step}")
        lines.append(f"  loss={loss_val:.6f}, ema_loss={self.ema_loss:.6f}, ratio={loss_val/self.ema_loss:.2f}x")
        lines.append(f"{'='*80}")

        def _stats(name, t):
            if t is None:
                return f"  {name}: None"
            t = t.float()
            return (
                f"  {name}: shape={list(t.shape)}, "
                f"mean={t.mean():.4f}, std={t.std():.4f}, "
                f"min={t.min():.4f}, max={t.max():.4f}, "
                f"absmax={t.abs().max():.4f}, "
                f"nan={t.isnan().any()}, inf={t.isinf().any()}"
            )

        # 1. Input/output stats
        lines.append("\n--- Inputs/Outputs ---")
        lines.append(_stats("raw_input", raw_input))
        lines.append(_stats("target_v", target_v))
        lines.append(_stats("pred", pred))
        lines.append(_stats("pred-target", pred - target_v if pred is not None and target_v is not None else None))
        lines.append(_stats("frames_output (to decoder)", frames_output))
        lines.append(_stats("motion_output (to decoder)", motion_output))

        # Encoder outputs
        if encoder_result is not None:
            lines.append(_stats("encoder.frames_output", encoder_result.frames_output))
            lines.append(_stats("encoder.motion_output", encoder_result.motion_output))
            if encoder_result.cls_output is not None:
                lines.append(_stats("encoder.cls_output", encoder_result.cls_output))

        # 2. Weight stats for all modules
        lines.append("\n--- Weight Stats (trainable only) ---")
        suspicious_weights = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            data = p.data.float()
            std = data.std().item()
            amx = data.abs().max().item()
            nan_count = data.isnan().sum().item()
            inf_count = data.isinf().sum().item()
            flag = ""
            if nan_count > 0:
                flag += f" NAN({nan_count}/{data.numel()})!"
            if inf_count > 0:
                flag += f" INF({inf_count}/{data.numel()})!"
            if amx > 100:
                flag += f" HUGE(absmax={amx:.1f})"
            if flag:
                suspicious_weights.append(f"  *** {name}: std={std:.4f}, absmax={amx:.4f}{flag}")
            # Always log decoder and encoder temporal layers
            if any(k in name for k in ["decoder.", "temporal", "motion_layer", "blocks."]):
                stats = (
                    f"  {name}: shape={list(p.shape)}, std={std:.6f}, absmax={amx:.4f}, "
                    f"min={data.min().item():.4f}, max={data.max().item():.4f}, "
                    f"mean={data.mean().item():.4f}, nan={nan_count}{flag}"
                )
                lines.append(stats)

        # 3. Activation analysis via hooks (run one forward pass)
        lines.append("\n--- Activation Analysis (forward pass) ---")
        activation_stats = {}
        hooks = []

        def make_hook(module_name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                elif hasattr(output, 'last_hidden_state'):
                    out = output.last_hidden_state
                else:
                    out = output
                if isinstance(out, torch.Tensor):
                    activation_stats[module_name] = {
                        "shape": list(out.shape),
                        "mean": out.float().mean().item(),
                        "std": out.float().std().item(),
                        "absmax": out.float().abs().max().item(),
                        "nan": out.isnan().any().item(),
                        "inf": out.isinf().any().item(),
                    }
            return hook_fn

        # Register hooks on key modules
        for name, module in self.named_modules():
            if any(k in name for k in [
                "encoder.layer.", "encoder.norm",
                "encoder.spatial_blocks.", "encoder.temporal_blocks.",
                "encoder.motion_layers.", "encoder.temporal_norm", "encoder.spatial_norm",
                "encoder.blocks.",
                "decoder.layers.", "decoder.final_norm", "decoder.unpatch",
                "decoder.img_proj", "decoder.motion_proj", "decoder.context_proj",
                "decoder.xt_proj", "decoder.cond_proj", "decoder.timestep_emb",
            ]):
                # Skip sub-modules (only hook the top-level named ones)
                if name.count(".") <= 3:
                    hooks.append(module.register_forward_hook(make_hook(name)))

        # Run full forward pass (encoder + decoder) to collect activations
        try:
            with torch.no_grad(), torch.autocast(raw_input.device.type, dtype=torch.float16):
                diag_input = raw_input[:2]
                enc_result = self.encoder(diag_input)
                # Build decoder inputs matching the training path
                diag_frames = enc_result.frames_output[:, 0:1].expand(-1, enc_result.frames_output.shape[1] - 1, -1, -1)
                diag_motion = enc_result.motion_output[:, 1:]
                diag_target = target_v[:2] if target_v is not None else None
                if diag_target is not None and self.decoder.decode_mode == "diffusion":
                    diag_t = torch.rand(2, device=raw_input.device)
                    diag_noise = torch.randn_like(diag_target)
                    diag_t_exp = diag_t[:, None, None, None, None]
                    diag_xt = (1 - diag_t_exp) * diag_target + diag_t_exp * diag_noise
                    dit_kw = {}
                    if self._decoder_is_dit:
                        if enc_result.cls_output is not None:
                            dit_kw["cls_emb"] = enc_result.cls_output[:, 1:enc_result.frames_output.shape[1]]
                        if self._decoder_has_source_encoder:
                            dit_kw["source_frames"] = raw_input[:2, 0:1].expand(-1, diag_frames.shape[1], -1, -1, -1)
                        else:
                            dit_kw["source_frame_emb"] = diag_frames
                    _ = self.decoder(diag_frames, diag_motion, xt=diag_xt, t=diag_t, **dit_kw)
        except Exception as e:
            lines.append(f"  Forward pass failed: {e}")

        for h in hooks:
            h.remove()

        for name in sorted(activation_stats.keys()):
            s = activation_stats[name]
            flag = ""
            if s["nan"]:
                flag += " NAN!"
            if s["inf"]:
                flag += " INF!"
            if s["absmax"] > 100:
                flag += f" HUGE"
            lines.append(
                f"  {name}: shape={s['shape']}, "
                f"mean={s['mean']:.4f}, std={s['std']:.4f}, absmax={s['absmax']:.4f}{flag}"
            )

        # 4. Summary
        if suspicious_weights:
            lines.append(f"\n--- SUSPICIOUS WEIGHTS ({len(suspicious_weights)}) ---")
            lines.extend(suspicious_weights)

        lines.append(f"\n{'='*80}")
        lines.append(f"END INSTABILITY DUMP")
        lines.append(f"{'='*80}")

        report = "\n".join(lines)
        with open(log_path, "w") as f:
            f.write(report)
        print(f"\n!!! INSTABILITY DETECTED: loss={loss_val:.4f} > 5x ema={self.ema_loss:.4f}")
        print(f"!!! Diagnostics saved to: {log_path}")
        # Also print summary to stdout
        if suspicious_weights:
            print(f"!!! {len(suspicious_weights)} suspicious weights found")
            for w in suspicious_weights[:10]:
                print(w)

    @torch.no_grad()
    def generate_video(
        self,
        frames_output: torch.Tensor,
        motion_output: torch.Tensor,
        target_shape: tuple | None = None,
        num_steps: int = 16,
        cls_emb: torch.Tensor | None = None,
        source_frame_emb: torch.Tensor | None = None,
        source_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate video from frame embeddings and motion tokens.

        Handles both regression and diffusion modes:
        - Regression: direct decoder forward
        - Diffusion: Euler ODE solver from noise to clean

        Args:
            frames_output: [B, T, L, D] frame embeddings
            motion_output: [B, T, M, D] motion tokens
            target_shape: (B, T*Pt, C, H, W) shape for diffusion mode noise
            num_steps: number of ODE steps for diffusion mode
            cls_emb: [B, T, D] CLS conditioning for DiT decoder
            source_frame_emb: [B, T, L, D] source frame features for DiT cross-attn
            source_frames: [B, T, C, H, W] raw source frames for frozen source encoder

        Returns:
            [B, T*Pt, C, H, W] generated video
        """
        dit_kwargs = {}
        if self._decoder_is_dit:
            dit_kwargs["cls_emb"] = cls_emb
            if source_frames is not None:
                dit_kwargs["source_frames"] = source_frames
            else:
                dit_kwargs["source_frame_emb"] = source_frame_emb

        if self.decoder.decode_mode == "diffusion":
            assert target_shape is not None, "target_shape required for diffusion mode"
            device = frames_output.device
            b = frames_output.shape[0]

            xt = torch.randn(target_shape, device=device)
            dt = 1.0 / num_steps

            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t = torch.full((b,), t_val, device=device)

                velocity = self.decoder(
                    frames_output, motion_output, xt=xt, t=t, **dit_kwargs
                )

                xt = xt - dt * velocity

            return xt
        else:
            return self.decoder(frames_output, motion_output, **dit_kwargs)

    @torch.no_grad()
    def _encode_targets(self, target_v: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space targets to decoder AE latent space for loss computation.

        When decoder_ae_enc is available and input is pixel space, we need targets
        in the decoder's latent space (e.g. F8C4) to match the decoder output.
        """
        if self.decoder_ae_enc is None:
            return target_v

        # Run VAE encoder in fp32, disable autocast
        with torch.autocast(device_type=target_v.device.type, enabled=False):
            target_v = target_v.float()
            if self._decoder_ae_mode == "image":
                b, t, c, h, w = target_v.shape
                flat = target_v.reshape(b * t, c, h, w)
                enc_out = self.decoder_ae_enc(flat)
                if isinstance(enc_out, tuple):
                    enc_out = enc_out[0]  # mean only
                result = enc_out.reshape(b, t, *enc_out.shape[1:])
            else:
                enc_out = self.decoder_ae_enc(target_v)
                if isinstance(enc_out, tuple):
                    enc_out = enc_out[0]
                result = enc_out

        if self.latent_mean is not None:
            result = (result - self.latent_mean) / self.latent_std
        return result

    @torch.no_grad()
    def _decode_output(self, motion_recon: torch.Tensor, key_frame: torch.Tensor) -> torch.Tensor:
        """
        Decode motion frame reconstruction from latent/internal space to pixel space.

        Handles three decoder paths:
        1. decoder_ae (separate image VAE): decode each frame independently
        2. latent_dec (video AE decoder): prepend key frame, decode full sequence
        3. No decoder (pixel input, no decoder_ae): output is already pixel space

        Args:
            motion_recon: [B, T, C, H, W] decoder output (motion frames only)
            key_frame: [B, 1, C, H, W] key frame (latent or pixel)

        Returns:
            [B, T, C, H', W'] decoded motion frames in pixel space
        """
        # Cast to fp32 first, then undo normalization — avoid fp16 overflow
        # during unnormalization (latent_std can be ~21).
        # Disable autocast so the VAE decoder runs in full fp32.
        motion_recon = motion_recon.float()

        if self.latent_mean is not None:
            motion_recon = motion_recon * self.latent_std + self.latent_mean

        # Pick which decoder to use: decoder_ae takes priority over latent_dec
        dec = self.decoder_ae_dec if self.decoder_ae_dec is not None else self.latent_dec
        if dec is None:
            return motion_recon.cpu()

        mode = self._decoder_ae_mode if self.decoder_ae_dec is not None else "video"
        if self.decoder_ae_dec is None and self.latent_dec is not None:
            mode = self._encoder_ae_mode

        if hasattr(dec, "reset"):
            dec.reset()

        # Disable AMP autocast so decoder runs in full fp32
        with torch.autocast(device_type=motion_recon.device.type, enabled=False):
            if mode == "image":
                b, t, c, h, w = motion_recon.shape
                flat = motion_recon.reshape(b * t, c, h, w)
                decoded_flat = dec(flat)
                decoded = decoded_flat.reshape(b, t, *decoded_flat.shape[1:]).cpu()
            else:
                full_recon = torch.cat([key_frame.float(), motion_recon], dim=1)
                decoded = dec(full_recon).cpu()
                decoded = decoded[:, 1:]

        if hasattr(dec, "reset"):
            dec.reset()
        return decoded

    @torch.no_grad()
    def run_logging(
        self,
        result,
        x: torch.Tensor,
        raw_video: torch.Tensor,
        num_samples: int = 8,
    ):
        """
        Run video generation for logging and save comparison video.

        Video structure:
            x: [B, 1+T_total*Pt, C, H, W] latent/pixel video (may include buffer frames)
            raw_video: [B, source_frames, C, H, W] raw video WITHOUT buffer frames

        For logging we always use k=1 shift (one-step-ahead prediction) regardless
        of training mode. This gives consistent visual comparison across modes.
        For twojump_ar, near-target motion = source motion shifted by 1 (adjacent frames).

        Args:
            result: encoder output (from full x including buffer frames)
            x: [B, 1+T_total*Pt, C, H, W] full latent/pixel video
            raw_video: [B, source_frames, C, H, W] raw video for comparison (truncated)
            num_samples: number of samples to log
        """
        torch.cuda.empty_cache()
        b = min(num_samples, x.shape[0])
        total_frames = x.shape[1]
        num_token_frames = (total_frames - 1) // self.temporal_patch_size  # T_total

        # Compute num_pairs: same logic as training to stay consistent
        if self.ar_shift_range is not None:
            num_pairs = num_token_frames - self.ar_shift_range[1]
        else:
            num_pairs = num_token_frames

        # ============================================================
        # Prepare inputs - mode-aware, using k=1 for logging
        # ============================================================
        all_motion = result.motion_output[:b]

        # Frame embeddings depend on mode
        if self.train_mode in ("regression", "mae"):
            frames_output = result.frames_output[:b, 0:1].expand(
                -1, num_pairs, -1, -1
            )  # [B, N, L, D]
        else:
            # AR modes: source-position frame embeddings
            frames_output = result.frames_output[:b, :num_pairs]  # [B, N, L, D]

        # Target shape for diffusion mode (motion frames only)
        # Must match decoder output space, not input space.
        # When decoder_ae_enc exists, targets are in decoder AE latent space.
        if self.decoder_ae_enc is not None:
            with torch.no_grad():
                probe = x[:1, 1:2]  # single frame to probe output shape
                if self._decoder_ae_mode == "image":
                    enc_out = self.decoder_ae_enc(probe[:, 0])
                    if isinstance(enc_out, tuple):
                        enc_out = enc_out[0]
                    target_spatial = enc_out.shape[1:]  # (C, H', W')
                else:
                    enc_out = self.decoder_ae_enc(probe)
                    if isinstance(enc_out, tuple):
                        enc_out = enc_out[0]
                    target_spatial = enc_out.shape[2:]
            target_shape = (b, num_pairs * self.temporal_patch_size, *target_spatial)
        else:
            target_shape = (b, num_pairs * self.temporal_patch_size, *x.shape[2:])

        # Key frame (latent or pixel, needed for _decode_output)
        key_frame = x[:b, 0:1]  # [B, 1, C, H, W]

        # DiT-specific: prepare cls_emb and source_frame_emb
        dit_kwargs = {}
        if self._decoder_is_dit:
            if result.cls_output is not None:
                if self.train_mode in ("regression", "mae"):
                    dit_kwargs["cls_emb"] = result.cls_output[:b, 1 : 1 + num_pairs]
                else:
                    dit_kwargs["cls_emb"] = result.cls_output[:b, :num_pairs]
            if self.train_mode not in ("mae",):
                if self._decoder_has_source_encoder:
                    dit_kwargs["source_frames"] = raw_video[:b, 0:1].expand(-1, num_pairs, -1, -1, -1)
                else:
                    dit_kwargs["source_frame_emb"] = frames_output

        # ============================================================
        # Generate videos
        # ============================================================
        def _prepare_motion(motion_all, token_count):
            """Prepare motion tokens for a given token count, mode-aware."""
            current = motion_all[:, :, :token_count]
            if self.train_mode in ("regression", "mae"):
                # Motion at target position: m[1:1+N]
                return current[:, 1 : 1 + num_pairs]
            elif self.train_mode == "adaptive_ar":
                # Source motion + delta_t token (k=1 for logging)
                src = current[:, :num_pairs]
                dt_token = self.delta_t_emb(1, b, num_pairs, x.device)
                return torch.cat([src, dt_token], dim=2)
            elif self.train_mode == "twojump_ar":
                # Source + near-target motion (k=1 for logging, so near=m[0:N])
                src = current[:, :num_pairs]
                near = current[:, 0:num_pairs]  # k-1=0 when k=1
                return torch.cat([src, near], dim=2)
            else:
                # autoregressive: source-position motion m[0:N]
                return current[:, :num_pairs]

        if self.token_counts is not None:
            all_recon = []
            for token in self.token_counts:
                torch.cuda.empty_cache()
                current_motion = _prepare_motion(all_motion, token)

                current_recon = self.generate_video(
                    frames_output, current_motion, target_shape=target_shape,
                    **dit_kwargs,
                )

                torch.cuda.empty_cache()
                decoded = self._decode_output(current_recon, key_frame)
                all_recon.append(decoded)
                torch.cuda.empty_cache()
            log_recon = torch.cat(all_recon, dim=-2)
            del all_recon
        else:
            current_motion = _prepare_motion(all_motion, all_motion.shape[2])

            motion_recon = self.generate_video(
                frames_output, current_motion, target_shape=target_shape,
                **dit_kwargs,
            )

            log_recon = self._decode_output(motion_recon, key_frame)

        # ============================================================
        # Log video: compare motion frames (skip key frame from raw_video)
        # raw_video is already truncated to source frames (no buffer)
        # ============================================================
        self.log_video(raw_video[:b, 1:], log_recon)
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx, *args, **kwargs):
        # ============================================================
        # Latent encoding (single pass, includes buffer frames for AR)
        # ============================================================
        with torch.no_grad():
            # Handle dual augmentation: batch is (enc_video, dec_video) tuple
            # or single tensor (default single-aug mode)
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                raw_video = batch[0].clone()  # encoder-augmented
                dec_video = batch[1].clone()  # decoder-augmented
            else:
                raw_video = batch.clone()
                dec_video = None  # single-aug: decoder uses same as encoder
            del batch

            if self._pixel_input:
                # Pixel input mode: no AE encoding, backbone receives raw pixels.
                # Frame count stays as-is (no temporal downsampling).
                x = raw_video
            elif self._encoder_ae_mode == "image":
                # Image AE: encode each frame independently
                b, t, c, h, w = raw_video.shape
                flat = raw_video.reshape(b * t, c, h, w)
                enc_out = self.latent_enc(flat)
                if isinstance(enc_out, tuple):
                    enc_out = enc_out[0]  # mean only (drop logvar)
                x = enc_out.reshape(b, t, *enc_out.shape[1:])
            else:
                # Video AE: encode full sequence with temporal compression
                x = self.latent_enc(raw_video)

            # Dual-aug: encode decoder video separately
            if dec_video is not None:
                self._dec_video = dec_video  # store for decoder target/source
            else:
                self._dec_video = None  # use encoder video for decoder too

            # For logging: truncate raw video to source frames (exclude buffer)
            # Buffer frames are extra frames provided by dataset for AR shift
            if self.source_frame_count is not None:
                raw_video_for_log = raw_video[:, : self.source_frame_count]
            else:
                raw_video_for_log = raw_video

        # ============================================================
        # Frame structure (after latent encoding or raw pixel pass-through):
        #
        #   x: [B, 1 + T_total*Pt, C, H, W] latent or pixel frames
        #      - x[:, 0]: key frame (exempt from temporal downsample)
        #      - x[:, 1:]: motion frames (T_total groups × Pt temporal patch)
        #
        #   With video AE (latent mode), e.g. 16x16x4 AE with FRAME_COUNT=33:
        #      33 raw → 9 latent (1 key + 8 motion) → T=8 token frames
        #
        #   With pixel input mode (no AE), e.g. FRAME_COUNT=8:
        #      8 pixel frames directly (1 key + 7 motion) → T=7 token frames
        #      No AE temporal factor; BUFFER_FRAMES are raw pixel frames.
        #
        #   With buffer frames for AR (e.g., k_max=3, ae_temporal=4 in latent mode):
        #      33+12=45 raw → 12 latent (1 key + 11 motion) → T_total=11
        #      num_pairs = T_total - k_max = 8 (same count as regression)
        #   With buffer frames for AR in pixel mode (e.g., k_max=3):
        #      8+3=11 pixel frames (1 key + 10 motion) → T_total=10
        #      num_pairs = T_total - k_max = 7 (same count as regression)
        #
        # Encoder output:
        #   frames_output: [B, 1+T_total, L, D] - spatial embeddings per frame
        #      - [:, 0]: key frame embedding
        #      - [:, i]: embedding of i-th motion frame (saw frames 0..i via causal attn)
        #   motion_output: [B, 1+T_total, M, D] - motion tokens per frame
        #      - [:, 0]: key frame motion (meaningless, no prior motion)
        #      - [:, i]: motion at position i (saw frames 0..i via causal attn)
        #
        # Training modes (all use single encoder pass):
        #   regression:     f₀ repeated + m[1:T+1]     → x[1:T+1]
        #   autoregressive: f[0:T]      + m[0:T]       → x[1:T+1]     (k=1 fixed)
        #   adaptive_ar:    f[0:N]      + [m[0:N];dt(k)] → x[k:k+N]    (k random, explicit dt token)
        #   twojump_ar:     f[0:N]      + [m[0:N];m[k-1:k-1+N]] → x[k:k+N]  (k random, implicit dt)
        #
        #   where N = num_pairs, k = latent token shift
        #   Regression: motion at TARGET position (m_t predicts x_t)
        #   AR modes:   motion at SOURCE position (m_t predicts x_{t+k})
        # ============================================================
        b, total_frames, c, h, w = x.shape
        num_token_frames = (total_frames - 1) // self.temporal_patch_size  # T_total
        encoder_patch_size = self._encoder_patch_size

        # ============================================================
        # MAE mode: separate training path with encoder-level tube masking
        # ============================================================
        if self.train_mode in ("mae", "mae_diffusion"):
            num_pairs = num_token_frames

            # 1. Compute spatial grid size and generate tube mask
            L = (h // encoder_patch_size) * (w // encoder_patch_size)
            num_visible = int(L * (1 - self.mae_mask_ratio))
            num_visible = max(num_visible, 1)  # keep at least 1 token

            perm = torch.randperm(L, device=x.device)
            spatial_mask = torch.zeros(L, dtype=torch.bool, device=x.device)
            spatial_mask[perm[:num_visible]] = True  # True = visible/keep

            # 2. Encoder forward with spatial mask (reduced tokens)
            result = self.encoder(x, spatial_mask=spatial_mask)
            # result.frames_output: [B, 1+T, L_vis, D]
            # result.motion_output: [B, 1+T, M, D]

            vis_frames = result.frames_output  # [B, 1+T, L_vis, D]
            D = vis_frames.shape[-1]

            # 3. Prepare regression-style inputs: f0 repeated + m[1:T+1]
            vis_frame0 = vis_frames[:, 0:1].expand(
                -1, num_pairs, -1, -1
            )  # [B, N, L_vis, D]

            all_motion = result.motion_output
            if self.token_counts is not None:
                token_count = random.choice(self.token_counts)
                all_motion = all_motion[:, :, :token_count]
            motion_output = all_motion[:, 1 : 1 + num_pairs]  # [B, N, M, D]

            # 4. Create full frame embeddings [B, N, L, D]:
            #    fill visible positions from encoder, masked positions from mask_token
            full_frames = self.mask_token.expand(b, num_pairs, L, -1).clone()
            full_frames[:, :, spatial_mask] = vis_frame0

            # 5. Target (needed before decoder for diffusion mode)
            with torch.no_grad():
                if self._dec_video is not None:
                    target_v = self._dec_video[:, 1 : 1 + num_pairs * self.temporal_patch_size]
                else:
                    target_v = x[:, 1 : 1 + num_pairs * self.temporal_patch_size]
                target_v = self._encode_targets(target_v)

            # 6. DiT kwargs for MAE (no cross-attn context)
            mae_dit_kwargs = {}
            if self._decoder_is_dit:
                if result.cls_output is not None:
                    mae_dit_kwargs["cls_emb"] = result.cls_output[:, 1 : 1 + num_pairs]
                mae_dit_kwargs["source_frame_emb"] = None  # MAE: no cross-attn

            # 7. Decoder forward: regression vs diffusion
            if self.decoder.decode_mode == "diffusion":
                # Flow matching: same as non-MAE path
                if (
                    getattr(self, "diffusion_timestep_sampling", "uniform")
                    == "logit_normal"
                ):
                    t = torch.sigmoid(torch.randn(b, device=x.device))
                else:
                    t = torch.rand(b, device=x.device)

                noise = torch.randn_like(target_v)
                t_expand = t[:, None, None, None, None]
                xt = (1 - t_expand) * target_v + t_expand * noise
                velocity_target = noise - target_v

                pred_velocity = self.decoder(
                    full_frames, motion_output, xt=xt, t=t, **mae_dit_kwargs
                )
                mse_loss = F.mse_loss(pred_velocity, velocity_target)
                frames_recon = pred_velocity  # for cleanup/logging compat
            else:
                frames_recon = self.decoder(full_frames, motion_output, **mae_dit_kwargs)

                # 8. MSE loss ONLY on masked spatial positions
                # Mask is in encoder spatial grid; target may be in a different
                # spatial resolution (e.g. decoder AE latent space).
                ph = h // encoder_patch_size
                pw = w // encoder_patch_size
                masked_positions = ~spatial_mask  # [L], True = masked
                masked_2d = masked_positions.view(ph, pw)  # [ph, pw]

                # Scale mask to match target spatial dims
                target_h, target_w = target_v.shape[-2], target_v.shape[-1]
                scale_h = target_h // ph
                scale_w = target_w // pw
                pixel_mask = masked_2d.repeat_interleave(
                    scale_h, 0
                ).repeat_interleave(
                    scale_w, 1
                )  # [target_h, target_w]
                pixel_mask = pixel_mask[None, None, None, :, :]

                masked_recon = frames_recon[pixel_mask.expand_as(frames_recon)]
                masked_target = target_v[pixel_mask.expand_as(target_v)]
                mse_loss = F.mse_loss(masked_recon, masked_target)

            loss = mse_loss

            if self.ema_loss == -1:
                self.start_global_step = self.global_step
                self.ema_loss = loss.item()
            else:
                ema_decay = min(
                    0.995,
                    (self.global_step - self.start_global_step)
                    / (10 + self.global_step - self.start_global_step),
                )
                self.ema_loss = self.ema_loss * ema_decay + loss.item() * (
                    1 - ema_decay
                )
            self.log("loss", loss, prog_bar=True, on_step=True)
            self.log("mse_loss", mse_loss, prog_bar=True, on_step=True)
            self.log("ema_loss", self.ema_loss, prog_bar=True, on_step=True)

            # Logging: run encoder WITHOUT mask for clean visualization
            if (
                self.global_step % self.log_interval == 0
                and self.global_step != self.previous_global_step
                and self.trainer.is_global_zero
                and self.ref is None
            ):
                self.previous_global_step = self.global_step
                clean_result = self.encoder(x)
                self.run_logging(clean_result, x, raw_video_for_log, num_samples=8)
                del clean_result

            del (
                frames_recon,
                target_v,
                full_frames,
                motion_output,
                result,
                raw_video_for_log,
            )
            return loss

        # ============================================================
        # Non-MAE modes: regression / autoregressive / adaptive_ar / twojump_ar
        # ============================================================

        # --- Determine shift k and number of training pairs ---
        # k: how many latent token positions to shift targets
        # num_pairs: number of (frame_emb, motion_emb) → target pairs
        if self.train_mode in ("adaptive_ar", "twojump_ar"):
            if self._ar_shift_weights is not None:
                k = random.choices(
                    range(self.ar_shift_range[0], self.ar_shift_range[1] + 1),
                    weights=self._ar_shift_weights,
                    k=1,
                )[0]
            else:
                k = random.randint(self.ar_shift_range[0], self.ar_shift_range[1])
            # Subtract k_max to keep num_pairs constant regardless of shift
            num_pairs = num_token_frames - self.ar_shift_range[1]
        elif self.train_mode == "autoregressive":
            k = 1
            num_pairs = num_token_frames
        else:  # regression
            k = 0
            num_pairs = num_token_frames

        result = self.encoder(x)

        # --- Token count selection (M dimension, nested token training) ---
        all_motion = result.motion_output
        if self.token_counts is not None:
            token_count = random.choice(self.token_counts)
            all_motion = all_motion[:, :, :token_count]

        # --- Mode-specific frame/motion/target selection ---
        if self.train_mode == "regression":
            # Regression: f₀ repeated, motion at target position, target = x[1:T+1]
            # Key frame embedding broadcast to all pairs
            frames_output = result.frames_output[:, 0:1].expand(
                -1, num_pairs, -1, -1
            )  # [B, N, L, D]
            # Motion from target positions: m[1], m[2], ..., m[N]
            motion_output = all_motion[:, 1 : 1 + num_pairs]  # [B, N, M, D]
            # Target: latent frames x[1], x[2], ..., x[N]
            target_slice = slice(1, 1 + num_pairs * self.temporal_patch_size)

        elif self.train_mode == "autoregressive":
            # AR: source-position motion, k=1 fixed shift
            # f[0:N] = frame embeddings at source positions
            frames_output = result.frames_output[:, :num_pairs]  # [B, N, L, D]
            # m[0:N] = motion at source positions (m_i has seen frames 0..i)
            motion_output = all_motion[:, :num_pairs]  # [B, N, M, D]
            # Target: x[1:N+1] (one step ahead of source)
            target_slice = slice(1, 1 + num_pairs * self.temporal_patch_size)

        elif self.train_mode == "adaptive_ar":
            # Adaptive AR: source-position motion + delta_t token, random k shift
            # f[0:N] = frame embeddings at source positions
            frames_output = result.frames_output[:, :num_pairs]  # [B, N, L, D]
            # m[0:N] = motion at source positions
            src_motion = all_motion[:, :num_pairs]  # [B, N, M, D]
            # Delta-t token: MLP(sincos(k)) → [B, N, 1, D], tells decoder the jump distance
            dt_token = self.delta_t_emb(k, b, num_pairs, x.device)  # [B, N, 1, D]
            motion_output = torch.cat([src_motion, dt_token], dim=2)  # [B, N, M+1, D]
            # Target: x[k:k+N] (k-step ahead of source, k random per batch)
            target_slice = slice(k, k + num_pairs * self.temporal_patch_size)

        elif self.train_mode == "twojump_ar":
            # Two-jump AR: source + near-target motion, random k shift
            # f[0:N] = frame embeddings at source positions
            frames_output = result.frames_output[:, :num_pairs]  # [B, N, L, D]
            # Source motion: m[0:N]
            src_motion = all_motion[:, :num_pairs]  # [B, N, M, D]
            # Near-target motion: m[k-1:k-1+N] (one step before target)
            # This prevents model from compressing frame info directly into motion
            near_motion = all_motion[:, k - 1 : k - 1 + num_pairs]  # [B, N, M, D]
            # Concatenate along M dimension: decoder handles arbitrary M natively
            motion_output = torch.cat([src_motion, near_motion], dim=2)  # [B, N, 2M, D]
            # Target: x[k:k+N]
            target_slice = slice(k, k + num_pairs * self.temporal_patch_size)

        if self.decoder_mask_ratio > 0:
            num_decode_frames = frames_output.shape[1]
            mask_h = h // self.mask_patch_size
            mask_w = w // self.mask_patch_size
            masks = torch.stack(
                [
                    torch.randperm(mask_h * mask_w, device=x.device)
                    for _ in range(b * num_decode_frames)
                ]
            ).reshape(b, num_decode_frames, mask_h, mask_w)
            masks = masks < (self.decoder_mask_ratio * h * w)
            masks = (
                masks.repeat_interleave(self.mask_patch_size, -1)
                .repeat_interleave(self.mask_patch_size, -2)
                .flatten(-2, -1)[..., None]
            )  # [B, T, L, 1]
            frames_output = frames_output * masks

        # --- Target selection ---
        with torch.no_grad():
            if self.ref is not None:
                # Ref outputs token-level frames [B, 1+T, L, D], not pixel-level
                ref_result = self.ref(x)
                if self.train_mode == "regression":
                    target_v = ref_result.frames_output[:, 1 : 1 + num_pairs]
                else:
                    target_v = ref_result.frames_output[:, k : k + num_pairs]
            else:
                if self._dec_video is not None:
                    # Dual-aug: decoder target from separately augmented video
                    target_v = self._dec_video[:, target_slice]
                else:
                    target_v = x[:, target_slice]
                target_v = self._encode_targets(target_v)

        # ============================================================
        # Decoder forward: regression vs diffusion mode
        # ============================================================

        # DiT-specific kwargs
        dit_kwargs = {}
        if self._decoder_is_dit:
            if result.cls_output is not None:
                if self.train_mode == "regression":
                    dit_kwargs["cls_emb"] = result.cls_output[:, 1 : 1 + num_pairs]
                else:
                    dit_kwargs["cls_emb"] = result.cls_output[:, :num_pairs]
            if self._decoder_has_source_encoder:
                # Dual-aug: decoder source encoder sees decoder-augmented frames
                src_video = self._dec_video if self._dec_video is not None else raw_video
                dit_kwargs["source_frames"] = src_video[:, 0:1].expand(-1, num_pairs, -1, -1, -1)
            else:
                dit_kwargs["source_frame_emb"] = frames_output

        if self.decoder.decode_mode == "diffusion":
            # Flow matching diffusion:
            #   xt = (1-t) * x0 + t * x1  where x0=target, x1=noise
            #   velocity = x1 - x0
            #   model predicts velocity, loss = MSE(pred_v, velocity)

            if (
                getattr(self, "diffusion_timestep_sampling", "uniform")
                == "logit_normal"
            ):
                t = torch.sigmoid(torch.randn(b, device=x.device))
            else:
                t = torch.rand(b, device=x.device)

            noise = torch.randn_like(target_v)
            t_expand = t[:, None, None, None, None]
            xt = (1 - t_expand) * target_v + t_expand * noise

            velocity_target = noise - target_v

            pred_velocity = self.decoder(
                frames_output, motion_output, xt=xt, t=t, **dit_kwargs
            )

            mse_loss = F.mse_loss(pred_velocity, velocity_target)
            frames_recon = pred_velocity
        else:
            frames_recon = self.decoder(frames_output, motion_output, **dit_kwargs)
            mse_loss = F.mse_loss(frames_recon, target_v)

        loss = mse_loss

        # NaN/Inf guard + ema_loss bookkeeping. loss.item() forces a CPU<->GPU
        # sync; with nan_guard_interval > 1 this only happens every N steps
        # (default 1: every step).
        sync_step = (
            self.nan_guard_interval <= 1
            or self.ema_loss == -1
            or self.global_step % self.nan_guard_interval == 0
        )
        loss_val = None
        if sync_step:
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                if self.trainer.is_global_zero:
                    self._dump_instability_diagnostics(
                        loss_val, target_v, pred_velocity if self.decoder.decode_mode == "diffusion" else frames_recon,
                        frames_output, motion_output, result, x,
                    )
                raise RuntimeError(
                    f"NaN/Inf loss detected at step {self.global_step}: loss={loss_val}"
                )

            if self.ema_loss == -1:
                self.start_global_step = self.global_step
                self.ema_loss = loss_val
            else:
                ema_decay = min(
                    0.995,
                    (self.global_step - self.start_global_step)
                    / (10 + self.global_step - self.start_global_step),
                )
                self.ema_loss = self.ema_loss * ema_decay + loss_val * (1 - ema_decay)
        self.log("loss", loss, prog_bar=True, on_step=True)
        self.log("mse_loss", mse_loss, prog_bar=True, on_step=True)
        self.log("ema_loss", self.ema_loss, prog_bar=True, on_step=True)

        # Instability detection: if loss > 3x ema_loss, dump diagnostics
        if (
            loss_val is not None
            and self.ema_loss > 0
            and self.global_step > self.start_global_step + 50
            and loss_val > 5.0 * self.ema_loss
            and self.trainer.is_global_zero
        ):
            self._dump_instability_diagnostics(
                loss_val, target_v, pred_velocity if self.decoder.decode_mode == "diffusion" else frames_recon,
                frames_output, motion_output, result, x,
            )

        if (
            self.global_step % self.log_interval == 0
            and self.global_step != self.previous_global_step
            and self.trainer.is_global_zero
            and self.ref is None
        ):
            self.previous_global_step = self.global_step
            self.run_logging(result, x, raw_video_for_log, num_samples=8)
        del (
            frames_recon,
            target_v,
            frames_output,
            motion_output,
            result,
            raw_video_for_log,
        )
        return loss

    @torch.no_grad()
    def log_video(self, video, video_hat):
        """
        video: [B, T, C, H, W]
        video_hat: [B, T, C, H, W]
        """
        torch.cuda.empty_cache()
        video = video[:8]
        video_hat = video_hat[:8]
        video = video[:, -(video_hat.size(1)) :]
        video_hat = video_hat[:, -(video.size(1)) :]
        result = torch.cat([video.cpu(), video_hat.cpu()], dim=-2)
        result = result.permute(1, 2, 3, 0, 4).flatten(-2, -1).contiguous()
        os.makedirs(f"output/{self.name}", exist_ok=True)
        save_video(result, f"output/{self.name}/{self.global_step}.mp4", fps=8)

    def configure_optimizers(self):
        decoder_lr = self.learning_rate * self.decoder_lr_multiplier

        # Collect non-decoder trainable params
        decoder_ids = {id(p) for p in self.decoder.parameters()}
        other_modules = [m for n, m in self.named_children() if n != "decoder" and any(p.requires_grad for p in m.parameters())]

        # muP param groups: per-layer LR scaling by base_dim/fan_in
        param_groups = []
        for m in other_modules:
            param_groups.extend(mup_param_group(m, self.learning_rate, self.base_dim, self.weight_decay))
        param_groups.extend(mup_param_group(self.decoder, decoder_lr, self.decoder_base_dim, self.weight_decay))

        print(f"[Optimizer] encoder base LR={self.learning_rate}, decoder base LR={decoder_lr} "
              f"({self.decoder_lr_multiplier}x), base_dim={self.base_dim}, {len(param_groups)} param groups")

        extra_opt_kwargs = {}
        if getattr(self, "fused_adam", False):
            extra_opt_kwargs["fused"] = True
            print("[Optimizer] using fused AdamW")
        optimizer = optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            eps=1e-6,
            betas=self.betas,
            weight_decay=self.weight_decay,
            **extra_opt_kwargs,
        )
        scheduler = AnySchedule(optimizer, config=self.scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
