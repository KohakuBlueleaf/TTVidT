"""Pretrain DiT decoder as a frame generator.

Two modes:
  A) "imagenet": Single-frame generation conditioned on DINOv3 CLS embedding.
     - DINOv3 CLS → AdaRMSNorm, zero frame_emb, no cross-attention.
  B) "video": Frame prediction conditioned on DINOv3 features from another frame.
     - DINOv3 CLS(frame_TdT) → AdaRMSNorm conditioning
     - DINOv3 features(frame_T) → cross-attention context (source_frame_emb)
     - Target: predict latent of frame_TdT

Decoder-only pretraining happens once; the exported weights initialise the decoder
of every encoder run (``DECODER_PRETRAINED`` in configs/pretrain).

Usage:
  ttvidt-run scripts/train/pretrain_decoder.py -c configs/decoder/imgnet_S_qknorm.py
  ttvidt-run scripts/train/pretrain_decoder.py -c configs/decoder/video_S_qknorm.py
  python scripts/train/export_decoder.py <checkpoint> --output checkpoints/decoders/<name>
"""

import os
import random
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore", ".*functools.partial will be.*")

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.checkpoint as ckpt
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import seed_everything
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torchvision.transforms import transforms as trns
import torchvision.transforms.functional as TF

from ttvidt.data.video_pair import VideoPairDataset
from ttvidt.model.frame_vae import check_latent_channels, load_frame_vae
from ttvidt.model.motion_decoder import MotionDecoder

# ============================================================================
# Defaults (overridden by the config)
# ============================================================================

SEED = 3407
CKPT_PATH = None
RUN_ID = None
TRAINER_RESUME = False

RUN_NAME = "pretrain_decoder"
BASE_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
AE_MODEL_PATH = "KBlueLeaf/latentmaid-vae"  # frozen frame VAE (see ttvidt.model.frame_vae)

DECODER_CONFIG = {
    "image_dim": 4,
    "patch_size": 2,
    "decoder_style": "dit",
    "num_layers": 12,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_heads": 12,
    "decode_mode": "diffusion",
}

LATENT_MEAN = [-0.69, -0.48, -0.60, 0.28]
LATENT_STD = [12.38, 11.22, 7.93, 21.22]

# Mode: "imagenet" or "video"
PRETRAIN_MODE = "imagenet"

# ImageNet config
DATASET_PATH = "data/imagenet-1k-tar"  # shards from scripts/data/convert_imagenet_to_tar.py

# Video config
DATASET_FOLDERS = [
    "data/openvid384-tar",
    "data/moments_in_time-tar",
]
MIN_GAP_SEC = 1 / 6   # min temporal gap in seconds (1 frame at fps6)
MAX_GAP_SEC = 0.5     # max temporal gap in seconds (3 frames at fps6)

IMAGE_SIZE = 256
TOTAL_STEPS = 200000
LEARNING_RATE = 5e-4
BASE_DIM = 256
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
WARMUP_STEPS = 10000

GPUS = [0]
BATCH_SIZE = 64
GRAD_ACC = 4
NUM_WORKERS = 16
GRAD_CKPT = False

LOG_INTERVAL = 500
CKPT_INTERVAL = 5000
LOGGER = "wandb"                 # "wandb" or "csv"
WANDB_PROJECT = "ttvidt-decoder"

# Improved decoder options (all optional)
LOGIT_NORMAL_SAMPLING = False   # sigmoid(randn) timestep sampling
COSINE_LOSS = False             # cosine distance loss instead of MSE
LOG_LOSS_C = None               # log(loss + C) weighting, e.g. 1e-3


# ============================================================================
# Trainer
# ============================================================================


class FrameDiTPretrainer(pl.LightningModule):
    def __init__(
        self,
        dino_model,
        ae_encoder,
        ae_decoder,
        decoder_config,
        image_dim,
        latent_h,
        latent_w,
        latent_mean=None,
        latent_std=None,
        pretrain_mode="imagenet",
        name="pretrain_decoder",
        log_interval=500,
        total_steps=200000,
        learning_rate=5e-4,
        base_dim=256,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        warmup_steps=10000,
        gradient_checkpointing=False,
        logit_normal_sampling=False,
        cosine_loss=False,
        log_loss_c=None,
    ):
        super().__init__()

        # Save full decoder_config before .pop() mutates it
        full_decoder_config = decoder_config.copy()

        self.dino = dino_model.eval().requires_grad_(False)
        self.ae_enc = ae_encoder.eval().requires_grad_(False)
        self.ae_dec = ae_decoder.eval().requires_grad_(False)

        self.image_dim = image_dim
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.log_interval = log_interval
        self.name = name
        self.pretrain_mode = pretrain_mode

        if latent_mean is not None and latent_std is not None:
            self.register_buffer("latent_mean", torch.tensor(latent_mean).view(1, 1, -1, 1, 1))
            self.register_buffer("latent_std", torch.tensor(latent_std).view(1, 1, -1, 1, 1))
        else:
            self.latent_mean = None
            self.latent_std = None

        encoder_hidden_size = dino_model.config.hidden_size
        self._D_enc = encoder_hidden_size

        dit_image_dim = decoder_config.pop("image_dim", image_dim)
        dit_patch_size = decoder_config.pop("patch_size", 1)
        self.dit_patch_size = dit_patch_size
        self.token_h = latent_h // dit_patch_size
        self.token_w = latent_w // dit_patch_size

        self.decoder = MotionDecoder(
            image_dim=dit_image_dim,
            patch_size=dit_patch_size,
            temporal_patch_size=1,
            encoder_hidden_size=encoder_hidden_size,
            **decoder_config,
        )
        self.decoder.gradient_checkpointing = gradient_checkpointing

        # Save hparams with full (un-popped) decoder_config
        self.save_hyperparameters(ignore=["dino_model", "ae_encoder", "ae_decoder", "decoder_config"])
        self.hparams["decoder_config"] = full_decoder_config
        self.hparams["encoder_hidden_size"] = encoder_hidden_size

        self._lr = learning_rate
        self._base_dim = base_dim
        self._weight_decay = weight_decay
        self._betas = betas
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps
        self._step_count = 0

        # Improved decoder options
        self.logit_normal_sampling = logit_normal_sampling
        self.cosine_loss = cosine_loss
        self.log_loss_c = log_loss_c

        # Delta-t embedding for video mode: sincos(dT_seconds) → MLP → [D] token
        if pretrain_mode == "video":
            self.delta_t_emb = self._build_delta_t_emb(encoder_hidden_size)
        else:
            self.delta_t_emb = None

    @staticmethod
    def _build_delta_t_emb(hidden_size, max_period=10000):
        import math
        half_dim = hidden_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        torch.nn.init.zeros_(proj[-1].weight)
        torch.nn.init.zeros_(proj[-1].bias)
        emb = torch.nn.Module()
        emb.register_buffer("freqs", freqs)
        emb.proj = proj
        return emb

    def _encode_delta_t(self, dt_seconds):
        """Encode dT (seconds, float) → [B, 1, 1, D] token."""
        # dt_seconds: [B]
        args = dt_seconds.unsqueeze(-1) * self.delta_t_emb.freqs  # [B, D/2]
        sincos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, D]
        token = self.delta_t_emb.proj(sincos)  # [B, D]
        return token.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D]

    def _sample_t(self, B, device):
        """Sample timesteps: logit-normal or uniform."""
        if self.logit_normal_sampling:
            return torch.sigmoid(torch.randn(B, device=device))
        return torch.rand(B, device=device)

    def _compute_loss(self, pred, target):
        """Compute loss with optional cosine distance and log weighting."""
        if self.cosine_loss:
            # Per-sample cosine distance: 1 - cos_sim
            cos_sim = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1)
            loss_per_sample = 1 - cos_sim  # [B]
        else:
            # Per-sample MSE
            loss_per_sample = (pred - target).pow(2).flatten(1).mean(1)  # [B]

        if self.log_loss_c is not None:
            loss_per_sample = torch.log(loss_per_sample + self.log_loss_c)

        return loss_per_sample.mean()

    def _encode_latent(self, images):
        """Encode [B, 3, H, W] images to normalized latent [B, 1, C, h, w]."""
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            latent = self.ae_enc(images.float())
            if isinstance(latent, tuple):
                latent = latent[0]
            if latent.dim() == 4:
                latent = latent.unsqueeze(1)
        if self.latent_mean is not None:
            latent = (latent - self.latent_mean) / self.latent_std
        return latent

    def _get_dino_features(self, images):
        """Get DINOv3 CLS token and spatial features."""
        with torch.no_grad():
            out = self.dino(images)
            hidden = out.last_hidden_state  # [B, prefix+L, D]
            cls_emb = hidden[:, 0]  # [B, D]
            # DINOv3 has 5 prefix tokens (1 CLS + 4 registers), spatial starts at index 5
            spatial = hidden[:, 5:]  # [B, L, D]
        return cls_emb, spatial

    def forward(self, batch):
        if self.pretrain_mode == "imagenet":
            return self._forward_imagenet(batch)
        else:
            return self._forward_video(batch)

    def _forward_imagenet(self, images):
        """Mode A: single-frame, DINOv3 CLS conditioning only."""
        B = images.shape[0]
        device = images.device

        cls_emb, _ = self._get_dino_features(images)
        x0 = self._encode_latent(images)

        # Decoder: zero frame_emb, no cross-attn
        L = self.token_h * self.token_w
        frame_emb = torch.zeros(B, 1, L, self._D_enc, device=device, dtype=images.dtype)
        motion_emb = torch.zeros(B, 1, 0, self._D_enc, device=device, dtype=images.dtype)

        if self.decoder.decode_mode == "diffusion":
            # Flow matching
            t = self._sample_t(B, device)
            noise = torch.randn_like(x0)
            xt = (1 - t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * noise
            velocity_target = noise - x0

            pred_v = self.decoder(
                frame_emb=frame_emb, motion_emb=motion_emb,
                h=self.token_h, w=self.token_w,
                xt=xt, t=t,
                cls_emb=cls_emb.unsqueeze(1),
                source_frame_emb=None,
            )
            loss = self._compute_loss(pred_v, velocity_target)
        else:
            # Regression: directly predict x0, no noise/timestep
            pred = self.decoder(
                frame_emb=frame_emb, motion_emb=motion_emb,
                h=self.token_h, w=self.token_w,
                cls_emb=cls_emb.unsqueeze(1),
                source_frame_emb=None,
            )
            loss = self._compute_loss(pred, x0)

        return loss, images

    def _forward_video(self, batch):
        """Mode B: predict frame_TdT from DINOv3 features of frame_T."""
        frame_T, frame_TdT, dt_seconds = batch  # each [B, 3, H, W], dt [B]
        B = frame_T.shape[0]
        device = frame_T.device

        # DINOv3: source features from frame_T, CLS from frame_TdT
        _, source_spatial = self._get_dino_features(frame_T)  # [B, L_dino, D]
        cls_emb, _ = self._get_dino_features(frame_TdT)  # [B, D]

        # Target: latent of frame_TdT
        x0 = self._encode_latent(frame_TdT)

        # Decoder: zero frame_emb, dT token as motion_emb, source features as cross-attn
        L = self.token_h * self.token_w
        frame_emb = torch.zeros(B, 1, L, self._D_enc, device=device, dtype=frame_T.dtype)
        motion_emb = self._encode_delta_t(dt_seconds)  # [B, 1, 1, D]

        # source_frame_emb needs [B, T=1, L_dino, D]
        source_frame_emb = source_spatial.unsqueeze(1)

        if self.decoder.decode_mode == "diffusion":
            # Flow matching
            t = self._sample_t(B, device)
            noise = torch.randn_like(x0)
            xt = (1 - t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * noise
            velocity_target = noise - x0

            pred_v = self.decoder(
                frame_emb=frame_emb, motion_emb=motion_emb,
                h=self.token_h, w=self.token_w,
                xt=xt, t=t,
                cls_emb=cls_emb.unsqueeze(1),
                source_frame_emb=source_frame_emb,
            )
            loss = self._compute_loss(pred_v, velocity_target)
        else:
            # Regression: directly predict x0, no noise/timestep
            pred = self.decoder(
                frame_emb=frame_emb, motion_emb=motion_emb,
                h=self.token_h, w=self.token_w,
                cls_emb=cls_emb.unsqueeze(1),
                source_frame_emb=source_frame_emb,
            )
            loss = self._compute_loss(pred, x0)

        return loss, frame_TdT

    def training_step(self, batch, batch_idx):
        import math
        loss, vis_images = self(batch)
        loss_val = loss.item()

        # NaN/Inf guard
        if not math.isfinite(loss_val):
            # Dump decoder param stats for diagnosis
            lines = [f"NaN/Inf loss at step {self._step_count}: loss={loss_val}"]
            for name, p in self.decoder.named_parameters():
                if p.requires_grad and p.numel() > 0:
                    pdata = p.data.float()
                    lines.append(
                        f"  {name}: shape={list(p.shape)}, "
                        f"absmax={pdata.abs().max().item():.4f}, "
                        f"mean={pdata.mean().item():.6f}, "
                        f"std={pdata.std().item():.6f}, "
                        f"has_nan={pdata.isnan().any().item()}, "
                        f"has_inf={pdata.isinf().any().item()}"
                    )
            print("\n".join(lines), flush=True)
            raise RuntimeError(f"NaN/Inf loss at step {self._step_count}: {loss_val}")

        self.log("train/loss", loss, prog_bar=True)
        self._step_count += 1

        if self._step_count % self.log_interval == 0:
            self._log_reconstruction(batch, vis_images)
        return loss

    @torch.no_grad()
    def _log_reconstruction(self, batch, vis_images):
        """Generate and log source/generated grid to wandb."""
        N = min(8, vis_images.shape[0])
        device = vis_images.device

        # Get conditioning for generation
        if self.pretrain_mode == "imagenet":
            cls_emb, _ = self._get_dino_features(vis_images[:N])
            source_frame_emb = None
            dt_motion_emb = None
        else:
            frame_T, frame_TdT, dt_seconds = batch
            _, source_spatial = self._get_dino_features(frame_T[:N])
            cls_emb, _ = self._get_dino_features(frame_TdT[:N])
            source_frame_emb = source_spatial.unsqueeze(1)
            dt_motion_emb = self._encode_delta_t(dt_seconds[:N])

        cls_cond = cls_emb.unsqueeze(1)

        # Euler ODE: 20 steps
        x0 = self._encode_latent(vis_images[:N])
        x = torch.randn_like(x0)
        steps = 20
        L = self.token_h * self.token_w
        for i in range(steps, 0, -1):
            t_cur = torch.full((N,), i / steps, device=device)
            frame_emb = torch.zeros(N, 1, L, self._D_enc, device=device, dtype=x.dtype)
            if dt_motion_emb is not None:
                motion_emb = dt_motion_emb
            else:
                motion_emb = torch.zeros(N, 1, 0, self._D_enc, device=device, dtype=x.dtype)

            v = self.decoder(
                frame_emb=frame_emb, motion_emb=motion_emb,
                h=self.token_h, w=self.token_w,
                xt=x, t=t_cur, cls_emb=cls_cond,
                source_frame_emb=source_frame_emb,
            )
            x = x - v / steps

        # Unnormalize and decode
        if self.latent_mean is not None:
            x = x * self.latent_std + self.latent_mean

        with torch.autocast(device_type=device.type, enabled=False):
            if hasattr(self.ae_dec, "reset"):
                self.ae_dec.reset()
            generated = self.ae_dec(x[:, 0].float())

        from torchvision.utils import make_grid

        if self.pretrain_mode == "imagenet":
            # 2-row grid: source | generated
            source_imgs = (vis_images[:N].float().clamp(-1, 1) * 0.5 + 0.5).cpu()
            gen_imgs = (generated.clamp(-1, 1) * 0.5 + 0.5).cpu()
            rows = [make_grid(source_imgs, nrow=N, padding=2),
                    make_grid(gen_imgs, nrow=N, padding=2)]
            caption = "Top: source, Bottom: generated"
        else:
            # 3-row grid: frame_T | frame_TdT (target) | predicted frame_TdT
            frame_T, frame_TdT, dt_seconds = batch
            row_T = (frame_T[:N].float().clamp(-1, 1) * 0.5 + 0.5).cpu()
            row_TdT = (frame_TdT[:N].float().clamp(-1, 1) * 0.5 + 0.5).cpu()
            gen_imgs = (generated.clamp(-1, 1) * 0.5 + 0.5).cpu()
            rows = [make_grid(row_T, nrow=N, padding=2),
                    make_grid(row_TdT, nrow=N, padding=2),
                    make_grid(gen_imgs, nrow=N, padding=2)]
            caption = "Row1: frame_T, Row2: frame_T+dT (target), Row3: predicted"

        grid = torch.cat(rows, dim=1)

        if self.logger:
            import wandb
            self.logger.experiment.log({
                "samples/source_vs_generated": wandb.Image(grid, caption=caption),
                "global_step": self._step_count,
            })

    def configure_optimizers(self):
        from anyschedule import AnySchedule
        from ttvidt.trainer import mup_param_group

        param_groups = mup_param_group(
            self.decoder, self._lr, self._base_dim, self._weight_decay,
        )
        optimizer = torch.optim.AdamW(
            param_groups, lr=self._lr, betas=self._betas, weight_decay=self._weight_decay,
        )
        scheduler_config = {
            "lr": {
                "mode": "constant",
                "warmup": self._warmup_steps,
            }
        }
        scheduler = AnySchedule(optimizer, config=scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ============================================================================
# Datasets
# ============================================================================


# ============================================================================
# Main
# ============================================================================


def main():
    seed_everything(SEED)

    mp_ctx = "forkserver" if NUM_WORKERS > 0 else None

    if PRETRAIN_MODE == "imagenet":
        from ttvidt.data.tar_image import TarImageBatchDataset, identity_collate
        dataset = TarImageBatchDataset(
            DATASET_PATH, image_size=IMAGE_SIZE, decode_device="cpu",
        )
        loader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=identity_collate,
            pin_memory=False,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
            multiprocessing_context=mp_ctx,
        )
    else:
        # Video mode: transforms handled inside VideoPairDataset for paired consistency
        dataset = VideoPairDataset(
            DATASET_FOLDERS,
            transform=True,
            min_gap_sec=MIN_GAP_SEC,
            max_gap_sec=MAX_GAP_SEC,
        )
        loader = data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=True,
            pin_memory=False,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
            multiprocessing_context=mp_ctx,
        )

    # Load DINOv3
    from transformers import DINOv3ViTModel
    dino = DINOv3ViTModel.from_pretrained(BASE_MODEL_NAME)

    # Frozen frame VAE: its latents are the reconstruction targets
    vae = load_frame_vae(AE_MODEL_PATH)
    check_latent_channels(vae, DECODER_CONFIG)
    ae_enc, ae_dec, image_dim = vae.encoder, vae.decoder, vae.latent_channels
    with torch.no_grad():
        dummy_latent = ae_enc(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        if isinstance(dummy_latent, tuple):
            dummy_latent = dummy_latent[0]
        _, _, latent_h, latent_w = dummy_latent.shape

    print(f"Latent shape: {image_dim}ch x {latent_h}x{latent_w}")
    print(f"Mode: {PRETRAIN_MODE}")

    # Calculate epochs from total_steps
    num_gpus = GPUS if isinstance(GPUS, int) else len(GPUS)
    steps_per_epoch = len(loader) // num_gpus // GRAD_ACC
    epochs = (TOTAL_STEPS + steps_per_epoch - 1) // steps_per_epoch
    print(f"Steps/epoch: {steps_per_epoch}, epochs: {epochs}, total target: {TOTAL_STEPS}")

    model = FrameDiTPretrainer(
        dino_model=dino,
        ae_encoder=ae_enc,
        ae_decoder=ae_dec,
        decoder_config=DECODER_CONFIG.copy(),
        image_dim=image_dim,
        latent_h=latent_h,
        latent_w=latent_w,
        latent_mean=LATENT_MEAN,
        latent_std=LATENT_STD,
        pretrain_mode=PRETRAIN_MODE,
        name=RUN_NAME,
        log_interval=LOG_INTERVAL,
        total_steps=TOTAL_STEPS,
        learning_rate=LEARNING_RATE,
        base_dim=BASE_DIM,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        warmup_steps=WARMUP_STEPS,
        gradient_checkpointing=GRAD_CKPT,
        logit_normal_sampling=LOGIT_NORMAL_SAMPLING,
        cosine_loss=COSINE_LOSS,
        log_loss_c=LOG_LOSS_C,
    )

    if LOGGER == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=WANDB_PROJECT, name=RUN_NAME, id=RUN_ID, version=RUN_ID)
    else:
        from lightning.pytorch.loggers import CSVLogger

        logger = CSVLogger(WANDB_PROJECT, name="", version=RUN_ID)

    if num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            ddp_comm_hook=default_hooks.fp16_compress_hook,
            process_group_backend="nccl",
            timeout=timedelta(minutes=60),
        )
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        max_steps=TOTAL_STEPS,
        devices=GPUS,
        precision="16-mixed",
        accelerator="gpu",
        strategy=strategy,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=GRAD_ACC,
        callbacks=[
            ModelCheckpoint(every_n_train_steps=CKPT_INTERVAL, save_top_k=-1),
        ],
    )

    if TRAINER_RESUME and CKPT_PATH is not None:
        trainer.fit(model, loader, ckpt_path=CKPT_PATH)
    else:
        trainer.fit(model, loader)


if __name__ == "__main__":
    main()
