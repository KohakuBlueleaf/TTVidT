"""Encoder pretraining (all architecture / objective combinations in the paper).

Run with a config from ``configs/pretrain``::

    ttvidt-run scripts/train/pretrain_encoder.py -c configs/pretrain/ttvidt_tt3d_diffcomp.py

Every UPPERCASE name below is a default that the config overrides. The model is
``ttvidt.trainer.TTVidTrainer``: an encoder (``BACKBONE_ARCH``) that turns an
8-frame clip into per-frame motion tokens, and a DiT decoder that reconstructs
target frames in the latent space of a frozen frame VAE (``DECODER_AE_PATH``)
according to ``TRAIN_MODE``.

Checkpoints are written to ``ttvidt/<RUN_ID>/checkpoints/``: ``epoch=N.ckpt`` at
the end of every epoch plus a step checkpoint every ``CKPT_INTERVAL`` steps.
"""

import copy
import os
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore", ".*functools.partial will be.*")

import lightning.pytorch as pl
import torch.utils.data as data
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from ttvidt.data.augment import default_transform, dual_aug_transforms
from ttvidt.data.dual_aug import DualAugDataset
from ttvidt.data.tar_video import TarVideoDataset
from ttvidt.trainer import TTVidTrainer

# ---------------------------------------------------------------- run / resume
SEED = 42
RUN_NAME = "ttvidt"
RUN_ID = None                   # wandb id + checkpoint dir name; configs generate one
CKPT_PATH = None                # initialise weights from this checkpoint
TRAINER_RESUME = False          # True: also restore optimizer / schedule / step

# ---------------------------------------------------------------- data
DATASET_FOLDERS = ["data/openvid384-tar", "data/moments_in_time-tar"]
DATASET_TYPE = "tar"            # "tar" (tar of JPEG frames) or "mp4" (raw videos)
FRAME_COUNT = 8                 # frames per clip
SAMPLE_FPS = 6                  # frame rate the clip is sampled at
BUFFER_FRAMES = 0               # extra frames loaded beyond FRAME_COUNT
DUAL_AUG = False                # DisMo-style dual augmentation (ttvidt.data.augment)
NUM_WORKERS = 16
NUM_FFMPEG_THREADS = 4          # only for DATASET_TYPE="mp4"

# ---------------------------------------------------------------- model
BACKBONE_ARCH = "ttvidt"        # "ttvidt" (TT1D/TT3D), "videomae_3d", "dismo_2d3d"
BASE_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # spatial init for "ttvidt"
BACKBONE_CONFIG = None          # architecture dict for "videomae_3d" / "dismo_2d3d"
MOTION_ENCODER_CONFIG = {}      # Temporal Transfer settings for "ttvidt"
MOTION_DECODER_CONFIG = {}      # DiT decoder settings
DECODER_PRETRAINED = None       # "<hf_repo>/<name>" of a pretrained decoder, None = random
DECODER_AE_PATH = "KBlueLeaf/latentmaid-vae"  # frozen frame VAE: diffusers folder / repo id (see ttvidt.model.frame_vae)
LATENT_MEAN = None              # per-channel latent normalisation; None = from the VAE config
LATENT_STD = None
TOKEN_COUNTS = [1]

# ---------------------------------------------------------------- objective
TRAIN_MODE = "regression"       # regression (Diff Compression with a diffusion
                                # decoder), autoregressive, adaptive_ar,
                                # twojump_ar, mae, mae_diffusion
MAE_MASK_RATIO = 0.75           # tube masking ratio for mae / mae_diffusion
AR_SHIFT_RANGE = None           # (min_k, max_k) prediction horizon for AdaAR / tjAR
AR_SHIFT_SAMPLING = "gamma"
DECODER_MASK_RATIO = 0.0
MASK_PATCH_SIZE = 4
DIFFUSION_TIMESTEP_SAMPLING = "uniform"
UNFREEZE_BACKBONE = True        # train the DINOv3 spatial path jointly
GRAD_CKPT = False

# ---------------------------------------------------------------- optimisation
LEARNING_RATE = 5e-4
BASE_DIM = 256                  # muP base width
DECODER_BASE_DIM = None
DECODER_LR_MULTIPLIER = 1.0
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
BASE_SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "end": -1, "min_value": 0.01, "warmup": 10000}}
GRAD_CLIP_VAL = 0.1
PRECISION = "16-mixed"

# ---------------------------------------------------------------- hardware / logging
ACCELERATOR = "gpu"
GPUS = [0, 1]                   # device ids; BATCH_SIZE is per device
EPOCH = 8
MAX_STEPS = -1                  # >0 stops early (smoke tests)
BATCH_SIZE = 16
GRAD_ACC = 1
LOGGER = "wandb"                # "wandb" or "csv"
WANDB_PROJECT = "ttvidt"
LOG_INTERVAL = 2500             # steps between logged reconstructions
CKPT_INTERVAL = 5000            # steps between step checkpoints


def build_dataset():
    frame_count = FRAME_COUNT + BUFFER_FRAMES
    if DUAL_AUG:
        encoder_transform, decoder_transform = dual_aug_transforms()
    else:
        encoder_transform, decoder_transform = default_transform(), None

    base_transform = None if DUAL_AUG else encoder_transform
    if DATASET_TYPE == "tar":
        parts = [TarVideoDataset(folder, transform=base_transform, target_length=frame_count,
                                 sample_fps=SAMPLE_FPS) for folder in DATASET_FOLDERS]
    elif DATASET_TYPE == "mp4":
        from ttvidt.data.video import FolderVideoDataset

        parts = [FolderVideoDataset(folder, transform=base_transform, target_length=frame_count,
                                    sample_fps=SAMPLE_FPS, num_ffmpeg_threads=NUM_FFMPEG_THREADS)
                 for folder in DATASET_FOLDERS]
    else:
        raise ValueError(f"unknown DATASET_TYPE {DATASET_TYPE!r}")
    dataset = data.ConcatDataset(parts)
    if DUAL_AUG:
        dataset = DualAugDataset(dataset, encoder_transform, decoder_transform)
    return dataset


def load_frame_vae(spec):
    """Frozen frame VAE whose latents are the reconstruction targets.
    Returns ((encoder, decoder), latent_mean, latent_std)."""
    from ttvidt.model.frame_vae import check_latent_channels, load_frame_vae as _load

    vae = _load(spec)
    check_latent_channels(vae, MOTION_DECODER_CONFIG)
    mean = LATENT_MEAN if LATENT_MEAN is not None else vae.latents_mean
    std = LATENT_STD if LATENT_STD is not None else vae.latents_std
    return (vae.encoder, vae.decoder), mean, std


def model_kwargs(scheduler_config):
    decoder_ae, latent_mean, latent_std = load_frame_vae(DECODER_AE_PATH) if DECODER_AE_PATH else (None, LATENT_MEAN, LATENT_STD)
    return dict(
        base_model_name=BASE_MODEL_NAME if BACKBONE_ARCH == "ttvidt" else None,
        motion_encoder_config=MOTION_ENCODER_CONFIG,
        motion_decoder_config=MOTION_DECODER_CONFIG,
        encoder_model=None,             # pixel input
        decoder_ae=decoder_ae,
        decoder_ae_mode="image",
        latent_mean=latent_mean,
        latent_std=latent_std,
        token_counts=TOKEN_COUNTS,
        gradient_checkpointing=GRAD_CKPT,
        train_mode=TRAIN_MODE,
        mae_mask_ratio=MAE_MASK_RATIO,
        ar_shift_range=AR_SHIFT_RANGE,
        ar_shift_sampling=AR_SHIFT_SAMPLING,
        source_frame_count=FRAME_COUNT,
        unfreeze_backbone=UNFREEZE_BACKBONE,
        decoder_mask_ratio=DECODER_MASK_RATIO,
        mask_patch_size=MASK_PATCH_SIZE,
        diffusion_timestep_sampling=DIFFUSION_TIMESTEP_SAMPLING,
        backbone_arch=BACKBONE_ARCH,
        backbone_config=BACKBONE_CONFIG,
        decoder_pretrained=DECODER_PRETRAINED,
        decoder_lr_multiplier=DECODER_LR_MULTIPLIER,
        name=RUN_NAME,
        log_interval=LOG_INTERVAL,
        learning_rate=LEARNING_RATE,
        base_dim=BASE_DIM,
        decoder_base_dim=DECODER_BASE_DIM,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        scheduler_config=scheduler_config,
    )


def main():
    seed_everything(SEED)
    print("local rank", int(os.environ.get("LOCAL_RANK", 0)))

    loader = data.DataLoader(build_dataset(), batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, drop_last=True, pin_memory=True,
                             persistent_workers=NUM_WORKERS > 0,
                             prefetch_factor=2 if NUM_WORKERS > 0 else None)

    num_devices = GPUS if isinstance(GPUS, int) else len(GPUS)
    training_steps = (len(loader) // num_devices // GRAD_ACC + 1) * EPOCH + 1
    scheduler_config = copy.deepcopy(BASE_SCHEDULER_CONFIG)
    scheduler_config["lr"]["end"] = training_steps  # cosine decays over the whole run
    print("total training steps:", training_steps)

    kwargs = model_kwargs(scheduler_config)
    if CKPT_PATH is not None:
        model = TTVidTrainer.load_from_checkpoint(CKPT_PATH, strict=False, map_location="cpu",
                                                  weights_only=False, **kwargs)
    else:
        model = TTVidTrainer(**kwargs)

    if LOGGER == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=WANDB_PROJECT, name=RUN_NAME, id=RUN_ID, version=RUN_ID)
    else:
        from lightning.pytorch.loggers import CSVLogger

        logger = CSVLogger("ttvidt", name="", version=RUN_ID)

    if num_devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True,
                               ddp_comm_hook=default_hooks.fp16_compress_hook,
                               timeout=timedelta(minutes=60))
    else:
        strategy = "auto"

    callbacks = [
        ModelCheckpoint(every_n_epochs=1, filename="{epoch}", save_top_k=-1),
        ModelCheckpoint(every_n_train_steps=CKPT_INTERVAL, save_top_k=-1),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        max_epochs=EPOCH,
        max_steps=MAX_STEPS,
        devices=GPUS,
        accelerator=ACCELERATOR,
        precision=PRECISION,
        strategy=strategy,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=GRAD_CLIP_VAL,
        accumulate_grad_batches=GRAD_ACC,
        callbacks=callbacks,
    )
    if TRAINER_RESUME and CKPT_PATH is not None:
        trainer.fit(model, loader, ckpt_path=CKPT_PATH, weights_only=False)
    else:
        trainer.fit(model, loader)


if __name__ == "__main__":
    main()
