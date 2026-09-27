"""Sweep (Table 1): ViT3D (VideoMAE-style) + Diff Compression.

Objective: reconstruct frame t from the first frame's spatial features (the appearance anchor)
and frame t's motion tokens (TRAIN_MODE="regression" with a diffusion decoder).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_vit3d_diffcomp"
BACKBONE_ARCH = 'videomae_3d'
BASE_MODEL_NAME = None
BACKBONE_CONFIG = VIT3D_BACKBONE
MOTION_ENCODER_CONFIG = {}


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
