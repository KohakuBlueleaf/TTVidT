"""Sweep (Table 1): ViT3D (VideoMAE-style) + MAE.

Objective: tube-masked reconstruction (mask ratio 0.75).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_vit3d_mae"
BACKBONE_ARCH = 'videomae_3d'
BASE_MODEL_NAME = None
BACKBONE_CONFIG = VIT3D_BACKBONE
MOTION_ENCODER_CONFIG = {}
TRAIN_MODE = 'mae'


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
