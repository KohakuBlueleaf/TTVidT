"""Sweep (Table 1): ViT3D (VideoMAE-style) + MAE.

Objective: tube-masked reconstruction (mask ratio 0.75).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_vit3d_mae"
BACKBONE_ARCH = 'videomae_3d'
BASE_MODEL_NAME = None
BACKBONE_CONFIG = _base["VIT3D_BACKBONE"]
MOTION_ENCODER_CONFIG = {}
TRAIN_MODE = 'mae'
