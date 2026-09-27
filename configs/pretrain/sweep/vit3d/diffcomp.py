"""Sweep (Table 1): ViT3D (VideoMAE-style) + Diff Compression.

Objective: reconstruct frame t from the first frame's spatial features (the appearance anchor)
and frame t's motion tokens (TRAIN_MODE="regression" with a diffusion decoder).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_vit3d_diffcomp"
BACKBONE_ARCH = 'videomae_3d'
BASE_MODEL_NAME = None
BACKBONE_CONFIG = _base["VIT3D_BACKBONE"]
MOTION_ENCODER_CONFIG = {}
