"""Sweep (Table 1): DisMo-2D3D + MAE-Diff.

Objective: MAE with a diffusion decoder head.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_dismo2d3d_mae_diff"
BACKBONE_ARCH = 'dismo_2d3d'
BACKBONE_CONFIG = _base["DISMO_BACKBONE"]
MOTION_ENCODER_CONFIG = {}
TRAIN_MODE = 'mae_diffusion'
