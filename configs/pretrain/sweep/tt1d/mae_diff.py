"""Sweep (Table 1): TT1D + MAE-Diff.

Objective: MAE with a diffusion decoder head.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_tt1d_mae_diff"
MOTION_ENCODER_CONFIG = _base["TT1D_ENCODER"]
TRAIN_MODE = 'mae_diffusion'
