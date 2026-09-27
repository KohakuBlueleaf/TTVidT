"""Sweep (Table 1): TT3D + MAE-Diff.

Objective: MAE with a diffusion decoder head.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

use_config("../../../_base/pretrain.py")

RUN_NAME = "sweep_tt3d_mae_diff"
TRAIN_MODE = 'mae_diffusion'
