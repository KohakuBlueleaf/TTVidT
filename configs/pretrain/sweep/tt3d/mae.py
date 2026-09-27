"""Sweep (Table 1): TT3D + MAE.

Objective: tube-masked reconstruction (mask ratio 0.75).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

use_config("../../../_base/pretrain.py")

RUN_NAME = "sweep_tt3d_mae"
TRAIN_MODE = 'mae'
