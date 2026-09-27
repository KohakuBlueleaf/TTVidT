"""Sweep (Table 1): TT1D + MAE.

Objective: tube-masked reconstruction (mask ratio 0.75).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_tt1d_mae"
MOTION_ENCODER_CONFIG = _base["TT1D_ENCODER"]
TRAIN_MODE = 'mae'
