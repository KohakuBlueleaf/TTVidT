"""Sweep (Table 1): TT1D + naive AR.

Objective: predict frame t+1 from frame t's features and motion tokens.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_tt1d_ar"
MOTION_ENCODER_CONFIG = _base["TT1D_ENCODER"]
TRAIN_MODE = 'autoregressive'
