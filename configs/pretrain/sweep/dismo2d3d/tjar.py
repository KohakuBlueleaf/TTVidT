"""Sweep (Table 1): DisMo-2D3D + two-jump AR.

Objective: predict frame t+k (k in [1, 3]) from frame t's features plus the motion tokens of frames
t and t+k-1.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_dismo2d3d_tjar"
BACKBONE_ARCH = 'dismo_2d3d'
BACKBONE_CONFIG = _base["DISMO_BACKBONE"]
MOTION_ENCODER_CONFIG = {}
TRAIN_MODE = 'twojump_ar'
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)
AR_SHIFT_SAMPLING = 'gamma'
