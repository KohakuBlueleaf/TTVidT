"""Sweep (Table 1): DisMo-2D3D + Adaptive AR.

Objective: predict frame t+k from frame t's features and motion tokens, k in [1, 3] (Gamma-weighted),
with a jump-length token telling the decoder k.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_dismo2d3d_adaar"
BACKBONE_ARCH = 'dismo_2d3d'
BACKBONE_CONFIG = _base["DISMO_BACKBONE"]
MOTION_ENCODER_CONFIG = {}
TRAIN_MODE = 'adaptive_ar'
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)
AR_SHIFT_SAMPLING = 'gamma'
