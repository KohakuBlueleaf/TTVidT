"""Sweep (Table 1): DisMo-2D3D + naive AR.

Objective: predict frame t+1 from frame t's features and motion tokens.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_dismo2d3d_ar"
BACKBONE_ARCH = 'dismo_2d3d'
BACKBONE_CONFIG = _base["DISMO_BACKBONE"]
MOTION_ENCODER_CONFIG = {}
TRAIN_MODE = 'autoregressive'
