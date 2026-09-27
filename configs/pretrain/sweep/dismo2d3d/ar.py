"""Sweep (Table 1): DisMo-2D3D + naive AR.

Objective: predict frame t+1 from frame t's features and motion tokens.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_dismo2d3d_ar"
BACKBONE_ARCH = 'dismo_2d3d'
BACKBONE_CONFIG = DISMO_BACKBONE
MOTION_ENCODER_CONFIG = {}
TRAIN_MODE = 'autoregressive'


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
