"""Sweep (Table 1): TT3D + naive AR.

Objective: predict frame t+1 from frame t's features and motion tokens.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_tt3d_ar"
TRAIN_MODE = 'autoregressive'


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
