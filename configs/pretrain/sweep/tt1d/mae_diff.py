"""Sweep (Table 1): TT1D + MAE-Diff.

Objective: MAE with a diffusion decoder head.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_tt1d_mae_diff"
MOTION_ENCODER_CONFIG = TT1D_ENCODER
TRAIN_MODE = 'mae_diffusion'


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
