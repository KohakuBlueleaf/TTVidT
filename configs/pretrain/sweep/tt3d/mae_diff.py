"""Sweep (Table 1): TT3D + MAE-Diff.

Objective: MAE with a diffusion decoder head.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_tt3d_mae_diff"
TRAIN_MODE = 'mae_diffusion'


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
