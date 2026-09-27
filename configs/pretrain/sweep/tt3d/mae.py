"""Sweep (Table 1): TT3D + MAE.

Objective: tube-masked reconstruction (mask ratio 0.75).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_tt3d_mae"
TRAIN_MODE = 'mae'


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
