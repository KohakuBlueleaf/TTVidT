"""Sweep (Table 1): TT1D + two-jump AR.

Objective: predict frame t+k (k in [1, 3]) from frame t's features plus the motion tokens of frames
t and t+k-1.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_tt1d_tjar"
MOTION_ENCODER_CONFIG = TT1D_ENCODER
TRAIN_MODE = 'twojump_ar'
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)
AR_SHIFT_SAMPLING = 'gamma'


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
