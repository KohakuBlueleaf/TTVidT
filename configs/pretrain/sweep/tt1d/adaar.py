"""Sweep (Table 1): TT1D + Adaptive AR.

Objective: predict frame t+k from frame t's features and motion tokens, k in [1, 3] (Gamma-weighted),
with a jump-length token telling the decoder k.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_tt1d_adaar"
MOTION_ENCODER_CONFIG = TT1D_ENCODER
TRAIN_MODE = 'adaptive_ar'
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)
AR_SHIFT_SAMPLING = 'gamma'


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
