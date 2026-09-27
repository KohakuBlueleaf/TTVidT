"""Sweep (Table 1): TT3D + two-jump AR.

Objective: predict frame t+k (k in [1, 3]) from frame t's features plus the motion tokens of frames
t and t+k-1.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

use_config("../../../_base/pretrain.py")

RUN_NAME = "sweep_tt3d_tjar"
TRAIN_MODE = 'twojump_ar'
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)
AR_SHIFT_SAMPLING = 'gamma'
