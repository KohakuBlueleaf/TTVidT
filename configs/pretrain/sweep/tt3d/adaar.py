"""Sweep (Table 1): TT3D + Adaptive AR.

Objective: predict frame t+k from frame t's features and motion tokens, k in [1, 3] (Gamma-weighted),
with a jump-length token telling the decoder k.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

use_config("../../../_base/pretrain.py")

RUN_NAME = "sweep_tt3d_adaar"
TRAIN_MODE = 'adaptive_ar'
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)
AR_SHIFT_SAMPLING = 'gamma'
