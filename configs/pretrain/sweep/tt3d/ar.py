"""Sweep (Table 1): TT3D + naive AR.

Objective: predict frame t+1 from frame t's features and motion tokens.
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

use_config("../../../_base/pretrain.py")

RUN_NAME = "sweep_tt3d_ar"
TRAIN_MODE = 'autoregressive'
