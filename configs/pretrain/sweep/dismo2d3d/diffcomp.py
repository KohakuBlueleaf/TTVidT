"""Sweep (Table 1): DisMo-2D3D + Diff Compression.

Objective: reconstruct frame t from the first frame's spatial features (the appearance anchor)
and frame t's motion tokens (TRAIN_MODE="regression" with a diffusion decoder).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_dismo2d3d_diffcomp"
BACKBONE_ARCH = 'dismo_2d3d'
BACKBONE_CONFIG = DISMO_BACKBONE
MOTION_ENCODER_CONFIG = {}


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
