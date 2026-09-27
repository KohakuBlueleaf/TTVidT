"""Sweep (Table 1): TT3D + Diff Compression.

Objective: reconstruct frame t from the first frame's spatial features (the appearance anchor)
and frame t's motion tokens (TRAIN_MODE="regression" with a diffusion decoder).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "sweep_tt3d_diffcomp"



def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
