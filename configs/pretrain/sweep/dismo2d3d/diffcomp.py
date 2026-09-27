"""Sweep (Table 1): DisMo-2D3D + Diff Compression.

Objective: reconstruct frame t from the first frame's spatial features (the appearance anchor)
and frame t's motion tokens (TRAIN_MODE="regression" with a diffusion decoder).
Decoder S, ImageNet-pretrained; no augmentation.
"""

from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict

RUN_NAME = "sweep_dismo2d3d_diffcomp"
BACKBONE_ARCH = 'dismo_2d3d'
BACKBONE_CONFIG = _base["DISMO_BACKBONE"]
MOTION_ENCODER_CONFIG = {}
