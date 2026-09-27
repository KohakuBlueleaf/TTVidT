"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder S, random init, diffusion loss.

Only the decoder size / initialisation differs from the flagship recipe.
"""

from kohakuengine import use_config

use_config("../../_base/pretrain.py")

RUN_NAME = "decoder_ablation_decS_rand_diff"
DECODER_PRETRAINED = None
