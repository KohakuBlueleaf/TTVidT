"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder S, ImageNet-pretrained (same run as sweep/tt3d/diffcomp).

Only the decoder size / initialisation differs from the flagship recipe.
"""

from kohakuengine import use_config

use_config("../../_base/pretrain.py")

RUN_NAME = "decoder_ablation_decS_imgnet"
