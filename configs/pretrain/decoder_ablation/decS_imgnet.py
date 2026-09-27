"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder S, ImageNet-pretrained (same run as sweep/tt3d/diffcomp).

Only the decoder size / initialisation differs from the flagship recipe.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "decoder_ablation_decS_imgnet"



def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
