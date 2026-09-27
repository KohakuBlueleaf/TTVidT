"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder S, random init, diffusion loss.

Only the decoder size / initialisation differs from the flagship recipe.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "decoder_ablation_decS_rand_diff"
DECODER_PRETRAINED = None


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
