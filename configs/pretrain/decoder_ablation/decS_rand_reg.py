"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder S, random init, regression loss.

Only the decoder size / initialisation differs from the flagship recipe.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "decoder_ablation_decS_rand_reg"
MOTION_DECODER_CONFIG = {**DECODER_S, 'decode_mode': 'regression'}
DECODER_PRETRAINED = None


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
