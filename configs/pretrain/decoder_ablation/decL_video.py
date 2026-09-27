"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder L, video-pretrained.

Only the decoder size / initialisation differs from the flagship recipe.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "decoder_ablation_decL_video"
MOTION_DECODER_CONFIG = DECODER_L
DECODER_PRETRAINED = f"{HF_DECODERS}/pretrain_video_L_qknorm"


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
