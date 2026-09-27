"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder B, video-pretrained.

Only the decoder size / initialisation differs from the flagship recipe.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "decoder_ablation_decB_video"
MOTION_DECODER_CONFIG = DECODER_B
DECODER_PRETRAINED = f"{HF_DECODERS}/pretrain_video_B_qknorm_nofinal"


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
