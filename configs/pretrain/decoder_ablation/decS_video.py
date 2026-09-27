"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder S, video-pretrained (the proposed TT-VidT).

Only the decoder size / initialisation differs from the flagship recipe.
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "decoder_ablation_decS_video"
DECODER_PRETRAINED = f"{HF_DECODERS}/pretrain_video_S_qknorm"


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
