"""Decoder B (1024d, 16 layers, no final norm), video-pretrained (Table 2)."""

from _base.decoder import *  # noqa: F401,F403

RUN_NAME = "pretrain_video_B_qknorm_nofinal"
PRETRAIN_MODE = "video"
DECODER_CONFIG = {**DECODER_B, "use_final_norm": False}


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
