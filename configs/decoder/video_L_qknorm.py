"""Decoder L (1152d, 28 layers), video-pretrained (Table 2). 4 GPUs x 64."""

from _base.decoder import *  # noqa: F401,F403

RUN_NAME = "pretrain_video_L_qknorm"
PRETRAIN_MODE = "video"
DECODER_CONFIG = {**DECODER_L, "use_final_norm": True}
GPUS = [0, 1, 2, 3]
GRAD_ACC = 1


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
