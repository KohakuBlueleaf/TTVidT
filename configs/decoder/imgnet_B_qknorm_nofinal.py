"""Decoder B (1024d, 16 layers, no final norm), ImageNet-pretrained (Table 2)."""

from _base.decoder import *  # noqa: F401,F403

RUN_NAME = "pretrain_imgnet_B_qknorm_nofinal"
PRETRAIN_MODE = "imagenet"
DECODER_CONFIG = {**DECODER_B, "use_final_norm": False}


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
