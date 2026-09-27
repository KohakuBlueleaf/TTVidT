"""Decoder S (768d, 12 layers), ImageNet-pretrained. Sweep default (Table 1)."""

from _base.decoder import *  # noqa: F401,F403

RUN_NAME = "pretrain_imgnet_S_qknorm"
PRETRAIN_MODE = "imagenet"
DECODER_CONFIG = {**DECODER_S, "use_final_norm": True}


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
