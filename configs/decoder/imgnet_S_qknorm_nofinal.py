"""Decoder S without final norm, ImageNet-pretrained. Used by DisMo with dual augmentation."""

from _base.decoder import *  # noqa: F401,F403

RUN_NAME = "pretrain_imgnet_S_qknorm_nofinal"
PRETRAIN_MODE = "imagenet"
DECODER_CONFIG = {**DECODER_S, "use_final_norm": False}


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
