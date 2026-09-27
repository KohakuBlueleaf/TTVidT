"""Decoder S without final norm, ImageNet-pretrained. Used by DisMo with dual augmentation."""

from kohakuengine import use_config

_base = use_config("../_base/decoder.py").globals_dict

RUN_NAME = "pretrain_imgnet_S_qknorm_nofinal"
PRETRAIN_MODE = "imagenet"
DECODER_CONFIG = {**_base["DECODER_S"], "use_final_norm": False}
