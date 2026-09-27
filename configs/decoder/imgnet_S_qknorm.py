"""Decoder S (768d, 12 layers), ImageNet-pretrained. Sweep default (Table 1)."""

from kohakuengine import use_config

_base = use_config("../_base/decoder.py").globals_dict

RUN_NAME = "pretrain_imgnet_S_qknorm"
PRETRAIN_MODE = "imagenet"
DECODER_CONFIG = {**_base["DECODER_S"], "use_final_norm": True}
