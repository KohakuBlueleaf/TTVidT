"""Decoder B (1024d, 16 layers, no final norm), ImageNet-pretrained (Table 2)."""

from kohakuengine import use_config

_base = use_config("../_base/decoder.py").globals_dict

RUN_NAME = "pretrain_imgnet_B_qknorm_nofinal"
PRETRAIN_MODE = "imagenet"
DECODER_CONFIG = {**_base["DECODER_B"], "use_final_norm": False}
