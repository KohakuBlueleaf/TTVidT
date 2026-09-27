"""Decoder L (1152d, 28 layers), ImageNet-pretrained (Table 2). 4 GPUs x 64."""

from kohakuengine import use_config

_base = use_config("../_base/decoder.py").globals_dict

RUN_NAME = "pretrain_imgnet_L_qknorm"
PRETRAIN_MODE = "imagenet"
DECODER_CONFIG = {**_base["DECODER_L"], "use_final_norm": True}
GPUS = [0, 1, 2, 3]
GRAD_ACC = 1
