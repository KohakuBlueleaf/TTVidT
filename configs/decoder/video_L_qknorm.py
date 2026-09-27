"""Decoder L (1152d, 28 layers), video-pretrained (Table 2). 4 GPUs x 64."""

from kohakuengine import use_config

_base = use_config("../_base/decoder.py").globals_dict

RUN_NAME = "pretrain_video_L_qknorm"
PRETRAIN_MODE = "video"
DECODER_CONFIG = {**_base["DECODER_L"], "use_final_norm": True}
GPUS = [0, 1, 2, 3]
GRAD_ACC = 1
