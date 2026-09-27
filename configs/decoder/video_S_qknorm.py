"""Decoder S (768d, 12 layers), video-pretrained. The TT-VidT default."""

from kohakuengine import use_config

_base = use_config("../_base/decoder.py").globals_dict

RUN_NAME = "pretrain_video_S_qknorm"
PRETRAIN_MODE = "video"
DECODER_CONFIG = {**_base["DECODER_S"], "use_final_norm": True}
