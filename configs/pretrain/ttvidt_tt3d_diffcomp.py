"""TT-VidT (proposed): TT3D + Diff Compression, decoder S video-pretrained.

The model reported in the final comparison (Table 4).
"""

from kohakuengine import use_config

_base = use_config("../_base/pretrain.py").globals_dict

RUN_NAME = "ttvidt_tt3d_diffcomp"
DECODER_PRETRAINED = f"{_base['HF_DECODERS']}/pretrain_video_S_qknorm"
