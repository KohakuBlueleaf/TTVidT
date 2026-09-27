"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder L, ImageNet-pretrained.

Only the decoder size / initialisation differs from the flagship recipe.
"""

from kohakuengine import use_config

_base = use_config("../../_base/pretrain.py").globals_dict

RUN_NAME = "decoder_ablation_decL_imgnet"
MOTION_DECODER_CONFIG = _base["DECODER_L"]
DECODER_PRETRAINED = f"{_base['HF_DECODERS']}/pretrain_imgnet_L_qknorm"
