"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder B, ImageNet-pretrained.

Only the decoder size / initialisation differs from the flagship recipe.
"""

from kohakuengine import use_config

_base = use_config("../../_base/pretrain.py").globals_dict

RUN_NAME = "decoder_ablation_decB_imgnet"
MOTION_DECODER_CONFIG = _base["DECODER_B"]
DECODER_PRETRAINED = f"{_base['HF_DECODERS']}/pretrain_imgnet_B_qknorm_nofinal"
