"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder S, random init, regression loss.

Only the decoder size / initialisation differs from the flagship recipe.
"""

from kohakuengine import use_config

_base = use_config("../../_base/pretrain.py").globals_dict

RUN_NAME = "decoder_ablation_decS_rand_reg"
MOTION_DECODER_CONFIG = {**_base["DECODER_S"], 'decode_mode': 'regression'}
DECODER_PRETRAINED = None
