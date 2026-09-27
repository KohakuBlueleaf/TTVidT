"""Decoder ablation (Table 2): TT3D + Diff Compression, decoder S, video-pretrained (the proposed TT-VidT).

Only the decoder size / initialisation differs from the flagship recipe.
"""

from kohakuengine import use_config

_base = use_config("../../_base/pretrain.py").globals_dict

RUN_NAME = "decoder_ablation_decS_video"
DECODER_PRETRAINED = f"{_base['HF_DECODERS']}/pretrain_video_S_qknorm"
