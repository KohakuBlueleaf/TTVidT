"""DisMo-2D3D + Adaptive AR with DisMo's native dual augmentation (final comparison).

The encoder sees an aggressively augmented view, the decoder reconstructs a mildly
augmented one (ttvidt.data.augment.dual_aug_transforms). Decoder S, ImageNet-pretrained,
no final norm.
"""

from kohakuengine import use_config

_base = use_config("../_base/pretrain.py").globals_dict

RUN_NAME = "dismo2d3d_adaar_dual_aug"
BACKBONE_ARCH = 'dismo_2d3d'
BACKBONE_CONFIG = _base["DISMO_BACKBONE"]
MOTION_ENCODER_CONFIG = {}
TRAIN_MODE = 'adaptive_ar'
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)
AR_SHIFT_SAMPLING = 'gamma'
MOTION_DECODER_CONFIG = {**_base["DECODER_S"], 'use_final_norm': False}
DECODER_PRETRAINED = f"{_base['HF_DECODERS']}/pretrain_imgnet_S_qknorm_nofinal"
DUAL_AUG = True
