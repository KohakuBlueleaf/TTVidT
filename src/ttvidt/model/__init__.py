"""Encoders and decoder used in the paper.

- ``DINOv3VidTModel``   TT-VidT encoder (DINOv3 ViT-B/16 spatial path + Temporal
                        Transfer; ``tt_mode="1d"`` for TT1D, ``"3d"`` for TT3D)
- ``VideoMAE3DViTModel`` joint space-time 3D ViT (the ViT3D / VideoMAE baseline)
- ``DisMo2DPlus3DModel`` DINOv3 2D path + 3D transformer blocks (DisMo baseline)
- ``MotionDecoder``     DiT decoder used by every pretraining objective
"""

from .dinov3_vit import DINOv3VidTConfig, DINOv3VidTModel, VideoModelOutputWithMotionTokens
from .dismo_2d3d import DisMo2DPlus3DConfig, DisMo2DPlus3DModel
from .motion_decoder import MotionDecoder
from .videomae_3d import VideoMAE3DConfig, VideoMAE3DViTModel

__all__ = [
    "DINOv3VidTConfig",
    "DINOv3VidTModel",
    "VideoModelOutputWithMotionTokens",
    "DisMo2DPlus3DConfig",
    "DisMo2DPlus3DModel",
    "MotionDecoder",
    "VideoMAE3DConfig",
    "VideoMAE3DViTModel",
]
