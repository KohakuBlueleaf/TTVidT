"""Shared encoder-pretraining recipe (paper Section 4.1 / Appendix C).

Every config in ``configs/pretrain`` starts from this file and overrides only
what its table cell changes (architecture, objective, decoder, augmentation):

    from kohakuengine import use_config

    _base = use_config("../_base/pretrain.py").globals_dict   # path relative to the config
    MOTION_ENCODER_CONFIG = _base["TT1D_ENCODER"]            # named building blocks

Recipe: OpenVid-1M (384 px) + Moments-in-Time v2, 8 frames at 6 fps, 256x256,
8 epochs, global batch 32, AdamW (5e-4, betas 0.9/0.98, wd 0.01), 10k warmup,
cosine decay to 1% of the peak, grad clip 0.1, fp16 mixed precision, muP with
base width 256. Encoders take pixels; the decoder reconstructs targets in the
32x32x4 latent space of a frozen frame VAE.
"""

import os
import random

HF_DECODERS = "KBlueLeaf/TTVidT-decoders"   # pretrained DiT decoders (see configs/decoder)
DINOV3 = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DATA_ROOT = os.environ.get("TTVIDT_DATA", "data")

# ------------------------------------------------------------- architectures
# TT-VidT: DINOv3 ViT-B/16 spatial path + 12 Temporal Transfer layers, K=8
# motion tokens per frame. TT3D downsamples spatial tokens 4x per axis
# (16x16 -> 4x4) and mixes them with the motion tokens in block-causal 3D
# attention; TT1D summarises each frame with cross-attention instead.
TT3D_ENCODER = {
    "temporal_patch_size": 1,
    "temporal_patch_overlap": 0,
    "motion_layers_period": 1,
    "num_motion_tokens": 8,
    "motion_ffn_type": "gelu",
    "tt_mode": "3d",
    "tt_downsample": 4,
}
TT1D_ENCODER = {
    "temporal_patch_size": 1,
    "temporal_patch_overlap": 0,
    "motion_layers_period": 1,
    "num_motion_tokens": 8,
    "motion_ffn_type": "gelu",
    "tt_mode": "1d",
}
# ViT3D / VideoMAE-style joint space-time ViT, 24 layers, trained from scratch.
VIT3D_BACKBONE = {
    "in_channels": 3,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 24,
    "num_attention_heads": 12,
    "spatial_patch_size": 16,
    "temporal_patch_size": 1,
    "temporal_patch_overlap": 0,
    "num_motion_tokens": 1,
    "motion_layers_period": 0,
    "ffn_type": "gelu",
}
# DisMo-style 2D+3D: 12 DINOv3-initialised per-frame layers + 12 3D layers.
DISMO_BACKBONE = {
    "init_from_dino": DINOV3,
    "in_channels": 3,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "spatial_patch_size": 16,
    "num_spatial_layers": 12,
    "num_temporal_layers": 12,
    "temporal_patch_size": 1,
    "temporal_patch_overlap": 0,
    "num_motion_tokens": 1,
}

# ------------------------------------------------------------- DiT decoders
_DIT = {"image_dim": 4, "patch_size": 2, "decoder_style": "dit", "decode_mode": "diffusion",
        "qk_norm": True, "attn_bias": False}
DECODER_S = {**_DIT, "num_layers": 12, "hidden_size": 768, "intermediate_size": 3072,
             "num_heads": 12, "use_final_norm": True}
DECODER_B = {**_DIT, "num_layers": 16, "hidden_size": 1024, "intermediate_size": 4096,
             "num_heads": 16, "use_final_norm": False}
DECODER_L = {**_DIT, "num_layers": 28, "hidden_size": 1152, "intermediate_size": 3456,
             "num_heads": 16, "use_final_norm": True}

# ------------------------------------------------------------- run
SEED = 42
RUN_ID = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))

# ------------------------------------------------------------- data
DATASET_FOLDERS = [f"{DATA_ROOT}/openvid384-tar", f"{DATA_ROOT}/moments_in_time-tar"]
DATASET_TYPE = "tar"
FRAME_COUNT = 8
SAMPLE_FPS = 6
NUM_WORKERS = 16

# ------------------------------------------------------------- model defaults
# (the sweep cell "TT3D + Diff Compression, decoder S ImageNet-pretrained")
BACKBONE_ARCH = "ttvidt"
BASE_MODEL_NAME = DINOV3
BACKBONE_CONFIG = None
MOTION_ENCODER_CONFIG = TT3D_ENCODER
MOTION_DECODER_CONFIG = DECODER_S
DECODER_PRETRAINED = f"{HF_DECODERS}/pretrain_imgnet_S_qknorm"
DECODER_AE_PATH = os.environ.get("TTVIDT_FRAME_VAE", "KBlueLeaf/latentmaid-vae")
LATENT_MEAN = [-0.69, -0.48, -0.60, 0.28]
LATENT_STD = [12.38, 11.22, 7.93, 21.22]
TOKEN_COUNTS = [1]
TRAIN_MODE = "regression"   # Diff Compression when paired with a diffusion decoder
UNFREEZE_BACKBONE = True
GRAD_CKPT = False

# ------------------------------------------------------------- optimisation
LEARNING_RATE = 5e-4
BASE_DIM = 256
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
BASE_SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "end": -1, "min_value": 0.01, "warmup": 10000}}
GRAD_CLIP_VAL = 0.1
PRECISION = "16-mixed"

# ------------------------------------------------------------- hardware
GPUS = [0, 1]          # per-device batch 16 x 2 devices = global batch 32
BATCH_SIZE = 16
GRAD_ACC = 1
EPOCH = 8
LOG_INTERVAL = 2500
CKPT_INTERVAL = 5000
