"""Shared DiT decoder pretraining recipe (paper Section 4.1 / Appendix D).

The decoder is pretrained once, before any encoder run, and the exported weights
initialise the decoder of the encoder runs (``DECODER_PRETRAINED``). Two sources:

* ``imagenet``: reconstruct an ImageNet-1k image conditioned on the DINOv3 ViT-B
  class token (AdaRMSNorm); cross-attention is not trained.
* ``video``: sample two frames of a training video 1/6-0.5 s apart; reconstruct the
  later frame given its DINOv3 class token and, through cross-attention, the DINOv3
  spatial features of the earlier frame.

Both use a diffusion head in the 32x32x4 latent space of the frozen frame VAE,
100k steps at global batch 256, AdamW 1e-4, 1k warmup.
"""

import os
import random

from ttvidt.config import Config  # noqa: F401  (re-exported for child configs)

DATA_ROOT = os.environ.get("TTVIDT_DATA", "data")

_DIT = {"image_dim": 4, "patch_size": 2, "decoder_style": "dit", "decode_mode": "diffusion",
        "qk_norm": True, "attn_bias": False}
DECODER_S = {**_DIT, "num_layers": 12, "hidden_size": 768, "intermediate_size": 3072, "num_heads": 12}
DECODER_B = {**_DIT, "num_layers": 16, "hidden_size": 1024, "intermediate_size": 4096, "num_heads": 16}
DECODER_L = {**_DIT, "num_layers": 28, "hidden_size": 1152, "intermediate_size": 3456, "num_heads": 16}

SEED = 3407
RUN_ID = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
BASE_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
AE_MODEL_PATH = os.environ.get("TTVIDT_FRAME_VAE", "KBlueLeaf/latentmaid-vae")
LATENT_MEAN = [-0.69, -0.48, -0.60, 0.28]
LATENT_STD = [12.38, 11.22, 7.93, 21.22]

# data (one of the two is used, depending on PRETRAIN_MODE)
DATASET_PATH = f"{DATA_ROOT}/imagenet-1k-tar"
DATASET_FOLDERS = [f"{DATA_ROOT}/openvid384-tar", f"{DATA_ROOT}/moments_in_time-tar"]
MIN_GAP_SEC = 1 / 6        # frame-pair gap for video pretraining
MAX_GAP_SEC = 0.5
IMAGE_SIZE = 256

TOTAL_STEPS = 100000
LEARNING_RATE = 1e-4
BASE_DIM = 256
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
WARMUP_STEPS = 1000

# global batch 256 = BATCH_SIZE x len(GPUS) x GRAD_ACC. For ImageNet, BATCH_SIZE
# must equal the images per tar shard written by convert_imagenet_to_tar.py (64).
BATCH_SIZE = 64
GPUS = [0, 1]
GRAD_ACC = 2
NUM_WORKERS = 16
GRAD_CKPT = False
LOG_INTERVAL = 500
CKPT_INTERVAL = 5000
