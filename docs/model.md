# Model

## Encoders

All encoders take 8 RGB frames at 256x256 (pixels in [-1, 1]) and output, per
frame, a motion embedding used for pretraining and evaluation
(`output.motion_output`, shape `[B, T, 1, 768]`). The architecture key is
`BACKBONE_ARCH` in the configs.

| Encoder | `BACKBONE_ARCH` | Class | Structure |
|---|---|---|---|
| **TT3D** (TT-VidT) | `ttvidt`, `tt_mode="3d"` | `ttvidt.model.DINOv3VidTModel` | DINOv3 ViT-B/16 spatial path (12 layers, DINOv3-initialised) + 12 Temporal Transfer 3D layers |
| **TT1D** (TT-VidT) | `ttvidt`, `tt_mode="1d"` | same | as above with Temporal Transfer 1D layers |
| **ViT3D** (VideoMAE-style) | `videomae_3d` | `ttvidt.model.VideoMAE3DViTModel` | 24-layer joint space-time ViT, trained from scratch |
| **DisMo-2D3D** | `dismo_2d3d` | `ttvidt.model.DisMo2DPlus3DModel` | 12 DINOv3-initialised per-frame layers + 12 3D layers |

### Temporal Transfer

Each frame gets K=8 learnable motion tokens. There are 12 Temporal Transfer
layers, one after each DINOv3 layer (`motion_layers_period=1`), and the whole
network is trained jointly.

- **TT1D** (`src/ttvidt/modules/tt.py`): the motion tokens of frame t read that
  frame's spatial features through cross-attention, then all T·K motion tokens
  attend to each other under a block-causal mask (frame t sees frames 1..t).
- **TT3D** (`src/ttvidt/modules/tt3d.py`): each frame's 16x16 spatial tokens are
  downsampled 4x per axis to 4x4, concatenated with that frame's motion tokens,
  and one block-causal attention runs over the T·(K+16) = 192 tokens. The result
  is upsampled back and added to the spatial stream.

The spatial down-sampling in TT3D pixel-unshuffles each 4x4 patch and applies a
fixed, parameter-free `D x D·16` weight computed from D and f, followed by a
trainable D x D channel mix; the up-sampling applies the channel mix and the
transpose of the fixed weight.

## Decoder and objectives

The decoder is a DiT (`ttvidt.model.MotionDecoder`, `decoder_style="dit"`) that
works in the 32x32x4 latent space of a frozen f8 frame VAE (patchified 2x2 into
16x16 tokens). It is conditioned on motion tokens (AdaRMSNorm) and cross-attends to
reference-frame features. Sizes: S (768d, 12 layers), B (1024d, 16), L (1152d, 28).
Decoders are pretrained once (see [training.md](training.md#decoder-pretraining))
and loaded via `DECODER_PRETRAINED`.

`TRAIN_MODE` selects the objective (`src/ttvidt/trainer.py`):

| Objective | `TRAIN_MODE` | Decoder input -> target |
|---|---|---|
| Diff Compression | `regression` (+ diffusion decoder) | first-frame features z_1 + motion tokens m_t -> frame t |
| MAE | `mae` | tube-masked input (ratio 0.75) -> frames |
| MAE-Diff | `mae_diffusion` | as MAE, diffusion head |
| naive AR | `autoregressive` | z_t + m_t -> frame t+1 |
| Adaptive AR | `adaptive_ar` | z_t + m_t + jump token(k) -> frame t+k, k in [1, 3] |
| two-jump AR | `twojump_ar` | z_t + m_t + m_{t+k-1} -> frame t+k, k in [1, 3] |

`decode_mode` in the decoder config chooses a diffusion (default) or a regression
head; `regression` with a diffusion decoder is Diff Compression.

## Loading models

```python
from ttvidt.hub import load_model
model = load_model("ttvidt/<run_id>/checkpoints/epoch=7.ckpt", device="cuda")   # encoder + decoder
encoder = load_model("ttvidt/<run_id>/checkpoints/epoch=7.ckpt", with_decoder=False).encoder
```

The architecture is read from the hyper-parameters stored in the checkpoint
(`use_ema=True` loads the EMA weights instead of the raw ones).
`scripts/tools/export_release.py` converts a checkpoint into a self-contained
directory (`config.json` with the full architecture and `TTVidTrainer` arguments,
`model.safetensors` with the weights) that `load_model` accepts as a local path or a
Hugging Face repo id, without downloading the DINOv3 base weights.

Analytical encoder cost at T=8, 256x256 (`python scripts/analysis/params_flops.py`):

| Encoder | Params | Forward GFLOPs |
|---|---:|---:|
| TT3D | 196.4M | 514.1 |
| TT1D | 175.2M | 400.5 |
| ViT3D | 170.5M | 1012.6 |
| DisMo-2D3D | 170.5M | 874.7 |
