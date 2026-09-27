# Training

Training is two-stage: DiT decoders are pretrained once, then every encoder run
initialises its decoder from one of them. The pretrained decoders are on the
Hugging Face Hub ([`KBlueLeaf/TTVidT-decoders`](https://huggingface.co/KBlueLeaf/TTVidT-decoders)) and are downloaded automatically, so
stage 1 is optional.

Every run is a config executed by `kogine run` (see [configs.md](configs.md)):

```bash
kogine run <script> -c <config>
```

## Encoder pretraining

```bash
kogine run scripts/train/pretrain_encoder.py -c configs/pretrain/ttvidt_tt3d_diffcomp.py
```

The shared recipe is `configs/_base/pretrain.py`:

| Setting | Value |
|---|---|
| Data | OpenVid-1M (384 px) + Moments-in-Time v2, ~1.78M clips |
| Clip | 8 frames at 6 fps, resize 256 + random 256x256 crop, random horizontal / vertical flip |
| Schedule | 8 epochs (~436k steps), global batch 32 (2 GPUs x 16) |
| Optimiser | AdamW, lr 5e-4, betas (0.9, 0.98), weight decay 0.01, muP (base width 256) |
| LR schedule | 10k linear warmup, cosine decay to 1% of the peak |
| Other | grad clip 0.1, fp16 mixed precision, all parameters trained (DINOv3 path included) |
| Decoder | S (768d, 12 layers), ImageNet-pretrained unless the config says otherwise |
| Targets | 32x32x4 latents of the frozen frame VAE (`KBlueLeaf/latentmaid-vae`, see [data.md](data.md#frame-vae)) |

To change anything, write a config on top of the one you start from:

```python
# configs/my_run.py
from kohakuengine import use_config

use_config("pretrain/ttvidt_tt3d_diffcomp.py")

GPUS = [0, 1, 2, 3]   # keep BATCH_SIZE x len(GPUS) x GRAD_ACC = 32
BATCH_SIZE = 8
LOGGER = "csv"        # no Weights & Biases account

# continue a run (weights + optimizer + schedule):
# CKPT_PATH = "ttvidt/<run_id>/checkpoints/epoch=3.ckpt"
# TRAINER_RESUME = True
# RUN_ID = "<run_id>"
```

```bash
kogine run scripts/train/pretrain_encoder.py -c configs/my_run.py
```

Checkpoints go to `ttvidt/<RUN_ID>/checkpoints/`: `epoch=N.ckpt` after each epoch
and `epoch=E-step=S.ckpt` every `CKPT_INTERVAL` steps. The final model is `epoch=7.ckpt`.

Before a long run, check that a config builds and trains on synthetic clips:

```bash
python scripts/tools/smoke_test.py configs/pretrain/ttvidt_tt3d_diffcomp.py --steps 2 [--gpu]
```

## Decoder pretraining

```bash
kogine run scripts/train/pretrain_decoder.py -c configs/decoder/video_S_qknorm.py
python scripts/train/export_decoder.py ttvidt-decoder/<run_id>/checkpoints/epoch=*-step=100000.ckpt \
    --output checkpoints/decoders/pretrain_video_S_qknorm
```

Recipe (`configs/_base/decoder.py`): 100k steps, global batch 256, AdamW lr 1e-4,
1k warmup, diffusion loss in the frame-VAE latent space, DINOv3 ViT-B conditioning.

* ImageNet decoders reconstruct an ImageNet image from its DINOv3 class token.
* Video decoders reconstruct a later frame of a video from its class token plus,
  through cross-attention, the spatial features of an earlier frame (1/6 to 0.5 s
  apart).

To use your own decoder, set `DECODER_PRETRAINED` in your config to the exported
file, e.g. `DECODER_PRETRAINED = "checkpoints/decoders/pretrain_video_S_qknorm"` (a local
`.safetensors`, extension optional). Values of the form `"<hf_repo>/<name>"` load
`<name>.safetensors` from a Hugging Face repo.

| Config | Used by |
|---|---|
| `imgnet_S_qknorm` | the whole sweep (Table 1) |
| `video_S_qknorm` | TT-VidT |
| `imgnet_S_qknorm_nofinal` | DisMo with dual augmentation |
| `imgnet_B_qknorm_nofinal`, `video_B_qknorm_nofinal` | decoder ablation, size B |
| `imgnet_L_qknorm`, `video_L_qknorm` | decoder ablation, size L |

## Compute

One 8-epoch encoder run takes roughly 80-90 GPU-hours on B200 GPUs.
