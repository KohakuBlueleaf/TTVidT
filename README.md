# [NeurIPS 2026] TT-VidT

**Decoupling the Temporal Axis for Efficient Motion-Centric Video Pretraining**

> Accepted at **NeurIPS 2026** (main track).

TT-VidT is a self-supervised video encoder built for *motion*. A DINOv3 ViT-B/16
processes every frame independently (the appearance path), while a compact
**Temporal Transfer** pathway turns each frame into a handful of motion tokens that
exchange information across time. It is pretrained with **Diff Compression**: a DiT
decoder must reconstruct every later frame from the *first* frame's spatial features
plus that frame's motion tokens, so the motion tokens are forced to carry exactly
what the first frame cannot explain.

This repository contains everything used for the paper: the encoders (TT-VidT in
TT1D / TT3D form and the ViT3D and DisMo-2D3D baselines), the six pretraining
objectives, decoder pretraining, the data pipeline, frozen-probe and fine-tuning
evaluation, and one config per experiment in the paper.

```
video (8 frames) ──► DINOv3 ViT-B/16, per frame ──────────────► spatial features z_t
                          │  (interleaved)
                          ▼
                 Temporal Transfer (12 layers):  K=8 motion tokens per frame,
                 block-causal attention over [motion tokens ; 4x-downsampled z_t]
                          │
                          ▼
                 motion tokens m_t ──┐
        first-frame features z_1 ────┴──► DiT decoder ──► frame t   (Diff Compression)
```

## Installation

```bash
git clone https://github.com/KohakuBlueleaf/TTVidT && cd TTVidT
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install -e .
```

Python >= 3.10. The DINOv3 weights (`facebook/dinov3-vitb16-pretrain-lvd1689m`) are
gated on the Hugging Face Hub: accept the license there and `huggingface-cli login`
before training. Loading a trained checkpoint does not need them.

## Using a trained model

```python
import torch
from ttvidt.hub import load_model

model = load_model("ttvidt/<run_id>/checkpoints/epoch=7.ckpt", device="cuda", with_decoder=False)
video = torch.rand(1, 8, 3, 256, 256, device="cuda") * 2 - 1     # [B, T, C, H, W] in [-1, 1]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    out = model.encoder(video)
motion = out.motion_output    # [B, T, 1, 768]  pooled motion embedding per frame
```

Frames should be resized to 256x256 and normalised with mean = std = 0.5. See
[docs/model.md](docs/model.md) for the architecture and parameter counts.

## Reproducing the paper

| Step | Guide |
|---|---|
| 1. Prepare pretraining data (OpenVid-1M 384px + Moments-in-Time) and the benchmarks | [docs/data.md](docs/data.md) |
| 2. (Optional) pretrain the DiT decoders; the pretrained ones on the Hub ([`KBlueLeaf/TTVidT-decoders`](https://huggingface.co/KBlueLeaf/TTVidT-decoders)) are used by default | [docs/training.md](docs/training.md#decoder-pretraining) |
| 3. Pretrain encoders: one config per table cell | [docs/training.md](docs/training.md) |
| 4. Frozen-probe evaluation, fine-tuning, diagnostics | [docs/evaluation.md](docs/evaluation.md) |

Every experiment is a Python config run by `ttvidt-run`:

```bash
# TT-VidT (TT3D + Diff Compression, decoder S video-pretrained), 2 GPUs
ttvidt-run scripts/train/pretrain_encoder.py -c configs/pretrain/ttvidt_tt3d_diffcomp.py

# frozen evaluation of the result: extract -> check -> probe (3 seeds) -> mean +- std
bash scripts/eval/run_frozen_eval.sh ttvidt/<run_id>/checkpoints/epoch=7.ckpt
```

Which config belongs to which table is listed in [docs/configs.md](docs/configs.md).

## Repository layout

```
src/ttvidt/           model, training and data code (installable package)
  model/              DINOv3VidTModel (TT-VidT), VideoMAE3DViTModel, DisMo2DPlus3DModel,
                      MotionDecoder (DiT), frame VAE
  modules/            Temporal Transfer layers (tt.py: TT1D, tt3d.py: TT3D), attention blocks
  trainer.py          TTVidTrainer: objectives, optimisation, EMA, logging
  data/               tar-of-JPEG video datasets, augmentation, decoder-pretraining data
  hub.py              load training checkpoints / exported models
  config.py           Python configs + the `ttvidt-run` launcher
src/vidmet/           benchmark dataset loaders (HMDB51, ARID, IARD, Jester, SSv2, Diving48, EK-100)
configs/
  _base/              shared recipes (encoder pretraining, decoder pretraining)
  pretrain/           encoder pretraining: sweep (Table 1), decoder ablation (Table 2), final models
  decoder/            DiT decoder pretraining
  eval/               feature extraction
scripts/
  train/              pretrain_encoder.py, pretrain_decoder.py, export_decoder.py
  eval/               extract_features.py, probe.py, finetune.py, diagnostics, V-JEPA 2 baseline
  data/               video -> tar conversion; benchmarks/: dataset setup scripts
  analysis/           analytical parameter / FLOP counts
  tools/              smoke test, checkpoint export
docs/                 guides
```

## License

Apache-2.0, see [LICENSE](LICENSE). DINOv3, the datasets and the V-JEPA 2 baseline
are subject to their own licenses.
