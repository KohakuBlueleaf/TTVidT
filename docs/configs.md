# Configs

Runs are driven by Python configs and [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine)
(`pip install kohaku-engine`, installed with this package). A config's UPPERCASE
names replace the defaults at the top of the script it runs:

```bash
kogine run scripts/train/pretrain_encoder.py -c configs/pretrain/sweep/tt1d/tjar.py
```

Configs build on each other with `use_config(path)` (path relative to the config
file): the base's values are inherited and the config's own names win. Shared
recipes live in `configs/_base/`; `use_config(...).globals_dict` exposes the base's
named building blocks:

```python
"""Sweep (Table 1): TT1D + two-jump AR."""
from kohakuengine import use_config

_base = use_config("../../../_base/pretrain.py").globals_dict   # the shared encoder recipe

RUN_NAME = "sweep_tt1d_tjar"
MOTION_ENCODER_CONFIG = _base["TT1D_ENCODER"]
TRAIN_MODE = "twojump_ar"
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)
```

To change a setting (GPUs, batch size, logger, data paths, checkpoints), write your
own config on top of the one you start from rather than editing it. Override the
names the script reads: values a base derives from other names (e.g.
`DATASET_FOLDERS` from `DATA_ROOT`) are computed when the base is loaded, so set
`DATASET_FOLDERS` itself (or the `TTVIDT_DATA` variable), not `DATA_ROOT`.

```python
# configs/my_run.py
from kohakuengine import use_config

use_config("pretrain/sweep/tt1d/tjar.py")
GPUS = [0, 1, 2, 3]
BATCH_SIZE = 8
```

## Encoder pretraining (`configs/pretrain/`)

Every file below was checked against the hyper-parameters stored in the paper's
checkpoint for that experiment.

**Final models (Table 4)**

| Model | Config |
|---|---|
| TT-VidT: TT3D + Diff Compression, decoder S video-pretrained | `ttvidt_tt3d_diffcomp` |
| DisMo-2D3D + AdaAR with dual augmentation | `dismo2d3d_adaar_dual_aug` |
| VideoMAE (ViT3D + MAE) | `sweep/vit3d/mae` |

**Architecture x objective sweep (Table 1)**: decoder S ImageNet-pretrained, no augmentation.

| Architecture | MAE | AdaAR | tjAR | AR | MAE-Diff | DiffComp |
|---|---|---|---|---|---|---|
| ViT3D | `sweep/vit3d/mae` | `sweep/vit3d/adaar` | `sweep/vit3d/tjar` | `sweep/vit3d/ar` | `sweep/vit3d/mae_diff` | `sweep/vit3d/diffcomp` |
| DisMo-2D3D | `sweep/dismo2d3d/mae` | `sweep/dismo2d3d/adaar` | `sweep/dismo2d3d/tjar` | `sweep/dismo2d3d/ar` | `sweep/dismo2d3d/mae_diff` | `sweep/dismo2d3d/diffcomp` |
| TT1D | `sweep/tt1d/mae` | `sweep/tt1d/adaar` | `sweep/tt1d/tjar` | `sweep/tt1d/ar` | `sweep/tt1d/mae_diff` | `sweep/tt1d/diffcomp` |
| TT3D | `sweep/tt3d/mae` | `sweep/tt3d/adaar` | `sweep/tt3d/tjar` | `sweep/tt3d/ar` | `sweep/tt3d/mae_diff` | `sweep/tt3d/diffcomp` |

**Decoder ablation (Table 2)**: TT3D + Diff Compression.

| Size | random + regression | random + diffusion | ImageNet | video |
|---|---|---|---|---|
| S | `decoder_ablation/decS_rand_reg` | `decoder_ablation/decS_rand_diff` | `decoder_ablation/decS_imgnet` | `decoder_ablation/decS_video` |
| B | | | `decoder_ablation/decB_imgnet` | `decoder_ablation/decB_video` |
| L | | | `decoder_ablation/decL_imgnet` | `decoder_ablation/decL_video` |

## Decoder pretraining (`configs/decoder/`)

See [training.md](training.md#decoder-pretraining).

## Evaluation (`configs/eval/`)

| Config | Use |
|---|---|
| `frozen.py` | frozen features for HMDB51, ARID, IARD, Jester, SSv2, Diving48, EK-100 verb |
| `ek100_anticipation.py` | frozen features for EK-100 verb anticipation |

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `TTVIDT_DATA` | `data` | pretraining data root used by `configs/_base` |
| `TTVIDT_FRAME_VAE` | `KBlueLeaf/latentmaid-vae` | frame VAE: diffusers folder, Hub id or `repo:subfolder` |
| `TTVIDT_EVAL_DATA` | `eval-dataset` | benchmark root for `iard_identity.py` |
| `VJEPA2_REPO`, `VJEPA2_CKPT_DIR` | | V-JEPA 2 baseline evaluation |
| `TORCH_COMPILE` | `1` | `0` disables `torch.compile` of the attention kernels |
