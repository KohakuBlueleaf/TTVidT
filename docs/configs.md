# Configs

A config is a Python file; its UPPERCASE names override the defaults at the top of
the script it runs. Configs share recipes through `configs/_base/`:

```python
"""Sweep (Table 1): TT1D + two-jump AR."""
from _base.pretrain import *          # the shared encoder recipe

RUN_NAME = "sweep_tt1d_tjar"
MOTION_ENCODER_CONFIG = TT1D_ENCODER  # named building blocks from the base
TRAIN_MODE = "twojump_ar"
BUFFER_FRAMES = 3
AR_SHIFT_RANGE = (1, 3)

def config_gen():
    return Config.from_globals()
```

Run a script with a config, optionally overriding any name on the command line
(values are parsed as Python literals, falling back to strings):

```bash
ttvidt-run scripts/train/pretrain_encoder.py -c configs/pretrain/sweep/tt1d/tjar.py GPUS=[0,1,2,3] BATCH_SIZE=8
```

`ttvidt-run` imports the script, sets the config's names on it and calls the
function in its `if __name__ == "__main__":` block. `config_gen()` may also be a
generator that yields one `Config` per run. The launcher follows the same contract
as [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine)'s `kogine run`.

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
