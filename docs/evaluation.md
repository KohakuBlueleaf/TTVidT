# Evaluation

Every evaluation script accepts any of: a training checkpoint
(`ttvidt/<run_id>/checkpoints/epoch=7.ckpt`), an exported model directory
(`scripts/tools/export_release.py`), or a Hugging Face repo holding one. The architecture is read from the checkpoint, so no model
config is needed.

## Frozen probes (Tables 1, 2 and 4)

The encoder is frozen; per-frame motion embeddings are extracted once and small
probes are trained on top.

```bash
# everything at once: extract -> check -> 3 probe seeds -> mean +- std
bash scripts/eval/run_frozen_eval.sh ttvidt/<run_id>/checkpoints/epoch=7.ckpt my_model
```

or step by step:

```python
# configs/eval/my_model.py
from kohakuengine import use_config

use_config("frozen.py")
CHECKPOINT_PATH = "ttvidt/<run_id>/checkpoints/epoch=7.ckpt"
MODEL_NAME = "my_model"
```

```bash
kogine run scripts/eval/extract_features.py -c configs/eval/my_model.py
python scripts/eval/check_features.py my_model --datasets hmdb51,arid,iard,jester,sthsthv2
python scripts/eval/probe.py my_model --seed 0 --gpu --out eval-results/my_model/my_model__seed0.json
python scripts/eval/aggregate_results.py eval-results/my_model
```

**Extraction** (`configs/eval/frozen.py`): 8 frames sampled uniformly over the
clip, 256x256, pixels in [-1, 1], fp16 autocast. Output per dataset and split:
`features/<model>/<dataset>/{train,test}.pt` with `tokens [N, T, 1, 768]` and
`labels [N]`.

**Probes** (`scripts/eval/probe.py`): the T tokens of a clip form the input sequence.

| Probe | Head |
|---|---|
| `knn` | cosine kNN, k=20, on the mean token |
| `linear_mean` / `linear_lw` | linear on the mean / a learned weighted mean |
| `mlp_mean` / `mlp_lw` | MLP (hidden 512, dropout 0.1) on the mean / weighted mean |
| `attentive` | single-query attention pooling + MLP; the metric reported in the paper |

AdamW, lr 1e-3, weight decay 1e-4, constant LR. HMDB51, ARID, IARD, Diving48:
batch 64 for 100 epochs. Jester, SSv2, EK-100: batch 256 for 20 epochs. Pass
`--seed` for reproducible probe initialisation; results vary by roughly 0.1-1 point
across probe seeds, more on the small datasets.

**Check the features** with `check_features.py` before probing. The loaders skip
missing clips silently, and the check compares clip counts with the complete
benchmarks (see [data.md](data.md#benchmarks)).

## EPIC-Kitchens verb anticipation (Table 4, "EK-V Antic.")

Same as above with `use_config("ek100_anticipation.py")` as the base:

```bash
kogine run scripts/eval/extract_features.py -c configs/eval/my_model_antic.py
python scripts/eval/probe.py my_model --datasets ek100_verb_anticip --seed 0 --gpu
```

Clips are the untrimmed observation windows ending 1 s before each action; 8 frames
at 6 fps, centered.

## Fine-tuning (Table 4, "D48 FT")

```bash
python scripts/eval/finetune.py --checkpoint <checkpoint> --dataset diving48 --gpu
```

The whole encoder is trained with an attentive head: AdamW with encoder LR 1e-5 and
head LR 1e-3, weight decay 0.05, 5 warmup epochs then cosine, effective batch 32
(2 x 16 accumulation), gradient clip 1.0, 50 epochs. Train clips use uniform
sampling and random horizontal flips; test clips use center sampling. Test accuracy
is computed every 5 epochs and the best value is reported. Result:
`eval-results/finetune_diving48_<run>.json`. Other benchmarks work the same way
(`--dataset jester`, `--dataset all`, ...).

## Diagnostics (Appendix A)

**Single-frame appearance reference.** DINOv3 ViT-B features of the center frame
only:

```bash
python scripts/eval/dinov3_single_frame.py --gpu
python scripts/eval/probe.py dinov3_vitb16_1frame --feature-root features_dino_1f --gpu
```

**IARD identity.** How much actor identity the representation keeps (5 actors,
chance 20%; lower means less appearance):

```bash
python scripts/eval/iard_identity.py --backbone checkpoint --checkpoint <checkpoint> --name my_model
python scripts/eval/iard_identity.py --backbone dinov3_1f
```

## V-JEPA 2 baseline

The V-JEPA 2 row was trained with the official code
([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)) on the same
data and schedule. Evaluation needs a clone of that repository and the run
directory (`latest.pt`, `params-pretrain.yaml`):

```bash
export VJEPA2_REPO=/path/to/vjepa2 VJEPA2_CKPT_DIR=/path/to/run
pip install -e ".[vjepa2]"
TARGET_LENGTH=16 TARGET_FPS=12 python scripts/eval/vjepa2_extract.py --gpu   # -> features_vjepa2/vjepa2_16f_12fps
python scripts/eval/probe.py vjepa2_16f_12fps --feature-root features_vjepa2 --gpu
python scripts/eval/vjepa2_finetune.py --dataset diving48 --gpu --config 16f_12fps
```

V-JEPA 2 has no per-frame class token; its patch tokens are mean-pooled per
temporal position. It uses ImageNet normalisation, while our models use mean = std = 0.5.

## Model size

```bash
python scripts/analysis/params_flops.py
```

prints analytical parameter counts and forward FLOPs of every encoder and decoder
(T=8, 256x256).
