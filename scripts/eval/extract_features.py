#!/usr/bin/env python3
"""Extract frozen motion-token features for the benchmark datasets.

The model is rebuilt from the checkpoint itself (training ``.ckpt`` or an exported
model directory / Hugging Face repo), so only its path is needed::

    ttvidt-run scripts/eval/extract_features.py -c configs/eval/frozen.py \
        CHECKPOINT_PATH=ttvidt/<run_id>/checkpoints/epoch=7.ckpt

Output, per dataset and split::

    {FEATURE_DIR}/{model_name}/{dataset}/{train,test}.pt
        tokens:   [N, T, M, D] motion tokens of every frame (no pooling)
        labels:   [N]
        metadata: dict

``model_name`` is ``<arch>_<run_id>_<ckpt stem>`` for training checkpoints and the
directory name for exported models (``MODEL_NAME`` overrides it). Probe the
features with ``scripts/eval/probe.py``.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ttvidt.hub import load_model as hub_load_model

# ============================================================================
# Configuration (overridden by the config / command line)
# ============================================================================

CHECKPOINT_PATH: str | None = None   # .ckpt, exported model dir, or HF repo id
MODEL_NAME: str | None = None        # output folder name; derived from the path if None
USE_EMA: bool = False                # training checkpoints: use the EMA weights
AMP_DTYPE: str = "fp16"              # autocast dtype ("fp16" / "bf16"); use bf16 for
                                     # bf16-trained checkpoints (fp16 can overflow)

# Datasets — comma-separated string or list
DATASETS: str = "hmdb51,arid,iard,jester,sthsthv2"
DATA_ROOT: str = "eval-dataset"
DATASET_FORMAT: str = "tar"  # "tar" for tar-JPEG, "video" for raw video
FEATURE_SUFFIX: str = ""  # appended to feature dir name (e.g. "_6fps", "_16f")

# Video sampling
TARGET_LENGTH: int = 8
TARGET_FPS: float | None = None
TARGET_DURATION: float | None = None
SAMPLING_MODE: str = "uniform"
RESIZE: tuple[int, int] = (256, 256)

# Runtime
BATCH_SIZE: int = 64
NUM_WORKERS: int = 16
DEVICE: str = "cuda"

# Output
FEATURE_DIR: str = "features"

# Dataset-specific
HMDB_SPLIT_ID: int = 1
IARD_SPLIT_BY: str = "actor"
IARD_TRAIN_RATIO: float = 0.8
ARID_SPLIT_ID: int = 0

SANITY_CHECK: bool = False
SANITY_BATCHES: int = 10

# ============================================================================
# Dataset Info
# ============================================================================

DATASET_INFO = {
    "sthsthv2": {"num_classes": 174, "train_split": "train", "val_split": "validation"},
    "hmdb51": {"num_classes": 51, "train_split": "train", "val_split": "test"},
    "jester": {"num_classes": 27, "train_split": "train", "val_split": "validation"},
    "iard": {"num_classes": 5, "train_split": "train", "val_split": "test"},
    "arid": {"num_classes": 11, "train_split": "train", "val_split": "test"},
    "diving48": {"num_classes": 48, "train_split": "train", "val_split": "test"},
    "ek100_verb": {"num_classes": 97, "train_split": "train", "val_split": "validation"},
    "ek100_noun": {"num_classes": 300, "train_split": "train", "val_split": "validation"},
    "ek100_verb_anticip": {"num_classes": 97, "train_split": "train", "val_split": "validation"},
}

EK100_LABEL_MODE: str = "verb"  # "verb" or "noun", used when DATASETS contains "ek100"


# ============================================================================
# Core
# ============================================================================


def load_model():
    if not CHECKPOINT_PATH:
        raise ValueError("set CHECKPOINT_PATH")
    print(f"Loading {CHECKPOINT_PATH}" + (" (EMA weights)" if USE_EMA else ""))
    return hub_load_model(CHECKPOINT_PATH, device=DEVICE, with_decoder=False, use_ema=USE_EMA)


def get_dataset(dataset_name: str, split: str):
    """Load dataset for evaluation."""
    from vidmet.datasets import (
        SSv2Dataset,
        HMDB51Dataset,
        JesterDataset,
        IARDDataset,
        ARIDDataset,
        Diving48Dataset,
    )
    from vidmet.ek100_dataset import EK100AnticipationDataset
    from torchvision import transforms

    info = DATASET_INFO[dataset_name]
    split_name = info["train_split"] if split == "train" else info["val_split"]
    root = f"{DATA_ROOT}/{dataset_name}-tar" if DATASET_FORMAT == "tar" else f"{DATA_ROOT}/{dataset_name}"

    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    common_kwargs = {
        "root": root,
        "split": split_name,
        "transform": transform,
        "target_length": TARGET_LENGTH,
        "target_fps": TARGET_FPS,
        "target_duration": TARGET_DURATION,
        "sampling_mode": SAMPLING_MODE,
        "backend": DATASET_FORMAT,
    }

    match dataset_name:
        case "sthsthv2":
            return SSv2Dataset(**common_kwargs)
        case "hmdb51":
            return HMDB51Dataset(**common_kwargs, split_id=HMDB_SPLIT_ID)
        case "jester":
            return JesterDataset(**common_kwargs)
        case "iard":
            return IARDDataset(
                **common_kwargs,
                split_by=IARD_SPLIT_BY,
                train_ratio=IARD_TRAIN_RATIO,
            )
        case "arid":
            return ARIDDataset(**common_kwargs, split_id=ARID_SPLIT_ID)
        case "diving48":
            return Diving48Dataset(**common_kwargs)
        case "ek100_verb":
            ek_root = f"{DATA_ROOT}/ek100-tar" if DATASET_FORMAT == "tar" else f"{DATA_ROOT}/ek100"
            return EK100AnticipationDataset(**{**common_kwargs, "root": ek_root}, label_type="verb")
        case "ek100_noun":
            ek_root = f"{DATA_ROOT}/ek100-tar" if DATASET_FORMAT == "tar" else f"{DATA_ROOT}/ek100"
            return EK100AnticipationDataset(**{**common_kwargs, "root": ek_root}, label_type="noun")
        case "ek100_verb_anticip":
            # Real verb anticipation: untrimmed observation windows (4s ending 1s before action)
            ek_root = f"{DATA_ROOT}/ek100_anticipation-tar"
            return EK100AnticipationDataset(**{**common_kwargs, "root": ek_root}, label_type="verb")
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")


@torch.no_grad()
def extract_features(model, dataloader, max_batches=None):
    """Extract motion token features.

    Returns:
        tokens: [N, T, M, D] - motion tokens (all temporal frames)
        labels: [N] - class labels
    """
    all_tokens = []
    all_labels = []

    for i, (video, labels) in enumerate(tqdm(dataloader, desc="Extracting")):
        if max_batches is not None and i >= max_batches:
            break

        video = video.to(DEVICE)

        amp = torch.bfloat16 if AMP_DTYPE in ("bf16", "bfloat16") else torch.float16
        with torch.autocast(torch.device(DEVICE).type, dtype=amp, enabled=DEVICE != "cpu"):
            result = model.encoder(video)  # pixel input
        all_tokens.append(result.motion_output.float().cpu())  # [B, T, M, D]
        all_labels.append(labels)

    return torch.cat(all_tokens), torch.cat(all_labels)


def get_model_name(arch):
    """``<arch>_<run_id>_<ckpt stem>`` for ttvidt/<run_id>/checkpoints/*.ckpt,
    otherwise the checkpoint's file or directory name."""
    if MODEL_NAME:
        return MODEL_NAME
    p = Path(CHECKPOINT_PATH)
    if p.suffix == ".ckpt":
        parts = p.parts
        if len(parts) >= 3 and parts[-2] == "checkpoints":
            return f"{arch}_{parts[-3]}_{p.stem}{FEATURE_SUFFIX}"
        return f"{arch}_{p.stem}{FEATURE_SUFFIX}"
    return f"{p.name.replace('/', '_')}{FEATURE_SUFFIX}"


# ============================================================================
# Main
# ============================================================================


def main():
    # Parse datasets
    if isinstance(DATASETS, str):
        dataset_list = [d.strip() for d in DATASETS.split(",") if d.strip()]
    else:
        dataset_list = list(DATASETS)

    print(f"\n{'='*60}")
    print(f"Feature Extraction")
    print(f"{'='*60}")
    print(f"Checkpoint: {CHECKPOINT_PATH or 'None'}")
    print(f"Datasets: {dataset_list}")
    print(f"Target frames: {TARGET_LENGTH}")
    print(f"Device: {DEVICE}")
    if SANITY_CHECK:
        print(f"*** SANITY CHECK MODE ({SANITY_BATCHES} batches) ***")
    print(f"{'='*60}\n")

    # Load model once
    model = load_model()
    arch = model.backbone_arch
    model_name = get_model_name(arch)
    print(f"Arch: {arch} -> {FEATURE_DIR}/{model_name}")
    max_batches = SANITY_BATCHES if SANITY_CHECK else None

    # Extract features for each dataset
    for dataset_name in dataset_list:
        if dataset_name not in DATASET_INFO:
            print(f"SKIP: unknown dataset '{dataset_name}'")
            continue

        output_dir = Path(FEATURE_DIR) / model_name / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        num_classes = DATASET_INFO[dataset_name]["num_classes"]
        print(f"\n--- {dataset_name} ({num_classes} classes) ---")

        metadata = {
            "backbone_arch": arch,
            "checkpoint": CHECKPOINT_PATH,
            "dataset": dataset_name,
            "num_classes": num_classes,
            "target_length": TARGET_LENGTH,
        }

        for split in ["train", "test"]:
            print(f"  Loading {split} set...")
            try:
                dataset = get_dataset(dataset_name, split)
            except Exception as e:
                print(f"  ERROR loading {dataset_name}/{split}: {e}")
                continue
            print(f"  Samples: {len(dataset)}")

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
                prefetch_factor=2 if NUM_WORKERS > 0 else None,
            )

            tokens, labels = extract_features(model, loader, max_batches)
            print(f"  Tokens: {tokens.shape}, Labels: {labels.shape}")

            save_data = {
                "tokens": tokens,
                "labels": labels,
                "metadata": {
                    **metadata,
                    "split": split,
                    "hidden_dim": tokens.shape[-1],
                    "num_frames": tokens.shape[1],
                    "num_samples": tokens.shape[0],
                },
            }

            output_path = output_dir / f"{split}.pt"
            torch.save(save_data, output_path)
            print(f"  Saved: {output_path}")

    print(f"\nDone! All features saved to: {FEATURE_DIR}/{model_name}/")


if __name__ == "__main__":
    main()
