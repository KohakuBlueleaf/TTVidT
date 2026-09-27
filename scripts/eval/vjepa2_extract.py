"""
Extract V-JEPA2 features on our sweep datasets.
Matches our sweep recipe: 8 frames uniform, 256x256.

V-JEPA2 output: [B, N, D] patch tokens where N = T_patches * H_patches * W_patches
With tubelet=2, 8 frames, 256x256, patch=16: N = 4*16*16 = 1024 tokens, D=1024

For probing: mean-pool spatial patches per temporal frame → [B, T_patches, D]
This mimics a "per frame class token" since V-JEPA2 has no CLS token.

Usage:
    VJEPA2_REPO=<vjepa2 clone> VJEPA2_CKPT_DIR=<run dir> python scripts/eval/vjepa2_extract.py --gpu
"""
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


DEVICE = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
FEATURE_DIR = Path(os.environ.get("FEATURE_BASE", "features_vjepa2"))
DATA_ROOT = "eval-dataset"
NUM_WORKERS = 8
RESIZE = (256, 256)

# Config via env vars
TARGET_LENGTH = int(os.environ.get("TARGET_LENGTH", 8))
TARGET_FPS = float(os.environ.get("TARGET_FPS", 6.0))
CONFIG_NAME = os.environ.get("CONFIG_NAME", f"{TARGET_LENGTH}f_{int(TARGET_FPS)}fps")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4 if TARGET_LENGTH <= 8 else 2))

CHECKPOINT = os.path.join(os.environ.get("VJEPA2_CKPT_DIR", "checkpoints/vjepa2"), "latest.pt")
PARAMS = os.path.join(os.environ.get("VJEPA2_CKPT_DIR", "checkpoints/vjepa2"), "params-pretrain.yaml")

DATASETS = {
    "hmdb51":   {"train_split": "train", "val_split": "test"},
    "arid":     {"train_split": "train", "val_split": "test"},
    "iard":     {"train_split": "train", "val_split": "test"},
    "jester":   {"train_split": "train", "val_split": "validation"},
    "sthsthv2": {"train_split": "train", "val_split": "validation"},
    "diving48": {"train_split": "train", "val_split": "test"},
    "ek100_verb": {"train_split": "train", "val_split": "validation"},
    "ek100_verb_anticip": {"train_split": "train", "val_split": "validation"},
}


def load_vjepa2():
    """Load V-JEPA2 encoder and return (model, img_size, num_frames)."""
    import yaml
    sys.path.insert(0, os.environ["VJEPA2_REPO"])  # clone of facebookresearch/vjepa2
    from src.models import vision_transformer as vt

    with open(PARAMS) as f:
        params = yaml.safe_load(f)

    # model_name is under 'model' (not 'meta')
    model_name = params["model"]["model_name"]
    img_size = int(params["data"]["crop_size"])
    patch_size = int(params["data"]["patch_size"])
    tubelet_size = int(params["data"]["tubelet_size"])
    # num_frames is under dataset_fpcs[0] in this config
    if "num_frames" in params["data"]:
        num_frames_train = int(params["data"]["num_frames"])
    else:
        num_frames_train = int(params["data"]["dataset_fpcs"][0])

    factory = getattr(vt, model_name)
    model = factory(
        img_size=img_size, patch_size=patch_size,
        num_frames=num_frames_train, tubelet_size=tubelet_size,
    )

    print(f"Loading checkpoint {CHECKPOINT}...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    sd = ckpt.get("target_encoder") or ckpt.get("encoder") or ckpt
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"  Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

    model = model.to(DEVICE).eval().requires_grad_(False)
    return model, img_size, tubelet_size


def get_dataset(dataset_name, split):
    """Use our sweep recipe but with ImageNet normalization (V-JEPA2 training norm)."""
    from vidmet.datasets import (
        SSv2Dataset, HMDB51Dataset, JesterDataset, IARDDataset, ARIDDataset, Diving48Dataset,
    )
    from vidmet.ek100_dataset import EK100AnticipationDataset

    info = DATASETS[dataset_name]
    split_name = info["train_split"] if split == "train" else info["val_split"]
    # Special-case ek100 anticipation: separate tar dir
    if dataset_name == "ek100_verb_anticip":
        root = f"{DATA_ROOT}/ek100_anticipation-tar"
    elif dataset_name == "ek100_verb":
        root = f"{DATA_ROOT}/ek100-tar"
    else:
        root = f"{DATA_ROOT}/{dataset_name}-tar"

    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
    ])

    common = dict(root=root, split=split_name, transform=transform,
                  target_length=TARGET_LENGTH, target_fps=TARGET_FPS,
                  sampling_mode="center", backend="tar")

    match dataset_name:
        case "sthsthv2": return SSv2Dataset(**common)
        case "hmdb51": return HMDB51Dataset(**common, split_id=1)
        case "jester": return JesterDataset(**common)
        case "iard": return IARDDataset(**common, split_by="actor", train_ratio=0.8)
        case "arid": return ARIDDataset(**common, split_id=0)
        case "diving48": return Diving48Dataset(**common)
        case "ek100_verb": return EK100AnticipationDataset(**common, label_type="verb")
        case "ek100_verb_anticip": return EK100AnticipationDataset(**common, label_type="verb")


@torch.no_grad()
def extract(model, dataloader, tubelet_size, img_size):
    all_tokens = []
    all_labels = []
    # Input: video [B, T, C, H, W] → V-JEPA2 expects [B, C, T, H, W]
    # Pad T to be divisible by tubelet (for T=8, tubelet=2 → T_patches=4)
    H_patches = img_size // 16
    W_patches = img_size // 16

    for videos, labels in tqdm(dataloader, desc="extract"):
        # videos: [B, T, C, H, W] → permute to [B, C, T, H, W]
        videos = videos.permute(0, 2, 1, 3, 4).to(DEVICE)
        B, C, T, H, W = videos.shape

        # Forward: returns [B, T_p * H_p * W_p, D]
        feats = model(videos)

        # Reshape to [B, T_p, H_p*W_p, D]
        T_p = T // tubelet_size
        D = feats.shape[-1]
        feats = feats.reshape(B, T_p, H_patches * W_patches, D)

        # Mean-pool spatial → [B, T_p, D] (per-frame token equivalent)
        feats = feats.mean(dim=2)

        all_tokens.append(feats.cpu().float())
        all_labels.append(labels)

    return torch.cat(all_tokens), torch.cat(all_labels)


def main():
    model, img_size, tubelet_size = load_vjepa2()
    print(f"img_size={img_size}, tubelet_size={tubelet_size}")

    # Sanity check output shape
    with torch.no_grad():
        test = torch.randn(1, 3, TARGET_LENGTH, img_size, img_size).to(DEVICE)
        out = model(test)
        T_p = TARGET_LENGTH // tubelet_size
        H_p = img_size // 16
        print(f"Output shape: {list(out.shape)} (expected B×{T_p*H_p*H_p}×D)")

    model_name_out = f"vjepa2_{CONFIG_NAME}"
    # ONLY_DATASETS env var: restrict to a comma-separated subset (e.g. re-extracting
    # just iard after the actor-split fix) instead of redoing all 8.
    only = os.environ.get("ONLY_DATASETS", "").strip()
    ds_iter = [d.strip() for d in only.split(",") if d.strip()] if only else list(DATASETS)
    for d in ds_iter:
        if d not in DATASETS:
            raise ValueError(f"unknown dataset {d}; known: {list(DATASETS)}")
    print(f"extracting datasets: {ds_iter}")
    for ds_name in ds_iter:
        print(f"\n--- {ds_name} ---")
        out_dir = FEATURE_DIR / model_name_out / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "train.pt").exists() and (out_dir / "test.pt").exists():
            print("  SKIP (already done)")
            continue

        for split, fname in [("train", "train.pt"), ("test", "test.pt")]:
            dataset = get_dataset(ds_name, split)
            print(f"  {split}: {len(dataset)} samples")
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                shuffle=False, drop_last=False)
            tokens, labels = extract(model, loader, tubelet_size, img_size)
            print(f"  -> tokens {list(tokens.shape)}, labels {list(labels.shape)}")
            torch.save({"tokens": tokens, "labels": labels}, out_dir / fname)

    print(f"\nDone! Features in {FEATURE_DIR}/{model_name_out}/")


if __name__ == "__main__":
    main()
