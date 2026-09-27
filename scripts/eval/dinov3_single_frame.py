"""
Extract single-frame DINOv3 ViT-B features for appearance baseline.
Takes center frame, extracts ALL tokens (CLS + registers + spatial patches).
Uses the same DINOv3 backbone as our pretrained models.

Output: [N, 261, 768] = 1 CLS + 4 registers + 256 spatial patches (16x16 grid at 256px)

Usage:
    python scripts/eval/dinov3_single_frame.py --gpu   (appearance reference, Appendix A)
"""
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


DEVICE = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
FEATURE_DIR = Path(os.environ.get("FEATURE_BASE", "features_dino_1f"))
DATA_ROOT = "eval-dataset"
BATCH_SIZE = 64
NUM_WORKERS = 4
RESIZE = (256, 256)
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

DATASETS = {
    "hmdb51":    {"num_classes": 51,  "train_split": "train", "val_split": "test"},
    "arid":      {"num_classes": 11,  "train_split": "train", "val_split": "test"},
    "iard":      {"num_classes": 5,   "train_split": "train", "val_split": "test"},
    "jester":    {"num_classes": 27,  "train_split": "train", "val_split": "validation"},
    "sthsthv2":  {"num_classes": 174, "train_split": "train", "val_split": "validation"},
    "diving48":  {"num_classes": 48,  "train_split": "train", "val_split": "test"},
    "ek100_verb": {"num_classes": 97, "train_split": "train", "val_split": "validation"},
    "ek100_noun": {"num_classes": 300, "train_split": "train", "val_split": "validation"},
    "ek100_verb_anticip": {"num_classes": 97, "train_split": "train", "val_split": "validation"},
}


def load_dino():
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
    print(f"Loading {MODEL_NAME}...")
    model = DINOv3ViTModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE).eval().requires_grad_(False)
    print(f"Loaded. hidden={model.config.hidden_size}, patch={model.config.patch_size}, regs={model.config.num_register_tokens}")
    return model


def get_dataset(dataset_name, split):
    from vidmet.datasets import (
        SSv2Dataset, HMDB51Dataset, JesterDataset, IARDDataset, ARIDDataset, Diving48Dataset,
    )
    from vidmet.ek100_dataset import EK100AnticipationDataset

    info = DATASETS[dataset_name]
    split_name = info["train_split"] if split == "train" else info["val_split"]
    root = f"{DATA_ROOT}/{dataset_name}-tar"

    # Single frame, normalize to [-1, 1] to match our training pipeline
    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    common = dict(root=root, split=split_name, transform=transform,
                  target_length=1, sampling_mode="center", backend="tar")

    match dataset_name:
        case "sthsthv2": return SSv2Dataset(**common)
        case "hmdb51": return HMDB51Dataset(**common, split_id=1)
        case "jester": return JesterDataset(**common)
        case "iard": return IARDDataset(**common, split_by="actor", train_ratio=0.8)
        case "arid": return ARIDDataset(**common, split_id=0)
        case "diving48": return Diving48Dataset(**common)
        case "ek100_verb":
            ek_root = f"{DATA_ROOT}/ek100-tar"
            return EK100AnticipationDataset(**{**common, "root": ek_root}, label_type="verb")
        case "ek100_noun":
            ek_root = f"{DATA_ROOT}/ek100-tar"
            return EK100AnticipationDataset(**{**common, "root": ek_root}, label_type="noun")
        case "ek100_verb_anticip":
            ek_root = f"{DATA_ROOT}/ek100_anticipation-tar"
            return EK100AnticipationDataset(**{**common, "root": ek_root}, label_type="verb")


@torch.no_grad()
def extract(model, dataloader):
    all_tokens = []
    all_labels = []
    for batch in tqdm(dataloader, desc="  extracting"):
        videos, labels = batch
        # videos: [B, 1, C, H, W] -> [B, C, H, W]
        frames = videos[:, 0].to(DEVICE)

        # DINOv3 forward: returns last_hidden_state [B, 1+regs+patches, D]
        out = model(frames)
        tokens = out.last_hidden_state  # [B, 261, 768]

        # Save in fp16 to halve memory; probe head can cast back if needed.
        all_tokens.append(tokens.half().cpu())
        all_labels.append(labels)

    return torch.cat(all_tokens), torch.cat(all_labels)


def main():
    model = load_dino()

    # Quick shape check
    with torch.no_grad():
        test = torch.randn(1, 3, 256, 256).to(DEVICE)
        out = model(test)
        print(f"Output shape: {list(out.last_hidden_state.shape)} (1 CLS + {model.config.num_register_tokens} regs + {(256//model.config.patch_size)**2} patches)")
    print()

    for ds_name in DATASETS:
        print(f"\n--- {ds_name} ---")
        out_dir = FEATURE_DIR / "dinov3_vitb16_1frame" / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "train.pt").exists() and (out_dir / "test.pt").exists():
            print(f"  Already extracted, skipping")
            continue

        for split, fname in [("train", "train.pt"), ("test", "test.pt")]:
            try:
                dataset = get_dataset(ds_name, split)
            except Exception as e:
                print(f"  SKIP {split}: {e}")
                continue

            print(f"  {split}: {len(dataset)} samples")
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                shuffle=False, drop_last=False)

            tokens, labels = extract(model, loader)
            print(f"  -> tokens: {list(tokens.shape)}, labels: {list(labels.shape)}")

            torch.save({"tokens": tokens, "labels": labels}, out_dir / fname)

    print("\nDone! Features saved to:", FEATURE_DIR / "dinov3_vitb16_1frame")


if __name__ == "__main__":
    main()
