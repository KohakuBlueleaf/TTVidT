"""End-to-end fine-tuning: pretrained encoder + attentive classification head.

The paper fine-tunes on Diving48 (Table 4, "D48 FT"); the other benchmarks are
supported with the same recipe. The encoder is rebuilt from the checkpoint itself.

    python scripts/eval/finetune.py --checkpoint <ckpt or release dir> --dataset diving48 --gpu

Recipe: AdamW, encoder LR 1e-5 / head LR 1e-3, weight decay 0.05, 5 warmup epochs
then cosine, batch 2 x 16 accumulation (effective 32), grad clip 1.0, 50 epochs on
Diving48. Train clips: uniform sampling + random horizontal flip; test clips:
center sampling. Test accuracy is evaluated every 5 epochs and the best value is
reported (as in the paper).
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ttvidt.hub import load_model

# ============================================================================
# Config
# ============================================================================
DATA_ROOT = "eval-dataset"
RESIZE = (256, 256)
TARGET_LENGTH = 8
SAMPLING_MODE = "uniform"

DATASET_INFO = {
    "hmdb51":    {"num_classes": 51,  "train_split": "train", "val_split": "test",       "bs": 2, "epochs": 50},
    "arid":      {"num_classes": 11,  "train_split": "train", "val_split": "test",       "bs": 2, "epochs": 50},
    "iard":      {"num_classes": 5,   "train_split": "train", "val_split": "test",       "bs": 2, "epochs": 50},
    "jester":    {"num_classes": 27,  "train_split": "train", "val_split": "validation", "bs": 2, "epochs": 30},
    "sthsthv2":  {"num_classes": 174, "train_split": "train", "val_split": "validation", "bs": 2, "epochs": 20},
    "diving48":  {"num_classes": 48,  "train_split": "train", "val_split": "test",       "bs": 2, "epochs": 50},
    "ek100_verb": {"num_classes": 97,  "train_split": "train", "val_split": "validation", "bs": 2, "epochs": 20},
    "ek100_noun": {"num_classes": 300, "train_split": "train", "val_split": "validation", "bs": 2, "epochs": 20},
}

# Training
ENCODER_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
NUM_WORKERS = 16
GRAD_ACC = 16  # effective batch = bs(2) * 16 = 32
MLP_HIDDEN = 512
DROPOUT = 0.1


class AttentiveHead(nn.Module):
    def __init__(self, dim, num_classes, hidden=512, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        attn = torch.matmul(self.query, x.transpose(-1, -2)) / math.sqrt(x.shape[-1])
        attn = F.softmax(attn, dim=-1)
        pooled = torch.matmul(attn, x).squeeze(1)
        return self.mlp(pooled)


class FinetuneModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, video):
        result = self.encoder(video)
        motion = result.motion_output
        B, T, M, D = motion.shape
        tokens = motion.reshape(B, T * M, D).float()
        return self.head(tokens)


def load_encoder(checkpoint):
    """Pretrained encoder, architecture taken from the checkpoint / release config."""
    return load_model(checkpoint, device="cpu", with_decoder=False).encoder


def get_dataset(dataset_name, split, augment=False):
    from vidmet.datasets import (
        SSv2Dataset, HMDB51Dataset, JesterDataset, IARDDataset, ARIDDataset, Diving48Dataset,
    )
    from vidmet.ek100_dataset import EK100AnticipationDataset

    info = DATASET_INFO[dataset_name]
    split_name = info["train_split"] if split == "train" else info["val_split"]
    root = f"{DATA_ROOT}/{dataset_name}-tar"

    if augment:
        transform = transforms.Compose([
            transforms.Resize(RESIZE), transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    common = dict(root=root, split=split_name, transform=transform,
                  target_length=TARGET_LENGTH, sampling_mode="uniform" if augment else "center",
                  backend="tar")

    match dataset_name:
        case "sthsthv2": return SSv2Dataset(**common)
        case "hmdb51": return HMDB51Dataset(**common, split_id=1)
        case "jester": return JesterDataset(**common)
        case "iard": return IARDDataset(**common, split_by="actor", train_ratio=0.8)
        case "arid": return ARIDDataset(**common, split_id=0)
        case "diving48": return Diving48Dataset(**common)
        case "ek100_verb":
            return EK100AnticipationDataset(**{**common, "root": f"{DATA_ROOT}/ek100-tar"}, label_type="verb")
        case "ek100_noun":
            return EK100AnticipationDataset(**{**common, "root": f"{DATA_ROOT}/ek100-tar"}, label_type="noun")


def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs):
    for pg in optimizer.param_groups:
        base_lr = pg["base_lr"]
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        pg["lr"] = lr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for video, labels in loader:
        video, labels = video.to(device), labels.to(device)
        logits = model(video)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)
    return correct / total * 100


def run_finetune(dataset_name, device, run_id, checkpoint, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    info = DATASET_INFO[dataset_name]
    bs = info["bs"]
    epochs = info["epochs"]

    train_ds = get_dataset(dataset_name, "train", augment=True)
    test_ds = get_dataset(dataset_name, "test", augment=False)

    # Use config num_classes as upper bound (hardcoded per dataset in DATASET_INFO)
    num_classes = info["num_classes"]
    print(f"\n{'='*60}")
    print(f"Finetuning: {dataset_name} ({num_classes} classes, {epochs} epochs, BS{bs}x{GRAD_ACC})")
    print(f"{'='*60}")

    # Reload encoder fresh for each dataset (avoids deepcopy issues on CUDA)
    enc = load_encoder(checkpoint).requires_grad_(True).train()
    head = AttentiveHead(enc.config.hidden_size, num_classes, MLP_HIDDEN, DROPOUT)
    model = FinetuneModel(enc, head).to(device)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             num_workers=NUM_WORKERS)

    optimizer = torch.optim.AdamW([
        {"params": enc.parameters(), "lr": ENCODER_LR, "base_lr": ENCODER_LR},
        {"params": head.parameters(), "lr": HEAD_LR, "base_lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    best_acc = 0

    for epoch in range(epochs):
        cosine_lr(optimizer, epoch, epochs, WARMUP_EPOCHS)
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, (video, labels) in enumerate(tqdm(train_loader, desc=f"Ep{epoch+1}", leave=False)):
            video, labels = video.to(device), labels.to(device)
            logits = model(video)
            loss = F.cross_entropy(logits, labels) / GRAD_ACC
            loss.backward()
            if (step + 1) % GRAD_ACC == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * GRAD_ACC

        avg_loss = total_loss / len(train_loader)

        out_path = Path(out_dir) / f"finetune_{dataset_name}_{run_id}.json"
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc = evaluate(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
            print(f"  Ep{epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.2f}%, best={best_acc:.2f}%")
            # Save after every eval so timeouts don't lose progress
            result = {
                "model": run_id, "dataset": dataset_name,
                "probe": "finetune_attentive", "accuracy": best_acc,
                "config": {"epochs": epochs, "enc_lr": ENCODER_LR, "head_lr": HEAD_LR, "batch_size": bs * GRAD_ACC,
                           "last_epoch": epoch + 1},
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        else:
            print(f"  Ep{epoch+1}/{epochs}: loss={avg_loss:.4f}")

    out_path = Path(out_dir) / f"finetune_{dataset_name}_{run_id}.json"
    print(f"  Best: {best_acc:.2f}% -> {out_path}")

    # Free memory
    del model, enc, head, optimizer
    torch.cuda.empty_cache()

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="End-to-end fine-tuning (see module docstring).")
    parser.add_argument("--checkpoint", required=True, help=".ckpt, release dir, or HF repo id")
    parser.add_argument("--dataset", required=True, help="dataset name, comma list, or 'all'")
    parser.add_argument("--name", default=None, help="run name in the output file (default: from the path)")
    parser.add_argument("--out-dir", default="eval-results")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    ck = Path(args.checkpoint)
    run_id = args.name or (ck.parts[-3] if ck.suffix == ".ckpt" and len(ck.parts) >= 3 else ck.name)
    if args.dataset == "all":
        datasets = list(DATASET_INFO)
    else:
        datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    print(f"checkpoint {args.checkpoint} | run {run_id} | datasets {datasets} | device {device}")

    results = {}
    for ds in datasets:
        if ds not in DATASET_INFO:
            raise SystemExit(f"unknown dataset {ds}; known: {list(DATASET_INFO)}")
        out_path = Path(args.out_dir) / f"finetune_{ds}_{run_id}.json"
        if out_path.exists():
            prev = json.loads(out_path.read_text())
            if prev.get("config", {}).get("last_epoch") == DATASET_INFO[ds]["epochs"]:
                print(f"skip {ds}: already finished ({prev['accuracy']:.2f}%)")
                results[ds] = prev["accuracy"]
                continue
        results[ds] = run_finetune(ds, device, run_id, args.checkpoint, args.out_dir)

    print(f"\nsummary {run_id}")
    for ds, acc in results.items():
        print(f"  {ds:12s} {acc:.2f}%")


if __name__ == "__main__":
    main()
