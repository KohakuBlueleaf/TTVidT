"""Finetune V-JEPA2 ViT-L on D48 / EK-V trimmed.
Same head + hyperparams as scripts/eval/finetune.py, but V-JEPA2 backbone.

Outputs: eval-results/finetune_<dataset>_vjepa2_vitl16_<config>.json

Usage:
    VJEPA2_REPO=<vjepa2 clone> VJEPA2_CKPT_DIR=<run dir> python scripts/eval/vjepa2_finetune.py --dataset diving48 --gpu --config 16f_12fps
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


DATA_ROOT = "eval-dataset"
RESIZE = (256, 256)

DATASET_INFO = {
    "diving48":  {"num_classes": 48,  "train_split": "train", "val_split": "test",       "bs": 2, "epochs": 50},
    "ek100_verb": {"num_classes": 97, "train_split": "train", "val_split": "validation", "bs": 2, "epochs": 20},
}

ENCODER_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
NUM_WORKERS = 8
GRAD_ACC = 16
MLP_HIDDEN = 512
DROPOUT = 0.1

CHECKPOINT = os.path.join(os.environ.get("VJEPA2_CKPT_DIR", "checkpoints/vjepa2"), "latest.pt")
PARAMS = os.path.join(os.environ.get("VJEPA2_CKPT_DIR", "checkpoints/vjepa2"), "params-pretrain.yaml")


class AttentiveHead(nn.Module):
    """Same as scripts/eval/finetune.py."""
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


class VJEPA2Finetune(nn.Module):
    """V-JEPA2 backbone + attentive head (same head as TT-VidT FT)."""
    def __init__(self, backbone, num_classes, img_size, tubelet_size, hidden_size, mlp_hidden=512, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.H_patches = img_size // 16
        self.W_patches = img_size // 16
        self.head = AttentiveHead(hidden_size, num_classes, mlp_hidden, dropout)

    def forward(self, video):
        # video: [B, T, C, H, W] → [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = video.shape
        feats = self.backbone(video)  # [B, T_p*H*W, D]
        T_p = T // self.tubelet_size
        D = feats.shape[-1]
        feats = feats.reshape(B, T_p, self.H_patches * self.W_patches, D)
        feats = feats.mean(dim=2)  # [B, T_p, D] per-frame mean-pool
        return self.head(feats)


def load_vjepa2_backbone():
    import yaml
    sys.path.insert(0, os.environ["VJEPA2_REPO"])  # clone of facebookresearch/vjepa2
    from src.models import vision_transformer as vt
    with open(PARAMS) as f:
        params = yaml.safe_load(f)
    model_name = params["model"]["model_name"]
    img_size = int(params["data"]["crop_size"])
    patch_size = int(params["data"]["patch_size"])
    tubelet_size = int(params["data"]["tubelet_size"])
    if "num_frames" in params["data"]:
        num_frames_train = int(params["data"]["num_frames"])
    else:
        num_frames_train = int(params["data"]["dataset_fpcs"][0])
    factory = getattr(vt, model_name)
    model = factory(img_size=img_size, patch_size=patch_size,
                    num_frames=num_frames_train, tubelet_size=tubelet_size)
    print(f"Loading V-JEPA2 ckpt {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    sd = ckpt.get("target_encoder") or ckpt.get("encoder") or ckpt
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"  Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    hidden_size = model.embed_dim
    return model, img_size, tubelet_size, hidden_size


def get_dataset(dataset_name, split, target_length, augment=False):
    from vidmet.datasets import Diving48Dataset
    from vidmet.ek100_dataset import EK100AnticipationDataset
    info = DATASET_INFO[dataset_name]
    split_name = info["train_split"] if split == "train" else info["val_split"]
    if augment:
        transform = transforms.Compose([
            transforms.Resize(RESIZE), transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    common = dict(split=split_name, transform=transform,
                  target_length=target_length, sampling_mode="uniform" if augment else "center",
                  backend="tar")
    if dataset_name == "diving48":
        return Diving48Dataset(root=f"{DATA_ROOT}/diving48-tar", **common)
    elif dataset_name == "ek100_verb":
        return EK100AnticipationDataset(root=f"{DATA_ROOT}/ek100-tar", label_type="verb", **common)
    raise ValueError(dataset_name)


def cosine_lr(opt, ep, total, warmup):
    for pg in opt.param_groups:
        base = pg["base_lr"]
        if ep < warmup:
            lr = base * (ep + 1) / warmup
        else:
            t = (ep - warmup) / (total - warmup)
            lr = base * 0.5 * (1 + math.cos(math.pi * t))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["diving48", "ek100_verb"])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--config", default="16f_12fps")
    args = ap.parse_args()
    target_length = 16 if "16f" in args.config else 8
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    info = DATASET_INFO[args.dataset]

    backbone, img_size, tubelet_size, hidden_size = load_vjepa2_backbone()
    model = VJEPA2Finetune(backbone, info["num_classes"], img_size, tubelet_size, hidden_size,
                           MLP_HIDDEN, DROPOUT).to(device)
    print(f"V-JEPA2 finetune on {args.dataset} (target_length={target_length})")
    print(f"Arch: ViT-L 256px tubelet={tubelet_size} hidden={hidden_size}")

    train_ds = get_dataset(args.dataset, "train", target_length, augment=True)
    test_ds = get_dataset(args.dataset, "test", target_length, augment=False)
    print(f"Train {len(train_ds)}, Test {len(test_ds)}")
    bs = info["bs"]; epochs = info["epochs"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False, num_workers=NUM_WORKERS)

    opt = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": ENCODER_LR, "base_lr": ENCODER_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR, "base_lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    out_path = Path("eval-results") / f"finetune_{args.dataset}_vjepa2_vitl16_{args.config}.json"
    best_acc = 0
    for ep in range(epochs):
        cosine_lr(opt, ep, epochs, WARMUP_EPOCHS)
        model.train()
        total_loss = 0
        opt.zero_grad()
        for step, (video, labels) in enumerate(tqdm(train_loader, desc=f"Ep{ep+1}", leave=False)):
            video, labels = video.to(device), labels.to(device)
            logits = model(video)
            loss = F.cross_entropy(logits, labels) / GRAD_ACC
            loss.backward()
            if (step + 1) % GRAD_ACC == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            total_loss += loss.item() * GRAD_ACC
        avg_loss = total_loss / len(train_loader)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            acc = evaluate(model, test_loader, device)
            best_acc = max(best_acc, acc)
            print(f"  Ep{ep+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.2f}% best={best_acc:.2f}%")
            with open(out_path, "w") as f:
                json.dump({"model": f"vjepa2_vitl16_{args.config}", "dataset": args.dataset,
                           "probe": "finetune_attentive", "accuracy": best_acc,
                           "config": {"epochs": epochs, "last_epoch": ep + 1,
                                      "enc_lr": ENCODER_LR, "head_lr": HEAD_LR}}, f, indent=2)
        else:
            print(f"  Ep{ep+1}/{epochs}: loss={avg_loss:.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
