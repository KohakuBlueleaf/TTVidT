"""IARD identity check (paper Appendix A, Table 5).

Re-labels IARD clips by actor (5 classes, random split with every actor in both
splits) and probes how much actor identity a frozen representation retains.
Lower is more motion-centric; chance is 20%.

    python scripts/eval/iard_identity.py --backbone checkpoint --checkpoint <ckpt or release dir> --name ttvidt
    python scripts/eval/iard_identity.py --backbone dinov3_1f          # appearance reference
    python scripts/eval/iard_identity.py --backbone vjepa2_16f         # needs VJEPA2_REPO / VJEPA2_CKPT_DIR

Reports kNN@20 (the number in the paper) and an attentive probe.
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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

# ---- Patch IARDDataset to use actor label + cache parsed actor ----
from vidmet.datasets import IARDDataset

ACTORS = ["andrea", "georgios", "gu", "leyla", "steve"]
ACTOR_TO_IDX = {a: i for i, a in enumerate(ACTORS)}

_orig_load = IARDDataset._load_annotations

def _patched_load(self):
    video_dir = self.root / "videos"
    self.classes = ACTORS  # 5-class identity
    self.label_to_idx = ACTOR_TO_IDX
    all_samples = []
    for action_dir in ["drink", "eat", "jump", "run", "walk"]:
        cdir = video_dir / action_dir
        if not cdir.exists(): continue
        tar_stems = set(); vids = []
        for v in cdir.glob("*.tar"):
            tar_stems.add(v.stem); vids.append(v)
        for v in cdir.glob("*.avi"):
            if v.stem not in tar_stems: vids.append(v)
        for vp in vids:
            parts = vp.stem.split("_")
            actor = parts[1] if len(parts) > 1 else "unknown"
            if actor not in ACTOR_TO_IDX: continue
            all_samples.append({"path": vp, "label": ACTOR_TO_IDX[actor]})
    import random
    rng = random.Random(self.seed)
    rng.shuffle(all_samples)
    split_idx = int(len(all_samples) * self.train_ratio)
    chosen = all_samples[:split_idx] if self.split == "train" else all_samples[split_idx:]
    self.samples = [(s["path"], s["label"]) for s in chosen]

IARDDataset._load_annotations = _patched_load


DATA_ROOT = os.environ.get("TTVIDT_EVAL_DATA", "eval-dataset")


def get_iard(target_length, target_fps, split, transform, sampling_mode="center"):
    return IARDDataset(
        root=f"{DATA_ROOT}/iard-tar", split=split,
        split_by="random", train_ratio=0.8,
        target_length=target_length, target_fps=target_fps,
        sampling_mode=sampling_mode, backend="tar", transform=transform,
    )


# ---- Backbone-specific extractors ----

def extract_dinov3_1f():
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
    model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to("cuda").eval().requires_grad_(False)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    out = {}
    for split in ["train", "test"]:
        ds = get_iard(1, None, split, transform)
        loader = DataLoader(ds, batch_size=64, num_workers=8, shuffle=False)
        toks, lbls = [], []
        for v, y in tqdm(loader, desc=f"dinov3 {split}"):
            v = v[:, 0].to("cuda")
            t = model(v).last_hidden_state.half().cpu()
            toks.append(t); lbls.append(y)
        out[split] = (torch.cat(toks), torch.cat(lbls))
    return out


def extract_vjepa2_16f():
    import yaml
    sys.path.insert(0, os.environ["VJEPA2_REPO"])  # clone of facebookresearch/vjepa2
    from src.models import vision_transformer as vt
    PARAMS = os.path.join(os.environ["VJEPA2_CKPT_DIR"], "params-pretrain.yaml")
    CKPT = os.path.join(os.environ["VJEPA2_CKPT_DIR"], "latest.pt")
    with open(PARAMS) as f: p = yaml.safe_load(f)
    factory = getattr(vt, p["model"]["model_name"])
    img_size = int(p["data"]["crop_size"])
    patch = int(p["data"]["patch_size"])
    tube = int(p["data"]["tubelet_size"])
    nf = int(p["data"].get("num_frames", p["data"]["dataset_fpcs"][0]))
    model = factory(img_size=img_size, patch_size=patch, num_frames=nf, tubelet_size=tube)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = sd.get("target_encoder") or sd.get("encoder") or sd
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda").eval().requires_grad_(False)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = {}
    Hp = Wp = img_size // 16
    for split in ["train", "test"]:
        ds = get_iard(16, 12.0, split, transform)
        loader = DataLoader(ds, batch_size=2, num_workers=8, shuffle=False)
        toks, lbls = [], []
        for v, y in tqdm(loader, desc=f"vjepa2 {split}"):
            v = v.permute(0, 2, 1, 3, 4).to("cuda")
            B, C, T, H, W = v.shape
            with torch.no_grad():
                f = model(v)
            Tp = T // tube; D = f.shape[-1]
            f = f.reshape(B, Tp, Hp * Wp, D).mean(dim=2)
            toks.append(f.half().cpu()); lbls.append(y)
        out[split] = (torch.cat(toks), torch.cat(lbls))
    return out


def extract_checkpoint(checkpoint):
    """Our encoders (TT-VidT / VideoMAE-3D / DisMo-2D3D): same sampling as the
    frozen-probe features (8 frames, uniform, 256x256, [-1, 1])."""
    from ttvidt.hub import load_model

    model = load_model(checkpoint, device="cuda", with_decoder=False)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    out = {}
    for split in ["train", "test"]:
        ds = get_iard(8, None, split, transform, sampling_mode="uniform")
        loader = DataLoader(ds, batch_size=64, num_workers=8, shuffle=False)
        toks, lbls = [], []
        for v, y in tqdm(loader, desc=f"{split}"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                toks.append(model.encoder(v.to("cuda")).motion_output.float().cpu())
            lbls.append(y)
        out[split] = (torch.cat(toks), torch.cat(lbls))
    return out


# ---- Probe ----

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    def forward(self, x):
        a = torch.matmul(self.q, x.transpose(-1, -2)) / math.sqrt(x.shape[-1])
        a = F.softmax(a, dim=-1)
        return torch.matmul(a, x).squeeze(1)


class AttentiveProbe(nn.Module):
    def __init__(self, dim, num_classes, hidden=512, dropout=0.1):
        super().__init__()
        self.pool = AttentionPool(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.mlp(self.pool(x))


def knn_probe(tr_x, tr_y, te_x, te_y, k=20):
    tr_x = tr_x.float(); te_x = te_x.float()
    if tr_x.dim() == 4:  # [N, T, M, D] -> [N, T*M, D]
        N, T, M, D = tr_x.shape
        tr_x = tr_x.reshape(N, T * M, D)
        te_x = te_x.reshape(te_x.shape[0], T * M, D)
    tr = F.normalize(tr_x.mean(dim=1), dim=-1)
    te = F.normalize(te_x.mean(dim=1), dim=-1)
    sims = te @ tr.t()
    _, idx = sims.topk(k, dim=1)
    preds = tr_y[idx].mode(dim=1).values
    return (preds == te_y).float().mean().item() * 100


def attentive_probe(tr_x, tr_y, te_x, te_y, num_classes, bs=64, epochs=100):
    tr_x = tr_x.float(); te_x = te_x.float()
    if tr_x.dim() == 4:  # [N, T, M, D] flatten T*M
        N, T, M, D = tr_x.shape
        tr_x = tr_x.reshape(N, T * M, D)
        te_x = te_x.reshape(te_x.shape[0], T * M, D)
    D = tr_x.shape[-1]
    model = AttentiveProbe(D, num_classes).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(tr_x, tr_y)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to("cuda"), yb.to("cuda")
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(te_x), bs):
            xb = te_x[i:i+bs].to("cuda")
            yb = te_y[i:i+bs].to("cuda")
            correct += (model(xb).argmax(-1) == yb).sum().item()
            total += len(yb)
    return correct / total * 100


def main():
    ap = argparse.ArgumentParser(description="IARD actor-identity probe (see module docstring).")
    ap.add_argument("--backbone", required=True, choices=["checkpoint", "dinov3_1f", "vjepa2_16f"])
    ap.add_argument("--checkpoint", default=None, help="for --backbone checkpoint: .ckpt / release dir / repo id")
    ap.add_argument("--name", default=None, help="name for outputs (default: the backbone)")
    args = ap.parse_args()
    name = args.name or args.backbone

    if args.backbone == "dinov3_1f":
        data = extract_dinov3_1f()
    elif args.backbone == "vjepa2_16f":
        data = extract_vjepa2_16f()
    else:
        if not args.checkpoint:
            ap.error("--backbone checkpoint needs --checkpoint")
        data = extract_checkpoint(args.checkpoint)

    out_dir = Path("features_iard_identity") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "data.pt")
    tr_x, tr_y = data["train"]
    te_x, te_y = data["test"]
    print(f"\n{name}: train {tuple(tr_x.shape)}, test {tuple(te_x.shape)}")
    knn = knn_probe(tr_x, tr_y, te_x, te_y, k=20)
    attn = attentive_probe(tr_x, tr_y, te_x, te_y, num_classes=5)
    print(f"  identity kNN@20: {knn:.2f}%   attentive: {attn:.2f}%")
    out_path = Path("eval-results") / f"iard_identity_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"backbone": name, "knn": knn, "attentive": attn,
                                    "n_train": len(tr_y), "n_test": len(te_y)}, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
