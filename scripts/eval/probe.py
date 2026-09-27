"""Frozen-feature probes: kNN plus five trained probes.

Reads features written by ``scripts/eval/extract_features.py``
(``<feature-root>/<model>/<dataset>/{train,test}.pt``, tokens ``[N, T, M, D]``,
flattened to a sequence of T*M tokens) and reports top-1 accuracy (%) for

  knn          cosine kNN (k=20) on mean-pooled tokens
  linear_mean  mean pool -> linear
  linear_lw    learned softmax-weighted pool -> linear
  mlp_mean     mean pool -> MLP (hidden 512, dropout 0.1)
  mlp_lw       learned weighted pool -> MLP
  attentive    single-query attention pool -> MLP   (the number used in the paper)

Probes: AdamW lr 1e-3, wd 1e-4, constant LR; batch 64 x 100 epochs on the small
datasets, batch 256 x 20 epochs on the large ones.

Usage:
  python scripts/eval/probe.py <model_name> [--feature-root features]
         [--datasets jester,sthsthv2] [--seed 0] [--out eval-results/<model>__seed0.json] [--gpu]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ALL_DATASETS = ["hmdb51", "arid", "iard", "jester", "sthsthv2", "diving48",
                "ek100_verb", "ek100_noun", "ek100_verb_anticip"]

# Per-dataset probe schedule: (batch_size, epochs)
DATASET_CONFIG = {
    "hmdb51":     {"bs": 64,  "epochs": 100},
    "arid":       {"bs": 64,  "epochs": 100},
    "iard":       {"bs": 64,  "epochs": 100},
    "diving48":   {"bs": 64,  "epochs": 100},
    "jester":     {"bs": 256, "epochs": 20},
    "sthsthv2":   {"bs": 256, "epochs": 20},
    "ek100_verb": {"bs": 256, "epochs": 20},
    "ek100_noun": {"bs": 256, "epochs": 20},
    "ek100_verb_anticip": {"bs": 256, "epochs": 20},
}

LR = 1e-3
WEIGHT_DECAY = 1e-4
MLP_HIDDEN = 512
MLP_DROPOUT = 0.1
KNN_K = 20
DEVICE = "cpu"  # set from --gpu in main()


# ============================================================================
# Pooling modules
# ============================================================================
class MeanPool(nn.Module):
    """Simple mean pooling across token dim."""
    def forward(self, x):
        # x: [B, M, D]
        return x.mean(dim=1)  # [B, D]


class LearnableWeightedPool(nn.Module):
    """Learnable weighted sum across tokens. Init: uniform (= mean)."""
    def __init__(self, num_tokens):
        super().__init__()
        # Init to zeros so softmax gives uniform 1/M weights
        self.logits = nn.Parameter(torch.zeros(num_tokens))

    def forward(self, x):
        # x: [B, M, D]
        weights = F.softmax(self.logits, dim=0)  # [M]
        return (x * weights[None, :, None]).sum(dim=1)  # [B, D]


class AttentivePool(nn.Module):
    """Single-query cross-attention pooling."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable query
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Init projections
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, M, D]
        B, M, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(self.query.expand(B, -1, -1))  # [B, 1, D]
        k = self.k_proj(x)  # [B, M, D]
        v = self.v_proj(x)  # [B, M, D]

        q = q.view(B, 1, H, hd).transpose(1, 2)   # [B, H, 1, hd]
        k = k.view(B, M, H, hd).transpose(1, 2)    # [B, H, M, hd]
        v = v.view(B, M, H, hd).transpose(1, 2)    # [B, H, M, hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, 1, M]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, D)  # [B, D]

        return self.out_proj(out)  # [B, D]


# ============================================================================
# Probe models
# ============================================================================
class LinearProbe(nn.Module):
    def __init__(self, dim, num_classes, pool):
        super().__init__()
        self.pool = pool
        self.head = nn.Linear(dim, num_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.01)

    def forward(self, x):
        return self.head(self.pool(x))


class MLPProbe(nn.Module):
    def __init__(self, dim, num_classes, pool, hidden=512, dropout=0.1):
        super().__init__()
        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.pool(x))


class AttentiveProbe(nn.Module):
    def __init__(self, dim, num_classes, num_tokens, hidden=512, dropout=0.1):
        super().__init__()
        self.attn_pool = AttentivePool(dim, num_heads=min(4, dim // 64))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.norm(self.attn_pool(x)))


# ============================================================================
# Training
# ============================================================================
def train_probe(probe, train_feats, train_labels, test_feats, test_labels,
                bs, epochs, lr=LR, wd=WEIGHT_DECAY):
    """Train a probe and return best test accuracy."""
    probe = probe.to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    total_steps = (len(train_feats) // bs + 1) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=bs, shuffle=True, drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_feats, test_labels),
        batch_size=min(512, len(test_feats)), shuffle=False,
    )

    best_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(probe(feats), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval
        probe.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, labels in test_loader:
                preds = probe(feats.to(DEVICE)).argmax(dim=-1)
                correct += (preds == labels.to(DEVICE)).sum().item()
                total += len(labels)
        acc = correct / total * 100
        best_acc = max(best_acc, acc)

    return best_acc


def knn_eval(train_feats, train_labels, test_feats, test_labels, k=20):
    """KNN on mean-pooled features. Handles [N, S, D] where S = T*M or M."""
    # Mean pool all non-batch dims to get [N, D]
    tf = F.normalize(train_feats.mean(1).float(), dim=-1)
    qf = F.normalize(test_feats.mean(1).float(), dim=-1)
    sim = qf @ tf.T
    _, topk_idx = sim.topk(k, dim=-1)
    pred = train_labels[topk_idx].mode(dim=-1).values
    return (pred == test_labels).float().mean().item() * 100


# ============================================================================
# Main
# ============================================================================
def probe_dataset(train_path, test_path, bs, epochs, tag):
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    train_tokens = train_data["tokens"].float()
    test_tokens = test_data["tokens"].float()
    train_labels = train_data["labels"]
    test_labels = test_data["labels"]
    num_classes = int(train_labels.max().item()) + 1

    if train_tokens.dim() == 4:  # [N, T, M, D] -> [N, T*M, D]
        N, T, M, D = train_tokens.shape
        train_tokens = train_tokens.reshape(N, T * M, D)
        test_tokens = test_tokens.reshape(test_tokens.shape[0], T * M, D)
    seq_len, D = train_tokens.shape[1], train_tokens.shape[2]
    print(f"=== {tag} (seq={seq_len}, D={D}, classes={num_classes}, "
          f"train={len(train_labels)}, test={len(test_labels)}) ===")

    res = {"knn": knn_eval(train_tokens, train_labels, test_tokens, test_labels, KNN_K)}
    probes = {
        "linear_mean": lambda: LinearProbe(D, num_classes, MeanPool()),
        "linear_lw": lambda: LinearProbe(D, num_classes, LearnableWeightedPool(seq_len)),
        "mlp_mean": lambda: MLPProbe(D, num_classes, MeanPool(), MLP_HIDDEN, MLP_DROPOUT),
        "mlp_lw": lambda: MLPProbe(D, num_classes, LearnableWeightedPool(seq_len), MLP_HIDDEN, MLP_DROPOUT),
        "attentive": lambda: AttentiveProbe(D, num_classes, seq_len, MLP_HIDDEN, MLP_DROPOUT),
    }
    for name, make in probes.items():
        res[name] = train_probe(make(), train_tokens, train_labels, test_tokens, test_labels, bs, epochs)
    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in res.items()))
    return res


def main():
    global DEVICE
    ap = argparse.ArgumentParser(description="Frozen-feature probes (see module docstring).")
    ap.add_argument("model", help="feature folder name under --feature-root")
    ap.add_argument("--feature-root", default="features")
    ap.add_argument("--datasets", default=",".join(ALL_DATASETS),
                    help="comma list; datasets without features are skipped")
    ap.add_argument("--seed", type=int, default=None, help="probe init / shuffling seed")
    ap.add_argument("--out", default=None, help="result JSON (default eval-results/<model>[__seed<S>].json)")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    DEVICE = "cuda" if a.gpu and torch.cuda.is_available() else "cpu"
    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
    feat_dir = Path(a.feature_root) / a.model
    if not feat_dir.is_dir():
        raise SystemExit(f"no features at {feat_dir}")

    results = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        tr, te = feat_dir / ds / "train.pt", feat_dir / ds / "test.pt"
        if not (tr.exists() and te.exists()):
            continue
        cfg = DATASET_CONFIG[ds]
        results[f"{a.model}_{ds}"] = probe_dataset(tr, te, cfg["bs"], cfg["epochs"], f"{a.model} / {ds}")
    if not results:
        raise SystemExit(f"no train/test features found under {feat_dir} for {a.datasets}")

    suffix = f"__seed{a.seed}" if a.seed is not None else ""
    out = Path(a.out) if a.out else Path("eval-results") / f"{a.model}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
