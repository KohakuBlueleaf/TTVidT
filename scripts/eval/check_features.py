#!/usr/bin/env python3
"""Sanity-check one model's extracted features before probing them.

Checks per dataset: both splits present and loadable, tokens finite and
non-degenerate, consistent [N, T, M, D] shapes, labels in range, and sample
counts equal to the full benchmark. The last check matters most: the dataset
classes skip clips that are missing on disk without any warning, so an
incomplete dataset copy silently shrinks the probe's training set and lowers
every model's accuracy by a few points.

Usage:
  python scripts/eval/check_features.py <model_name> [--feature-root features] [--datasets a,b,c]
Exit 0 = all good, 1 = something needs a look.
"""
import argparse
import sys
from pathlib import Path

import torch

# (train, test) clip counts of the complete benchmarks, as used in the paper
EXPECTED = {
    "hmdb51": (3518, 1511),      # split 1; 52 + 19 undecodable clips are dropped
    "arid": (3350, 2011),
    "iard": (3640, 910),         # actor split, IARD_SPLIT_BY="actor"
    "jester": (118562, 14787),
    "sthsthv2": (168913, 24777),
    "diving48": (15027, 1970),
}
TOLERANCE = 0.005  # fraction of clips that may be missing (decode failures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--datasets", default="hmdb51,arid,iard,jester,sthsthv2")
    ap.add_argument("--feature-root", default="features")
    a = ap.parse_args()
    base = Path(a.feature_root) / a.model
    if not base.is_dir():
        print(f"FAIL: {base} does not exist")
        return 1
    print(f"=== checking {base} ===")
    ok = True
    dim = None
    for ds in a.datasets.split(","):
        ds = ds.strip()
        exp = EXPECTED.get(ds)
        for split, exp_n in zip(("train", "test"), exp or (None, None)):
            p = base / ds / f"{split}.pt"
            if not p.exists():
                print(f"  {ds}/{split}: MISSING {p}")
                ok = False
                continue
            try:
                d = torch.load(p, map_location="cpu", weights_only=False)
                t, y = d["tokens"], d["labels"]
            except Exception as e:  # noqa: BLE001
                print(f"  {ds}/{split}: UNREADABLE ({e})")
                ok = False
                continue
            n = t.shape[0]
            msgs = []
            if t.dim() != 4:
                msgs.append(f"tokens dim {t.dim()} != 4")
            if dim is None:
                dim = t.shape[-1]
            elif t.shape[-1] != dim:
                msgs.append(f"feature dim {t.shape[-1]} != {dim}")
            tf = t.float()
            if not torch.isfinite(tf).all():
                msgs.append("NON-FINITE tokens")
            elif tf.std().item() < 1e-4:
                msgs.append(f"degenerate tokens (std {tf.std().item():.2e})")
            if y.shape[0] != n:
                msgs.append(f"labels {y.shape[0]} != tokens {n}")
            if y.min().item() < 0:
                msgs.append("negative label")
            if exp_n is not None:
                if n < exp_n * (1 - TOLERANCE):
                    msgs.append(f"only {n}/{exp_n} clips ({100 * n / exp_n:.1f}%) -- dataset copy incomplete?")
                elif n > exp_n:
                    msgs.append(f"{n} clips > expected {exp_n} -- duplicated or wrong split?")
            state = "ok " if not msgs else "BAD"
            ok &= not msgs
            print(f"  {state} {ds}/{split}: n={n}" + (f"/{exp_n}" if exp_n else "")
                  + f" shape={tuple(t.shape)} classes={int(y.max()) + 1}"
                  + ("" if not msgs else "  <- " + "; ".join(msgs)))
    print("=== PASS ===" if ok else "=== FAIL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
