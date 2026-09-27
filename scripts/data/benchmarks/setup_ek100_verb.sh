#!/bin/bash
# Setup EK100 using pre-trimmed clips from HuggingFace (~24 GB).
# Matches H100 cluster setup: ek100-tar/{train,val}/*.tar
#
# Usage:
#   bash scripts/data/benchmarks/setup_ek100_verb.sh [output_dir]

set -euo pipefail

DST_DIR="${1:-./eval-dataset/ek100}"
TAR_DIR="./eval-dataset/ek100-tar"

echo "=== EK100 Setup (Pre-trimmed Clips) ==="
echo "Output: $DST_DIR"
echo "Tar output: $TAR_DIR"
echo ""

mkdir -p "$DST_DIR" "$TAR_DIR/train" "$TAR_DIR/val"

# --- Step 1: Clone annotations ---
echo "Step 1: Getting annotations..."
ANN_DIR="$DST_DIR/annotations"
if [ ! -f "$ANN_DIR/EPIC_100_train.csv" ]; then
    git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-100-annotations.git "$ANN_DIR"
    echo "Annotations cloned"
else
    echo "Annotations already present"
fi

# Symlink annotations into tar dir
ln -sfn "$(realpath "$ANN_DIR")" "$TAR_DIR/annotations" 2>/dev/null || true

# --- Step 2: Download pre-trimmed clips ---
echo ""
echo "Step 2: Downloading pre-trimmed clips from HuggingFace (~24 GB)..."
python3 << 'PYEOF'
import os, sys

dst = os.environ.get("DST_DIR", "./eval-dataset/ek100")
video_dir = os.path.join(dst, "trimmed_videos")
os.makedirs(video_dir, exist_ok=True)

from huggingface_hub import snapshot_download
snapshot_download(
    "kiyoonkim/EPIC-KITCHENS-100-trimmed",
    repo_type="dataset",
    local_dir=video_dir,
    max_workers=8,
)
print(f"Downloaded to: {video_dir}")
PYEOF

export DST_DIR TAR_DIR

# --- Step 3: Convert to tar-JPEG by split ---
echo ""
echo "Step 3: Converting to tar-JPEG..."
python3 << 'PYEOF'
import csv
import io
import json
import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import cv2

dst_dir = os.environ.get("DST_DIR", "./eval-dataset/ek100")
tar_dir = os.environ.get("TAR_DIR", "./eval-dataset/ek100-tar")
ann_dir = os.path.join(dst_dir, "annotations")
video_base = Path(dst_dir) / "trimmed_videos"

# Read annotations to get split membership + labels
splits = {}  # narration_id -> {"split", "verb_class", "noun_class", "video_id", "participant_id"}
for split_name, csv_file in [("train", "EPIC_100_train.csv"), ("val", "EPIC_100_validation.csv")]:
    csv_path = os.path.join(ann_dir, csv_file)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = row["narration_id"]
            splits[nid] = {
                "split": split_name,
                "verb_class": int(row["verb_class"]),
                "noun_class": int(row["noun_class"]),
                "video_id": row["video_id"],
                "participant_id": row["participant_id"],
            }

print(f"Train segments: {sum(1 for v in splits.values() if v['split']=='train')}")
print(f"Val segments: {sum(1 for v in splits.values() if v['split']=='val')}")

# Find all trimmed video files
video_files = list(video_base.rglob("*.mp4")) + list(video_base.rglob("*.MP4"))
print(f"Found {len(video_files)} video files")

written = 0
skipped = 0

for vpath in tqdm(video_files, desc="Converting"):
    # narration_id is the stem (e.g., P01_01_0)
    nid = vpath.stem

    if nid not in splits:
        skipped += 1
        continue

    info = splits[nid]
    split = info["split"]
    tar_path = Path(tar_dir) / split / f"{nid}.tar"

    if tar_path.exists():
        continue

    # Read video frames
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        skipped += 1
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpg = cv2.imencode(".jpg", frame)
        frames.append(jpg.tobytes())
    cap.release()

    if not frames:
        skipped += 1
        continue

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "w") as tar:
        meta = json.dumps({
            "frame_count": len(frames),
            "fps": fps,
            "verb_class": info["verb_class"],
            "noun_class": info["noun_class"],
        })
        mi = tarfile.TarInfo("meta.json")
        mi.size = len(meta)
        tar.addfile(mi, io.BytesIO(meta.encode()))
        for i, jpg_bytes in enumerate(frames):
            fi = tarfile.TarInfo(f"{i:06d}.jpg")
            fi.size = len(jpg_bytes)
            tar.addfile(fi, io.BytesIO(jpg_bytes))

    written += 1

print(f"\nDone: {written} written, {skipped} skipped")
print(f"Train tars: {len(list(Path(tar_dir, 'train').glob('*.tar')))}")
print(f"Val tars: {len(list(Path(tar_dir, 'val').glob('*.tar')))}")
PYEOF

echo ""
echo "=== Setup Complete ==="
echo "Tar-JPEG: $TAR_DIR/{train,val}/"
echo "Annotations: $TAR_DIR/annotations/"
echo ""
echo "Add to eval: DATASETS=\"ek100\" with EK100_LABEL_MODE=\"verb\" or \"noun\""
