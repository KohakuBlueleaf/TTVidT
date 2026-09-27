#!/bin/bash
# Fully automated Diving48 setup — downloads videos + annotations from HuggingFace.
#
# Usage:
#   bash scripts/data/benchmarks/setup_diving48.sh [output_dir]
#   bash scripts/data/benchmarks/setup_diving48.sh eval-dataset/diving48

set -euo pipefail

DST_DIR="${1:-./eval-dataset/diving48}"
TAR_DIR="${DST_DIR}-tar"

echo "=== Diving48 Dataset Setup (Fully Automated) ==="
echo "Output: $DST_DIR"
echo "Tar output: $TAR_DIR"
echo ""

mkdir -p "$DST_DIR/videos" "$DST_DIR/annotations"

# --- Step 1: Download videos + annotations from HuggingFace ---
echo "Step 1: Downloading from HuggingFace..."
python3 << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

dst = os.environ.get("DST_DIR", sys.argv[1] if len(sys.argv) > 1 else "./eval-dataset/diving48")

# Videos
rgb_path = os.path.join(dst, "Diving48_rgb.tar.gz")
if not os.path.exists(rgb_path):
    print("Downloading Diving48_rgb.tar.gz (~10.3 GB)...")
    hf_hub_download("bkprocovid19/diving48", "Diving48_rgb.tar.gz",
                     repo_type="dataset", local_dir=dst)
else:
    print(f"Already downloaded: {rgb_path}")

# Annotations — try HuggingFace first, fall back to official URL
import urllib.request
ann_dir = os.path.join(dst, "annotations")
os.makedirs(ann_dir, exist_ok=True)

for fname in ["Diving48_V2_train.json", "Diving48_V2_test.json", "Diving48_vocab.json"]:
    fpath = os.path.join(ann_dir, fname)
    if os.path.exists(fpath):
        print(f"  Already have: {fname}")
        continue
    # Try HuggingFace
    try:
        hf_hub_download("bkprocovid19/diving48", fname,
                         repo_type="dataset", local_dir=ann_dir)
        print(f"  Downloaded from HuggingFace: {fname}")
        continue
    except Exception:
        pass
    # Try official URL
    url = f"http://www.svcl.ucsd.edu/projects/resound/{fname}"
    try:
        urllib.request.urlretrieve(url, fpath)
        print(f"  Downloaded from UCSD: {fname}")
    except Exception as e:
        print(f"  ERROR: Could not download {fname}: {e}")
        print(f"  Please manually place {fname} in {ann_dir}/")
        sys.exit(1)
PYEOF

export DST_DIR

# --- Step 2: Extract videos ---
echo ""
echo "Step 2: Extracting videos..."
if [ "$(find "$DST_DIR/videos/" -name "*.mp4" 2>/dev/null | wc -l)" -lt 100 ]; then
    tar -xzf "$DST_DIR/Diving48_rgb.tar.gz" -C "$DST_DIR/videos/" --strip-components=1 2>/dev/null || \
    tar -xzf "$DST_DIR/Diving48_rgb.tar.gz" -C "$DST_DIR/" 2>/dev/null || true

    # Handle case where videos extract into a subdirectory
    if [ -d "$DST_DIR/rgb" ]; then
        mv "$DST_DIR/rgb/"* "$DST_DIR/videos/" 2>/dev/null || true
        rmdir "$DST_DIR/rgb" 2>/dev/null || true
    fi

    COUNT=$(find "$DST_DIR/videos/" -name "*.mp4" | wc -l)
    echo "Extracted $COUNT videos"
else
    COUNT=$(find "$DST_DIR/videos/" -name "*.mp4" | wc -l)
    echo "Already extracted: $COUNT videos"
fi

# --- Step 3: Verify annotations ---
echo ""
echo "Step 3: Verifying annotations..."
for ann in Diving48_V2_train.json Diving48_V2_test.json Diving48_vocab.json; do
    if [ ! -f "$DST_DIR/annotations/$ann" ]; then
        echo "  MISSING: $ann — setup cannot continue"
        exit 1
    fi
    echo "  OK: $ann"
done

# --- Step 4: Convert to tar-JPEG ---
echo ""
echo "Step 4: Converting to tar-JPEG format..."
python3 << 'PYEOF'
import json, os, sys, tarfile, io, cv2
from pathlib import Path
from tqdm import tqdm

dst_dir = os.environ.get("DST_DIR", "./eval-dataset/diving48")
tar_dir = dst_dir + "-tar"

with open(f"{dst_dir}/annotations/Diving48_V2_train.json") as f:
    train_data = json.load(f)
with open(f"{dst_dir}/annotations/Diving48_V2_test.json") as f:
    test_data = json.load(f)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

video_dir = Path(dst_dir) / "videos"
all_data = train_data + test_data
skipped = 0

for item in tqdm(all_data, desc="Converting"):
    vid_id = item["vid_name"]
    label = item["label"]
    class_dir = Path(tar_dir) / str(label)
    tar_path = class_dir / f"{vid_id}.tar"

    if tar_path.exists():
        continue

    video_path = None
    for ext in [".mp4", ".avi", ".mkv"]:
        c = video_dir / f"{vid_id}{ext}"
        if c.exists():
            video_path = c
            break

    if video_path is None:
        skipped += 1
        continue

    cap = cv2.VideoCapture(str(video_path))
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

    class_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "w") as tar:
        meta = json.dumps({"frame_count": len(frames), "fps": fps})
        mi = tarfile.TarInfo("meta.json")
        mi.size = len(meta)
        tar.addfile(mi, io.BytesIO(meta.encode()))
        for i, jpg_bytes in enumerate(frames):
            fi = tarfile.TarInfo(f"{i:06d}.jpg")
            fi.size = len(jpg_bytes)
            tar.addfile(fi, io.BytesIO(jpg_bytes))

if skipped:
    print(f"Skipped {skipped} videos (missing/empty)")
print(f"Done! Tar-JPEG in: {tar_dir}/")
PYEOF

echo ""
echo "=== Setup Complete ==="
echo "Videos: $DST_DIR/videos/"
echo "Tar-JPEG: $TAR_DIR/"
echo ""
echo "Add to eval: DATASETS=\"diving48\""
