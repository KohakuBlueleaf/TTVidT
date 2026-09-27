#!/bin/bash
# Download untrimmed EK100 videos + extract anticipation observation windows.
#
# Downloads from Academic Torrents (no auth, fastest mirror).
# Then extracts observation windows: for each action at start_frame,
# take frames from [start_frame - (tau+obs_dur)*fps] to [start_frame - tau*fps]
# where tau=1s (anticipation gap) and obs_dur=4s (observation window).
#
# Usage:
#   bash scripts/data/benchmarks/setup_ek100_anticipation.sh [output_dir]
#
# Output:
#   eval-dataset/ek100_anticipation-tar/{train,val}/<narration_id>.tar

set -euo pipefail

DST_DIR="${1:-./eval-dataset/ek100_untrimmed}"
ANT_TAR_DIR="./eval-dataset/ek100_anticipation-tar"
ANN_DIR="./eval-dataset/ek100/annotations"

echo "=== EK100 Untrimmed Video Download + Anticipation Setup ==="
echo "Video dir: $DST_DIR"
echo "Anticipation tar dir: $ANT_TAR_DIR"
echo ""

mkdir -p "$DST_DIR" "$ANT_TAR_DIR/train" "$ANT_TAR_DIR/val"

# --- Step 1: Ensure annotations exist ---
echo "Step 1: Checking annotations..."
if [ ! -f "$ANN_DIR/EPIC_100_train.csv" ]; then
    echo "Annotations not found. Cloning..."
    mkdir -p "$ANN_DIR"
    git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-100-annotations.git "$ANN_DIR"
fi
echo "Annotations: $ANN_DIR"

# --- Step 2: Download untrimmed videos ---
echo ""
echo "Step 2: Downloading untrimmed videos (~100 GB)..."
echo "Trying multiple sources in order of speed..."

# Method 1: Academic Torrents via aria2c (fastest)
if command -v aria2c &> /dev/null; then
    echo "Using aria2c for Academic Torrents download..."
    TORRENT_URL="https://academictorrents.com/download/c92b4a3cd3834e9af9666ac82379ff15ca289a83.torrent"
    aria2c --seed-time=0 --dir="$DST_DIR" --select-file="*.MP4" \
        --max-concurrent-downloads=8 --split=8 \
        "$TORRENT_URL" 2>&1 || echo "aria2c failed, trying alternative..."
fi

# Method 2: Official download script (slower but reliable)
if [ "$(find "$DST_DIR" -name '*.MP4' 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "Trying official download script..."
    DLSCRIPT_DIR="$DST_DIR/download-scripts"
    if [ ! -d "$DLSCRIPT_DIR" ]; then
        git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git "$DLSCRIPT_DIR"
    fi
    cd "$DLSCRIPT_DIR"
    python3 epic_downloader.py --videos --output-path "$DST_DIR/videos" || {
        echo ""
        echo "=== DOWNLOAD FAILED ==="
        echo "Manual alternatives:"
        echo "  1. Academic Torrents: https://academictorrents.com/details/c92b4a3cd3834e9af9666ac82379ff15ca289a83"
        echo "  2. Bristol mirror: https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m"
        echo "  3. HuggingFace: https://huggingface.co/datasets/awsaf49/epic_kitchens_100"
        echo ""
        echo "Place .MP4 files in: $DST_DIR/videos/<participant_id>/<video_id>.MP4"
        cd - > /dev/null
        exit 1
    }
    cd - > /dev/null
fi

# Locate video directory (may be nested)
VIDEO_DIR="$DST_DIR/videos"
if [ ! -d "$VIDEO_DIR" ]; then
    # Check for EPIC-KITCHENS subdirectory
    for candidate in "$DST_DIR/EPIC-KITCHENS" "$DST_DIR/EPIC_KITCHENS"; do
        if [ -d "$candidate" ]; then
            VIDEO_DIR="$candidate"
            break
        fi
    done
fi

VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.MP4" -o -name "*.mp4" 2>/dev/null | wc -l)
echo "Found $VIDEO_COUNT video files in $VIDEO_DIR"

if [ "$VIDEO_COUNT" -lt 100 ]; then
    echo "ERROR: Too few videos found. Check download."
    exit 1
fi

# --- Step 3: Extract anticipation observation windows ---
echo ""
echo "Step 3: Extracting anticipation observation windows..."
python3 scripts/data/benchmarks/convert_ek100_anticipation.py \
    --ann-dir "$ANN_DIR" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$ANT_TAR_DIR" \
    --tau 1.0 \
    --obs-duration 4.0

# Symlink annotations
ln -sfn "$(realpath "$ANN_DIR")" "$ANT_TAR_DIR/annotations" 2>/dev/null || true

echo ""
echo "=== Setup Complete ==="
echo "Anticipation tar-JPEG: $ANT_TAR_DIR/{train,val}/"
echo ""
echo "Add to eval: DATASETS=\"ek100_anticipation_verb,ek100_anticipation_noun\""
