#!/bin/bash
# Prepare Something-Something V2 dataset
# Usage: ./setup_sthsthv2.sh data/sthsthv2 eval-dataset/sthsthv2
#
# Input structure (from 20bn download):
#   data/sthsthv2/
#     20bn-something-something-v2-00  (multipart gzip)
#     20bn-something-something-v2-01
#     20bn-something-something-download-package-labels.zip
#
# Output structure:
#   eval-dataset/sthsthv2/
#     videos/
#       1.webm, 2.webm, ...
#     labels/
#       labels.json
#       train.json
#       validation.json
#       test.json
#       test-answers.csv

set -e

SRC_DIR="${1:-data/sthsthv2}"
DST_DIR="${2:-eval-dataset/sthsthv2}"

echo "=== Preparing Something-Something V2 ==="
echo "Source: $SRC_DIR"
echo "Destination: $DST_DIR"

# Check source files exist
if [ ! -f "$SRC_DIR/20bn-something-something-v2-00" ]; then
    echo "ERROR: Source video file not found: $SRC_DIR/20bn-something-something-v2-00"
    exit 1
fi

if [ ! -f "$SRC_DIR/20bn-something-something-download-package-labels.zip" ]; then
    echo "ERROR: Labels file not found: $SRC_DIR/20bn-something-something-download-package-labels.zip"
    exit 1
fi

# Create output directories
mkdir -p "$DST_DIR/videos"
mkdir -p "$DST_DIR/labels"

# Step 1: Extract labels
echo ""
echo "=== Step 1: Extracting labels ==="
if [ -f "$DST_DIR/labels/labels.json" ]; then
    echo "Labels already extracted, skipping..."
else
    unzip -o "$SRC_DIR/20bn-something-something-download-package-labels.zip" -d "$DST_DIR/"
    echo "Labels extracted to $DST_DIR/labels/"
fi

# Step 2: Concatenate and extract video parts
echo ""
echo "=== Step 2: Extracting videos ==="
echo "This may take a while for ~19GB of data..."

# Check if videos already extracted
VIDEO_COUNT=$(find "$DST_DIR/videos" -name "*.webm" 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -gt 200000 ]; then
    echo "Videos already extracted ($VIDEO_COUNT files), skipping..."
else
    # The 20bn files are multipart tar.gz split files
    # Concatenate and extract in one step
    echo "Concatenating and extracting video parts..."

    # Find all parts and sort them
    PARTS=$(ls "$SRC_DIR"/20bn-something-something-v2-* 2>/dev/null | grep -v labels | sort)
    PART_COUNT=$(echo "$PARTS" | wc -l)
    echo "Found $PART_COUNT parts"

    # Concatenate and extract
    # The files are split tar.gz archives, strip top-level dir (20bn-something-something-v2/)
    cat $PARTS | tar -xzf - -C "$DST_DIR/videos/" --strip-components=1

    VIDEO_COUNT=$(find "$DST_DIR/videos" -name "*.webm" | wc -l)
    echo "Extracted $VIDEO_COUNT video files"
fi

# Step 3: Verify
echo ""
echo "=== Step 3: Verification ==="
VIDEO_COUNT=$(find "$DST_DIR/videos" -name "*.webm" | wc -l)
echo "Total videos: $VIDEO_COUNT"

if [ -f "$DST_DIR/labels/train.json" ]; then
    TRAIN_COUNT=$(python3 -c "import json; print(len(json.load(open('$DST_DIR/labels/train.json'))))")
    echo "Train samples in labels: $TRAIN_COUNT"
fi

if [ -f "$DST_DIR/labels/validation.json" ]; then
    VAL_COUNT=$(python3 -c "import json; print(len(json.load(open('$DST_DIR/labels/validation.json'))))")
    echo "Validation samples in labels: $VAL_COUNT"
fi

echo ""
echo "=== Done! ==="
echo "Dataset prepared at: $DST_DIR"
