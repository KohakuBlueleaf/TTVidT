#!/bin/bash
# Setup Jester (Hand Gesture Recognition) dataset
#
# Downloads annotations, extracts video frames, and organizes for eval.
#
# Usage:
#   ./setup_jester.sh [output_dir]
#   ./setup_jester.sh eval-dataset/jester
#
# Jester video data requires manual download from Qualcomm (registration required).
# This script handles:
#   - Downloading annotation CSVs (automatic)
#   - Extracting multi-part archives (if present)
#   - Organizing into the expected structure
#
# Output structure (ready for JesterDataset loader):
#   <output_dir>/
#     videos/
#       1/  2/  3/  ... (148k folders, one per video)
#         00001.jpg  00002.jpg  ...
#     jester-v1-labels.csv
#     jester-v1-train.csv
#     jester-v1-validation.csv

set -euo pipefail

DST_DIR="${1:-./jester}"

LABELS_BASE_URL="https://raw.githubusercontent.com/udacity/CVND---Gesture-Recognition/master/20bn-jester-v1/annotations"

echo "=== Jester Dataset Setup ==="
echo "Output: $DST_DIR"
echo ""

mkdir -p "$DST_DIR/videos"

# --- Step 1: Check if already set up ---
FRAME_DIR_COUNT=$(find "$DST_DIR/videos" -type d -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
if [ "$FRAME_DIR_COUNT" -gt 140000 ]; then
    echo "Dataset already set up: $FRAME_DIR_COUNT video folders"
    # Still ensure annotations exist
else
    echo "Video folders found: $FRAME_DIR_COUNT (expected: ~148092)"
fi

# --- Step 2: Download annotation CSVs ---
echo ""
echo "=== Step 1: Downloading annotations ==="

for CSV_FILE in jester-v1-labels.csv jester-v1-train.csv jester-v1-validation.csv; do
    if [ -f "$DST_DIR/$CSV_FILE" ]; then
        echo "  $CSV_FILE: already present"
    else
        echo "  Downloading $CSV_FILE..."
        curl -sL "$LABELS_BASE_URL/$CSV_FILE" -o "$DST_DIR/$CSV_FILE"

        if [ ! -s "$DST_DIR/$CSV_FILE" ]; then
            echo "  WARNING: Downloaded file is empty, removing"
            rm -f "$DST_DIR/$CSV_FILE"
        fi
    fi
done

# --- Step 3: Extract video frames if archives are present ---
if [ "$FRAME_DIR_COUNT" -lt 140000 ]; then
    echo ""
    echo "=== Step 2: Extracting video frames ==="

    # Look for multi-part archives in common locations
    SRC_DIR=""
    for candidate in \
        "$DST_DIR/../jester" \
        "$DST_DIR/.." \
        "$DST_DIR"; do
        if [ -f "$candidate/20bnjester-v1-00" ] || [ -f "$candidate/20bn-jester-v1-00" ]; then
            SRC_DIR="$candidate"
            break
        fi
    done

    if [ -n "$SRC_DIR" ]; then
        echo "Found source archives in: $SRC_DIR"

        # Find all parts
        PARTS=$(ls "$SRC_DIR"/20bn*jester-v1-* 2>/dev/null | sort)
        PART_COUNT=$(echo "$PARTS" | wc -l)
        echo "Found $PART_COUNT parts"

        echo "Extracting (this may take a while for ~23 GB)..."
        cat $PARTS | tar -xzf - -C "$DST_DIR/videos/" --strip-components=1

        FRAME_DIR_COUNT=$(find "$DST_DIR/videos" -type d -mindepth 1 -maxdepth 1 | wc -l)
        echo "Extracted $FRAME_DIR_COUNT video folders"
    else
        # Check for frames/ directory (alternative location used by some scripts)
        if [ -d "$DST_DIR/frames" ]; then
            FRAMES_IN_FRAMES=$(find "$DST_DIR/frames" -type d -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
            if [ "$FRAMES_IN_FRAMES" -gt 0 ]; then
                echo "Found frames in $DST_DIR/frames/ ($FRAMES_IN_FRAMES folders)"
                echo "Creating symlinks in videos/..."
                # Symlink each video folder from frames/ to videos/
                for d in "$DST_DIR/frames"/*/; do
                    vid_id=$(basename "$d")
                    if [ ! -e "$DST_DIR/videos/$vid_id" ]; then
                        ln -s "$(realpath "$d")" "$DST_DIR/videos/$vid_id"
                    fi
                done
                FRAME_DIR_COUNT=$(find "$DST_DIR/videos" -type d -o -type l -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
                echo "Linked $FRAME_DIR_COUNT video folders"
            fi
        fi

        if [ "$FRAME_DIR_COUNT" -lt 1000 ]; then
            echo ""
            echo "WARNING: Video data not found."
            echo ""
            echo "Jester requires manual download (registration):"
            echo "  1. Go to: https://www.qualcomm.com/developer/software/jester-dataset"
            echo "  2. Register and download all parts (20bnjester-v1-00 through 20bnjester-v1-22)"
            echo "  3. Place them in a directory, e.g.: $DST_DIR/../jester/"
            echo "  4. Re-run this script"
            echo ""
            echo "Annotations have been downloaded. Only video data is missing."
        fi
    fi
fi

# --- Step 4: Verify ---
echo ""
echo "=== Step 3: Verification ==="

FRAME_DIR_COUNT=$(find "$DST_DIR/videos" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) 2>/dev/null | wc -l)
echo "Video folders: $FRAME_DIR_COUNT (expected: ~148092)"

if [ -f "$DST_DIR/jester-v1-labels.csv" ]; then
    LABEL_COUNT=$(wc -l < "$DST_DIR/jester-v1-labels.csv")
    echo "Classes: $LABEL_COUNT (expected: 27)"
fi

if [ -f "$DST_DIR/jester-v1-train.csv" ]; then
    TRAIN_COUNT=$(wc -l < "$DST_DIR/jester-v1-train.csv")
    echo "Training samples: $TRAIN_COUNT (expected: ~118562)"
fi

if [ -f "$DST_DIR/jester-v1-validation.csv" ]; then
    VAL_COUNT=$(wc -l < "$DST_DIR/jester-v1-validation.csv")
    echo "Validation samples: $VAL_COUNT (expected: ~14787)"
fi

# Show sample video
if [ "$FRAME_DIR_COUNT" -gt 0 ]; then
    SAMPLE_DIR=$(find "$DST_DIR/videos" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | head -1)
    if [ -n "$SAMPLE_DIR" ]; then
        FRAME_COUNT=$(ls "$SAMPLE_DIR"/*.jpg 2>/dev/null | wc -l)
        echo "Sample video $(basename "$SAMPLE_DIR"): $FRAME_COUNT frames"
    fi
fi

echo ""
echo "=== Done! ==="
echo "Dataset ready at: $DST_DIR"
echo ""
echo "Usage in eval:"
echo "  JesterDataset(root='$DST_DIR', split='validation')"
