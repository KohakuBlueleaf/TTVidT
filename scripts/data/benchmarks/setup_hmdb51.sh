#!/bin/bash
# Setup HMDB51 dataset
#
# Downloads videos + splits, extracts, and organizes for eval.
#
# Usage:
#   ./setup_hmdb51.sh [output_dir]
#   ./setup_hmdb51.sh eval-dataset/hmdb51
#
# Output structure (ready for HMDB51Dataset loader):
#   <output_dir>/
#     videos/
#       brush_hair/  cartwheel/  catch/  ... (51 classes)
#     splits/
#       brush_hair_test_split1.txt  ... (306 files)

set -euo pipefail

DST_DIR="${1:-./hmdb51}"

VIDEO_RAR_URL="http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
SPLITS_RAR_URL="http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"

echo "=== HMDB51 Dataset Setup ==="
echo "Output: $DST_DIR"
echo ""

mkdir -p "$DST_DIR/videos" "$DST_DIR/splits"

# --- Step 1: Check if already set up ---
VIDEO_COUNT=$(find "$DST_DIR/videos" -name "*.avi" 2>/dev/null | wc -l)
SPLITS_COUNT=$(find "$DST_DIR/splits" -name "*.txt" 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -gt 6000 ] && [ "$SPLITS_COUNT" -gt 100 ]; then
    echo "Dataset already set up: $VIDEO_COUNT videos, $SPLITS_COUNT split files"
    echo "Done!"
    exit 0
fi

# Check for extraction tools
HAS_UNRAR=false
HAS_7Z=false
command -v unrar &>/dev/null && HAS_UNRAR=true
command -v 7z &>/dev/null && HAS_7Z=true

if ! $HAS_UNRAR && ! $HAS_7Z; then
    echo "ERROR: Need 'unrar' or '7z' to extract HMDB51 (RAR format)."
    echo "Install with:"
    echo "  sudo apt install unrar     # Debian/Ubuntu"
    echo "  sudo apt install p7zip-full"
    echo "  sudo yum install unrar     # CentOS/RHEL"
    echo "  brew install unrar         # macOS"
    exit 1
fi

DOWNLOAD_DIR="$DST_DIR/_downloads"
mkdir -p "$DOWNLOAD_DIR"

# --- Step 2: Download videos ---
echo "=== Step 1: Getting video archive ==="

# Look for pre-downloaded files
VIDEO_ARCHIVE=""
for candidate in \
    "$DST_DIR/../hmdb51_org.rar" \
    "$DST_DIR/../hmdb51.zip" \
    "$DST_DIR/hmdb51_org.rar" \
    "$DOWNLOAD_DIR/hmdb51_org.rar"; do
    if [ -f "$candidate" ]; then
        # Check it's not a failed HTML download
        if ! file "$candidate" | grep -qi "HTML"; then
            VIDEO_ARCHIVE="$candidate"
            break
        fi
    fi
done

if [ -z "$VIDEO_ARCHIVE" ] && [ "$VIDEO_COUNT" -lt 6000 ]; then
    echo "Downloading hmdb51_org.rar (~2 GB)..."
    curl -L -o "$DOWNLOAD_DIR/hmdb51_org.rar" "$VIDEO_RAR_URL"

    # Verify it's not HTML
    if file "$DOWNLOAD_DIR/hmdb51_org.rar" | grep -qi "HTML"; then
        echo "ERROR: Download returned HTML instead of RAR."
        echo "The direct link may have changed. Please download manually from:"
        echo "  http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/"
        echo "Place the file at: $DOWNLOAD_DIR/hmdb51_org.rar"
        rm -f "$DOWNLOAD_DIR/hmdb51_org.rar"
        exit 1
    fi

    VIDEO_ARCHIVE="$DOWNLOAD_DIR/hmdb51_org.rar"
    echo "Downloaded."
else
    if [ "$VIDEO_COUNT" -ge 6000 ]; then
        echo "Videos already extracted ($VIDEO_COUNT files), skipping download."
    else
        echo "Using: $VIDEO_ARCHIVE"
    fi
fi

# --- Step 3: Extract videos ---
if [ "$VIDEO_COUNT" -lt 6000 ] && [ -n "$VIDEO_ARCHIVE" ]; then
    echo ""
    echo "=== Step 2: Extracting videos ==="

    FILE_TYPE=$(file "$VIDEO_ARCHIVE" | head -1)

    if echo "$FILE_TYPE" | grep -qi "rar"; then
        # RAR contains per-class RAR files inside
        TEMP_DIR="$DST_DIR/_temp_extract"
        mkdir -p "$TEMP_DIR"

        if $HAS_UNRAR; then
            unrar x -o+ "$VIDEO_ARCHIVE" "$TEMP_DIR/"
        else
            7z x -y -o"$TEMP_DIR" "$VIDEO_ARCHIVE"
        fi

        # Each class is a separate RAR file inside
        echo "Extracting per-class archives..."
        for CLASS_RAR in "$TEMP_DIR"/*.rar; do
            if [ -f "$CLASS_RAR" ]; then
                CLASS_NAME=$(basename "$CLASS_RAR" .rar)
                mkdir -p "$DST_DIR/videos/$CLASS_NAME"
                if $HAS_UNRAR; then
                    unrar x -o+ "$CLASS_RAR" "$DST_DIR/videos/$CLASS_NAME/"
                else
                    7z x -y -o"$DST_DIR/videos/$CLASS_NAME" "$CLASS_RAR"
                fi
            fi
        done
        rm -rf "$TEMP_DIR"

    elif echo "$FILE_TYPE" | grep -qi "zip"; then
        unzip -o "$VIDEO_ARCHIVE" -d "$DST_DIR/"
        # Move from hmdb51/ to videos/ if needed
        if [ -d "$DST_DIR/hmdb51" ]; then
            mv "$DST_DIR/hmdb51"/* "$DST_DIR/videos/" 2>/dev/null || true
            rmdir "$DST_DIR/hmdb51" 2>/dev/null || true
        fi
    fi

    VIDEO_COUNT=$(find "$DST_DIR/videos" -name "*.avi" | wc -l)
    echo "Extracted $VIDEO_COUNT video files"
fi

# --- Step 4: Download and extract splits ---
echo ""
echo "=== Step 3: Getting split files ==="

if [ "$SPLITS_COUNT" -gt 100 ]; then
    echo "Splits already present ($SPLITS_COUNT files), skipping."
else
    SPLITS_ARCHIVE=""
    for candidate in \
        "$DST_DIR/../test_train_splits.rar" \
        "$DST_DIR/../test_train_splits.zip" \
        "$DOWNLOAD_DIR/test_train_splits.rar"; do
        if [ -f "$candidate" ]; then
            SPLITS_ARCHIVE="$candidate"
            break
        fi
    done

    if [ -z "$SPLITS_ARCHIVE" ]; then
        echo "Downloading test_train_splits.rar..."
        curl -L -o "$DOWNLOAD_DIR/test_train_splits.rar" "$SPLITS_RAR_URL"

        if file "$DOWNLOAD_DIR/test_train_splits.rar" | grep -qi "HTML"; then
            echo "WARNING: Splits download returned HTML. Trying alternative..."
            rm -f "$DOWNLOAD_DIR/test_train_splits.rar"
        else
            SPLITS_ARCHIVE="$DOWNLOAD_DIR/test_train_splits.rar"
        fi
    fi

    if [ -n "$SPLITS_ARCHIVE" ]; then
        echo "Extracting splits from: $SPLITS_ARCHIVE"
        FILE_TYPE=$(file "$SPLITS_ARCHIVE" | head -1)

        if echo "$FILE_TYPE" | grep -qi "rar"; then
            if $HAS_UNRAR; then
                unrar x -o+ "$SPLITS_ARCHIVE" "$DST_DIR/splits/"
            else
                7z x -y -o"$DST_DIR/splits" "$SPLITS_ARCHIVE"
            fi
        elif echo "$FILE_TYPE" | grep -qi "zip"; then
            unzip -o "$SPLITS_ARCHIVE" -d "$DST_DIR/splits/"
        fi

        # Flatten if nested in testTrainMulti_7030_splits/
        if [ -d "$DST_DIR/splits/testTrainMulti_7030_splits" ]; then
            mv "$DST_DIR/splits/testTrainMulti_7030_splits"/* "$DST_DIR/splits/" 2>/dev/null || true
            rmdir "$DST_DIR/splits/testTrainMulti_7030_splits" 2>/dev/null || true
        fi
    else
        echo "ERROR: Could not download splits."
        echo "Please download manually from:"
        echo "  $SPLITS_RAR_URL"
        echo "Place at: $DOWNLOAD_DIR/test_train_splits.rar"
    fi
fi

# --- Step 5: Verify ---
echo ""
echo "=== Step 4: Verification ==="

VIDEO_COUNT=$(find "$DST_DIR/videos" -name "*.avi" | wc -l)
CLASS_COUNT=$(find "$DST_DIR/videos" -type d -mindepth 1 -maxdepth 1 | wc -l)
SPLITS_COUNT=$(find "$DST_DIR/splits" -name "*.txt" 2>/dev/null | wc -l)

echo "Videos: $VIDEO_COUNT (expected: ~6766)"
echo "Classes: $CLASS_COUNT (expected: 51)"
echo "Split files: $SPLITS_COUNT (expected: 306)"
echo ""

echo "Sample classes:"
find "$DST_DIR/videos" -type d -mindepth 1 -maxdepth 1 -printf "  %f\n" | sort | head -10
echo "  ..."

# Cleanup downloads
if [ "$VIDEO_COUNT" -gt 6000 ] && [ -d "$DOWNLOAD_DIR" ]; then
    echo ""
    echo "Cleaning up downloads..."
    rm -rf "$DOWNLOAD_DIR"
fi

echo ""
echo "=== Done! ==="
echo "Dataset ready at: $DST_DIR"
echo ""
echo "Usage in eval:"
echo "  HMDB51Dataset(root='$DST_DIR', split='test', split_id=1)"
