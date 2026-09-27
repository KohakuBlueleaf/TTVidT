#!/bin/bash
# Setup ARID (Action Recognition in the Dark) dataset
#
# Downloads, extracts, and organizes ARID v1.5 so it is ready for eval.
#
# Usage:
#   ./setup_arid.sh [output_dir]
#   ./setup_arid.sh eval-dataset/arid
#
# ARID requires manual download (form submission).
# If the script cannot auto-download, it will tell you where to get the files
# and where to place them, then you can re-run.
#
# Output structure (ready for ARIDDataset loader):
#   <output_dir>/
#     clips_v1.5/
#       Drink/  Jump/  Pick/  Pour/  Push/  Run/  Sit/  Stand/  Turn/  Walk/  Wave/
#     list_cvt/
#       split_0/
#         split0_train.txt
#         split0_test.txt
#       split_1/
#         split1_train.txt
#         split1_test.txt

set -euo pipefail

DST_DIR="${1:-./arid}"
ARID_URL="https://xuyu0010.github.io/arid.html"

echo "=== ARID Dataset Setup ==="
echo "Output: $DST_DIR"
echo ""

mkdir -p "$DST_DIR"

# --- Step 1: Check if already set up ---
if [ -d "$DST_DIR/clips_v1.5" ] && [ -d "$DST_DIR/list_cvt" ]; then
    VIDEO_COUNT=$(find "$DST_DIR/clips_v1.5" -name "*.mp4" 2>/dev/null | wc -l)
    SPLIT_COUNT=$(find "$DST_DIR/list_cvt" -name "*.txt" 2>/dev/null | wc -l)
    if [ "$VIDEO_COUNT" -gt 5000 ] && [ "$SPLIT_COUNT" -ge 4 ]; then
        echo "Dataset already set up: $VIDEO_COUNT videos, $SPLIT_COUNT split files"
        echo "Done!"
        exit 0
    fi
fi

# --- Step 2: Look for downloaded archive ---
# ARID is typically distributed as a zip containing clips_v1.5/ and list_cvt/
ARCHIVE=""
for candidate in \
    "$DST_DIR/../ARID_v1.5.zip" \
    "$DST_DIR/../arid_v1.5.zip" \
    "$DST_DIR/../ARID.zip" \
    "$DST_DIR/../arid.zip" \
    "$DST_DIR/ARID_v1.5.zip" \
    "$DST_DIR/arid.zip"; do
    if [ -f "$candidate" ]; then
        ARCHIVE="$candidate"
        break
    fi
done

if [ -z "$ARCHIVE" ]; then
    echo "ERROR: Cannot find ARID archive."
    echo ""
    echo "ARID requires manual download (form submission):"
    echo "  1. Go to: $ARID_URL"
    echo "  2. Fill out the request form to get download links"
    echo "  3. Download the ARID v1.5 dataset (clips + splits)"
    echo "  4. Place the zip file next to this output directory:"
    echo "       $(cd "$DST_DIR/.." && pwd)/ARID_v1.5.zip"
    echo "  5. Re-run this script"
    echo ""
    echo "Alternatively, if you have the extracted files already,"
    echo "copy clips_v1.5/ and list_cvt/ directly into: $DST_DIR/"
    exit 1
fi

echo "Found archive: $ARCHIVE"

# --- Step 3: Extract ---
echo ""
echo "=== Extracting archive ==="

# Detect archive type
FILE_TYPE=$(file "$ARCHIVE" | head -1)

if echo "$FILE_TYPE" | grep -qi "zip"; then
    unzip -o "$ARCHIVE" -d "$DST_DIR/"
elif echo "$FILE_TYPE" | grep -qi "rar"; then
    if command -v unrar &>/dev/null; then
        unrar x -o+ "$ARCHIVE" "$DST_DIR/"
    elif command -v 7z &>/dev/null; then
        7z x -y -o"$DST_DIR" "$ARCHIVE"
    else
        echo "ERROR: unrar or 7z required to extract RAR. Install with:"
        echo "  sudo apt install unrar   # or"
        echo "  sudo apt install p7zip-full"
        exit 1
    fi
elif echo "$FILE_TYPE" | grep -qi "gzip\|tar"; then
    tar -xzf "$ARCHIVE" -C "$DST_DIR/"
else
    echo "ERROR: Unknown archive type: $FILE_TYPE"
    exit 1
fi

# Handle nested directory (e.g., archive extracts to ARID/ or ARID_v1.5/)
for nested in "$DST_DIR"/ARID* "$DST_DIR"/arid*; do
    if [ -d "$nested" ] && [ "$nested" != "$DST_DIR" ]; then
        if [ -d "$nested/clips_v1.5" ]; then
            echo "Moving from nested directory: $nested"
            cp -rn "$nested"/* "$DST_DIR/" 2>/dev/null || true
            rm -rf "$nested"
        fi
    fi
done

# --- Step 4: Verify ---
echo ""
echo "=== Verification ==="

if [ ! -d "$DST_DIR/clips_v1.5" ]; then
    echo "ERROR: clips_v1.5/ not found after extraction"
    echo "Please check archive contents and ensure clips_v1.5/ is present"
    exit 1
fi

if [ ! -d "$DST_DIR/list_cvt" ]; then
    echo "ERROR: list_cvt/ not found after extraction"
    echo "Please ensure split files (list_cvt/) are included in the download"
    exit 1
fi

VIDEO_COUNT=$(find "$DST_DIR/clips_v1.5" -name "*.mp4" | wc -l)
CLASS_COUNT=$(find "$DST_DIR/clips_v1.5" -type d -mindepth 1 -maxdepth 1 | wc -l)
SPLIT_COUNT=$(find "$DST_DIR/list_cvt" -name "*.txt" | wc -l)

echo "Classes: $CLASS_COUNT (expected: 11)"
echo "  Drink, Jump, Pick, Pour, Push, Run, Sit, Stand, Turn, Walk, Wave"
echo "Videos: $VIDEO_COUNT (expected: ~5572)"
echo "Split files: $SPLIT_COUNT"
echo ""

# List classes found
echo "Classes found:"
find "$DST_DIR/clips_v1.5" -type d -mindepth 1 -maxdepth 1 -printf "  %f\n" | sort

echo ""
echo "=== Done! ==="
echo "Dataset ready at: $DST_DIR"
echo ""
echo "Usage in eval:"
echo "  ARIDDataset(root='$DST_DIR', split='test', split_id=0)"
