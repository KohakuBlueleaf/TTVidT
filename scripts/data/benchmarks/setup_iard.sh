#!/bin/bash
# Setup IARD (Invariant Action Recognition Dataset) dataset
#
# Downloads from Harvard Dataverse, extracts, and organizes for eval.
#
# Usage:
#   ./setup_iard.sh [output_dir]
#   ./setup_iard.sh eval-dataset/iard
#
# Output structure (ready for IARDDataset loader):
#   <output_dir>/
#     videos/
#       drink/  eat/  jump/  run/  walk/  [still/]  [plw/]
#     labels.json

set -euo pipefail

DST_DIR="${1:-./iard}"

# Harvard Dataverse API for this dataset
DATAVERSE_DOI="doi:10.7910/DVN/DMT0PG"
DATAVERSE_API="https://dataverse.harvard.edu/api"
DATAVERSE_PAGE="https://dataverse.harvard.edu/dataset.xhtml?persistentId=$DATAVERSE_DOI"

echo "=== IARD Dataset Setup ==="
echo "Output: $DST_DIR"
echo "Source: $DATAVERSE_PAGE"
echo ""

mkdir -p "$DST_DIR/videos"

# --- Step 1: Check if already set up ---
EXISTING=$(find "$DST_DIR/videos" -name "*.avi" 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 100 ]; then
    echo "Dataset already set up: $EXISTING videos"
    echo "Done!"
    exit 0
fi

# --- Step 2: Download tar files from Harvard Dataverse ---
echo "=== Step 1: Downloading from Harvard Dataverse ==="

# Check for pre-downloaded tar files first
TAR_COUNT=$(ls "$DST_DIR"/../IARD/*.tar 2>/dev/null | wc -l || echo 0)
SRC_DIR=""
if [ "$TAR_COUNT" -gt 0 ]; then
    SRC_DIR="$DST_DIR/../IARD"
    echo "Found $TAR_COUNT pre-downloaded tar files in $SRC_DIR"
else
    TAR_COUNT=$(ls "$DST_DIR"/*.tar 2>/dev/null | wc -l || echo 0)
    if [ "$TAR_COUNT" -gt 0 ]; then
        SRC_DIR="$DST_DIR"
        echo "Found $TAR_COUNT pre-downloaded tar files in $SRC_DIR"
    fi
fi

if [ -z "$SRC_DIR" ]; then
    # Try to download via Dataverse API
    echo "Attempting to download from Harvard Dataverse..."
    echo ""

    # Get dataset file listing
    DOWNLOAD_DIR="$DST_DIR/_downloads"
    mkdir -p "$DOWNLOAD_DIR"

    # Try API to list files
    FILE_LIST=$(curl -sL "$DATAVERSE_API/datasets/:persistentId/?persistentId=$DATAVERSE_DOI" 2>/dev/null | \
        python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    files = data.get('data', {}).get('latestVersion', {}).get('files', [])
    for f in files:
        df = f.get('dataFile', {})
        name = df.get('filename', '')
        fid = df.get('id', '')
        if name.endswith('.tar'):
            print(f'{fid}\t{name}')
except:
    pass
" 2>/dev/null) || true

    if [ -n "$FILE_LIST" ]; then
        echo "Found files to download:"
        echo "$FILE_LIST" | while IFS=$'\t' read -r fid fname; do
            echo "  $fname (id: $fid)"
        done
        echo ""

        echo "$FILE_LIST" | while IFS=$'\t' read -r fid fname; do
            OUTFILE="$DOWNLOAD_DIR/$fname"
            if [ -f "$OUTFILE" ]; then
                echo "  $fname: already downloaded"
                continue
            fi
            echo "  Downloading $fname..."
            curl -L -o "$OUTFILE" "$DATAVERSE_API/access/datafile/$fid"
        done

        SRC_DIR="$DOWNLOAD_DIR"
    else
        echo "ERROR: Could not list files from Dataverse API."
        echo ""
        echo "Please download manually:"
        echo "  1. Go to: $DATAVERSE_PAGE"
        echo "  2. Download all .tar files (drink1.tar, eat1.tar, etc.)"
        echo "  3. Place them in: $DST_DIR/../IARD/"
        echo "  4. Re-run this script"
        exit 1
    fi
fi

# --- Step 3: Extract tar files ---
echo ""
echo "=== Step 2: Extracting tar files ==="

for TAR_FILE in "$SRC_DIR"/*.tar; do
    TAR_NAME=$(basename "$TAR_FILE" .tar)
    echo "  Extracting $TAR_NAME..."
    tar -xf "$TAR_FILE" -C "$DST_DIR/videos/"
done

# --- Step 4: Organize by action (flatten nested dirs) ---
echo ""
echo "=== Step 3: Organizing videos ==="

for ACTION in drink eat jump run walk still plw; do
    # Flatten action1/, action2/ into action/
    if ls "$DST_DIR/videos/${ACTION}"[0-9]* 1>/dev/null 2>&1; then
        mkdir -p "$DST_DIR/videos/$ACTION"
        for SUBDIR in "$DST_DIR/videos/${ACTION}"[0-9]*; do
            if [ -d "$SUBDIR" ]; then
                find "$SUBDIR" -name "*.avi" -exec mv {} "$DST_DIR/videos/$ACTION/" \;
                rm -rf "$SUBDIR"
            fi
        done
    fi
done

# Remove macOS resource fork files
find "$DST_DIR/videos" -name "._*" -delete 2>/dev/null || true

# --- Step 5: Create labels.json ---
echo ""
echo "=== Step 4: Creating labels.json ==="

cat > "$DST_DIR/labels.json" << 'EOF'
{
    "classes": ["drink", "eat", "jump", "run", "walk"],
    "extra_classes": ["still", "plw"],
    "actors": ["georgios", "gu", "haim", "oren", "yair"],
    "views": [0, 45, 90, 135, 180],
    "description": "Invariant Action Recognition Dataset from CBMM/MIT",
    "source": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DMT0PG",
    "filename_format": "{action}_{actor}_{background}_{view}_{variant}_{frame}.avi"
}
EOF

# --- Step 6: Verify ---
echo ""
echo "=== Step 5: Verification ==="

TOTAL=0
for ACTION in drink eat jump run walk still plw; do
    if [ -d "$DST_DIR/videos/$ACTION" ]; then
        COUNT=$(find "$DST_DIR/videos/$ACTION" -name "*.avi" 2>/dev/null | wc -l)
        echo "  $ACTION: $COUNT videos"
        TOTAL=$((TOTAL + COUNT))
    fi
done

echo ""
echo "Total videos: $TOTAL"
echo ""

# Cleanup downloads if extraction succeeded
if [ "$TOTAL" -gt 100 ] && [ -d "$DST_DIR/_downloads" ]; then
    echo "Cleaning up downloads..."
    rm -rf "$DST_DIR/_downloads"
fi

echo "=== Done! ==="
echo "Dataset ready at: $DST_DIR"
echo ""
echo "Usage in eval:"
echo "  IARDDataset(root='$DST_DIR', split='test', split_by='actor')"
