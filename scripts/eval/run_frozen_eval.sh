#!/usr/bin/env bash
# Frozen evaluation of one model, end to end:
#   extract features -> sanity-check them -> probe with N seeds -> mean +- std
#
#   bash scripts/eval/run_frozen_eval.sh <checkpoint> [name] [datasets] [n_probe_seeds]
#
#   bash scripts/eval/run_frozen_eval.sh ttvidt/<run_id>/checkpoints/epoch=7.ckpt
#   bash scripts/eval/run_frozen_eval.sh release/ttvidt-tt3d-diffcomp ttvidt jester,sthsthv2 3
#
# <checkpoint> is a training .ckpt, a released model dir, or a Hugging Face repo id.
# Results: eval-results/<name>/<name>__seed<S>.json
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT="${1:?checkpoint}"
NAME="${2:-$(basename "$(dirname "$(dirname "$CKPT")")")}"
DS="${3:-hmdb51,arid,iard,jester,sthsthv2}"
NSEEDS="${4:-3}"
[ "${CKPT##*.}" = "ckpt" ] || NAME="${2:-$(basename "$CKPT")}"

echo "=== [1/4] extract ($DS) -> features/$NAME ==="
kogine run scripts/eval/extract_features.py -c configs/eval/frozen.py \
    --set CHECKPOINT_PATH="$CKPT" --set MODEL_NAME="$NAME" --set DATASETS="$DS"

echo "=== [2/4] check ==="
python scripts/eval/check_features.py "$NAME" --datasets "$DS"

echo "=== [3/4] probe x $NSEEDS seeds ==="
OUT="eval-results/$NAME"; mkdir -p "$OUT"
for S in $(seq 0 $((NSEEDS - 1))); do
  python scripts/eval/probe.py "$NAME" --datasets "$DS" --seed "$S" --gpu --out "$OUT/${NAME}__seed${S}.json"
done

echo "=== [4/4] summary ==="
python scripts/eval/aggregate_results.py "$OUT" --datasets "$DS"
