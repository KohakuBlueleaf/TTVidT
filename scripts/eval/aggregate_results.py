#!/usr/bin/env python3
"""Mean +- std over probe-seed result files, as a markdown table.

Reads every <model>__probeseed<S>.json (or <model>__seed<S>.json) in the given
directories -- the files written by `probe.py --seed S` -- and aggregates one
metric per (model, dataset) over the seeds found.

Usage:
  python scripts/eval/aggregate_results.py eval-results/<name> [more dirs...]
         [--metric attentive|linear_lw|linear_mean|mlp_lw|mlp_mean|knn] [--datasets a,b,c]
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

DEFAULT_DS = ["hmdb51", "arid", "iard", "jester", "sthsthv2"]
FNAME = re.compile(r"^(?P<model>.+?)__(?:probe)?seed(?P<seed>\d+)\.json$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--metric", default="attentive")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DS))
    a = ap.parse_args()
    ds_list = [d.strip() for d in a.datasets.split(",") if d.strip()]

    vals = defaultdict(lambda: defaultdict(dict))  # model -> ds -> seed -> value
    for d in a.dirs:
        for f in sorted(Path(d).glob("*.json")):
            m = FNAME.match(f.name)
            if not m:
                continue
            data = json.load(open(f))
            for key, rec in data.items():
                if not isinstance(rec, dict) or a.metric not in rec:
                    continue
                for ds in ds_list:
                    if key == f"{m['model']}_{ds}":
                        vals[m["model"]][ds][int(m["seed"])] = rec[a.metric]
    if not vals:
        raise SystemExit("no matching result files")

    print(f"metric: {a.metric}; cell = mean +- std (n seeds)\n")
    print("| model | " + " | ".join(ds_list) + " |")
    print("|---|" + "---|" * len(ds_list))
    for model in sorted(vals):
        cells = []
        for ds in ds_list:
            v = list(vals[model][ds].values())
            if not v:
                cells.append("--")
            elif len(v) == 1:
                cells.append(f"{v[0]:.2f} (1)")
            else:
                cells.append(f"{st.mean(v):.2f} +- {st.stdev(v):.2f} ({len(v)})")
        print(f"| {model} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
