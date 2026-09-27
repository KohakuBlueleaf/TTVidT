"""Build metadata caches for TarVideoDataset folders.

Generates tarvidds.files.npy and tarvidds.meta.npy so that
TarVideoDataset.__init__ can skip the slow scan+read phase.

Usage:
  python scripts/data/build_tar_index.py data/openvid384-tar
  python scripts/data/build_tar_index.py DIR1 DIR2 --workers 32
"""

import argparse
import io
import json
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_meta(tar_path: str) -> dict:
    """Read meta.json from a tar file. Returns dict with at least fps/frame_count."""
    try:
        with open(tar_path, "rb") as f:
            raw = f.read()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
            meta_f = tar.extractfile(tar.getmember("meta.json"))
            return json.loads(meta_f.read())
    except Exception as e:
        return {"frame_count": 0, "fps": 0.0, "_error": str(e)}


def scan_tar_files(folder: Path) -> list[str]:
    """Find all .tar files under folder."""
    return sorted(str(p) for p in folder.rglob("*.tar") if p.is_file())


def build_cache(folder: Path, workers: int, cache_name: str = "tarvidds"):
    files_cache = folder / f"{cache_name}.files.npy"
    meta_cache = folder / f"{cache_name}.meta.npy"

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {folder}")
    print(f"  Workers: {workers}")
    print(f"  Cache:   {cache_name}.{{files,meta}}.npy")
    print(f"{'=' * 60}")

    # Phase 1: scan filesystem
    print("\n[1/2] Scanning for .tar files...")
    t0 = time.perf_counter()
    tar_files = scan_tar_files(folder)
    scan_time = time.perf_counter() - t0
    print(f"  Found {len(tar_files):,} tar files in {scan_time:.1f}s")

    if not tar_files:
        print("  ERROR: no tar files found, skipping")
        return

    # Phase 2: read metadata in parallel
    print(f"\n[2/2] Reading metadata from {len(tar_files):,} tar files...")
    t0 = time.perf_counter()
    meta_list = [None] * len(tar_files)
    errors = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(read_meta, path): i
            for i, path in enumerate(tar_files)
        }
        with tqdm(total=len(tar_files), unit="file", smoothing=0.05) as pbar:
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    meta = fut.result()
                except Exception as e:
                    meta = {"frame_count": 0, "fps": 0.0, "_error": str(e)}
                if "_error" in meta:
                    errors += 1
                meta_list[idx] = meta
                pbar.update(1)

    read_time = time.perf_counter() - t0

    # Stats
    frame_counts = [m.get("frame_count", 0) for m in meta_list]
    fps_values = [m.get("fps", 0.0) for m in meta_list]
    valid = sum(1 for fc in frame_counts if fc > 0)
    total_frames = sum(frame_counts)

    print(f"\n  Time:         {read_time:.1f}s ({len(tar_files) / read_time:.0f} files/s)")
    print(f"  Valid:        {valid:,} / {len(tar_files):,}")
    print(f"  Errors:       {errors:,}")
    print(f"  Total frames: {total_frames:,}")
    if valid > 0:
        valid_fc = [fc for fc in frame_counts if fc > 0]
        valid_fps = [fp for fp, fc in zip(fps_values, frame_counts) if fc > 0]
        print(f"  Frames/video: min={min(valid_fc)}, median={sorted(valid_fc)[len(valid_fc)//2]}, max={max(valid_fc)}")
        print(f"  FPS range:    {min(valid_fps):.1f} - {max(valid_fps):.1f}")

    # Save
    np.save(files_cache, np.array(tar_files, dtype=object))
    np.save(meta_cache, np.array(meta_list, dtype=object))
    print(f"\n  Saved: {files_cache}")
    print(f"  Saved: {meta_cache}")


def main():
    parser = argparse.ArgumentParser(description="Build TarVideoDataset metadata caches")
    parser.add_argument("folders", nargs="+", help="Tar dataset folder(s)")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers (default 32)")
    parser.add_argument("--cache-name", type=str, default="tarvidds", help="Cache file prefix")
    args = parser.parse_args()

    for folder_str in args.folders:
        folder = Path(folder_str)
        if not folder.is_dir():
            print(f"ERROR: {folder} is not a directory, skipping")
            continue
        build_cache(folder, args.workers, args.cache_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
