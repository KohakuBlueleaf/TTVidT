"""Convert nested mp4 dataset to tar-of-JPEG, preserving folder structure.

Usage:
  python scripts/data/convert_videos_to_tar.py SRC_DIR DST_DIR [--quality 85] [--workers 56] [--dry-run]

Examples:
  # OpenVid384px
  python scripts/data/convert_videos_to_tar.py \
    data/openvid384 \
    data/openvid384-tar \
    --workers 56

  # Moments in Time
  python scripts/data/convert_videos_to_tar.py \
    data/moments_in_time \
    data/moments_in_time-tar \
    --workers 56

Preserves nested folder structure: src/sub/video.mp4 → dst/sub/video.tar
Uses ffmpeg CLI for fast H.264→JPEG conversion (all in C).
Temp files go to /tmp (local disk), not NFS.
Skips already-converted files (resume-safe).
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def convert_one(src_path: str, dst_path: str, qv: int) -> dict:
    """Convert one mp4 → tar of JPEGs. Returns stats dict."""
    # Skip if already converted
    if os.path.exists(dst_path):
        return {"skipped": True, "src": src_path}

    src_size = os.path.getsize(src_path)

    # Single ffmpeg call: decode + encode to JPEG in /tmp
    tmp_dir = tempfile.mkdtemp(prefix="v2tar_")
    try:
        frame_pattern = os.path.join(tmp_dir, "%06d.jpg")
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-i", src_path,
            "-q:v", str(qv),
            "-start_number", "0",
            frame_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)

        frame_files = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".jpg"))
        n_frames = len(frame_files)
        if n_frames == 0:
            return {"failed": True, "src": src_path, "error": "no frames"}

        # Get fps via ffprobe (one call, fast)
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,width,height",
            "-print_format", "json", src_path,
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if probe.returncode == 0:
            info = json.loads(probe.stdout)
            stream = info.get("streams", [{}])[0]
            fps_str = stream.get("avg_frame_rate", "30/1")
            if "/" in fps_str:
                n, d = fps_str.split("/")
                fps = float(n) / float(d) if float(d) != 0 else 30.0
            else:
                fps = float(fps_str) if fps_str else 30.0
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
        else:
            fps, width, height = 30.0, 0, 0

        meta = {"fps": fps, "frame_count": n_frames, "width": width, "height": height}

        # Build tar in memory → single NFS write
        buf = io.BytesIO()
        total_jpg = 0
        with tarfile.open(fileobj=buf, mode="w") as tar:
            meta_bytes = json.dumps(meta).encode()
            ti = tarfile.TarInfo(name="meta.json")
            ti.size = len(meta_bytes)
            tar.addfile(ti, io.BytesIO(meta_bytes))

            for i, fname in enumerate(frame_files):
                fpath = os.path.join(tmp_dir, fname)
                jpg_data = open(fpath, "rb").read()
                total_jpg += len(jpg_data)
                ti = tarfile.TarInfo(name=f"{i:06d}.jpg")
                ti.size = len(jpg_data)
                tar.addfile(ti, io.BytesIO(jpg_data))

        tar_bytes = buf.getvalue()

        # Ensure parent dir exists, write tar
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "wb") as f:
            f.write(tar_bytes)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "src": src_path,
        "frames": n_frames,
        "src_bytes": src_size,
        "tar_bytes": len(tar_bytes),
        "jpg_bytes": total_jpg,
    }


def find_videos(src_dir: Path, cache_name: str = "dataset_tc"):
    """Find all video files. Uses .files.npy cache if available (instant)."""
    cache = src_dir / f"{cache_name}.files.npy"
    if cache.exists():
        import numpy as np
        files = np.load(cache, allow_pickle=True)
        print(f"  Loaded {len(files)} paths from {cache.name} (cached)")
        return [str(f) for f in files]

    print(f"  No cache found, scanning filesystem (slow on NFS)...")
    exts = {".mp4", ".webm", ".avi", ".mkv", ".mov"}
    result = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if Path(f).suffix.lower() in exts:
                result.append(str(Path(root) / f))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description="Convert video dataset to tar-of-JPEG")
    parser.add_argument("src_dir", help="Source directory with nested mp4 files")
    parser.add_argument("dst_dir", help="Destination directory for tar files")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100 (default 85)")
    parser.add_argument("--workers", type=int, default=56, help="Parallel workers (default 56)")
    parser.add_argument("--dry-run", action="store_true", help="List files without converting")
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} not found on PATH")
            sys.exit(1)

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    qv = max(1, min(31, round(31 - (args.quality * 30 / 100))))

    print(f"Scanning {src_dir} for videos...")
    videos = find_videos(src_dir)
    print(f"Found {len(videos)} videos")

    src_str = str(src_dir)

    if args.dry_run:
        # Show structure preview
        subdirs = {}
        for v in videos:
            rel_dir = os.path.dirname(v[len(src_str):].lstrip("/"))
            subdirs[rel_dir] = subdirs.get(rel_dir, 0) + 1
        print(f"Subdirectories: {len(subdirs)}")
        for s in sorted(subdirs)[:20]:
            print(f"  {s}/ ({subdirs[s]} videos)")
        if len(subdirs) > 20:
            print(f"  ... and {len(subdirs) - 20} more")
        return

    # Build task list: (src_path, dst_path)
    tasks = []
    skipped_existing = 0
    for v in videos:
        rel = v[len(src_str):].lstrip("/")
        dst = os.path.join(str(dst_dir), os.path.splitext(rel)[0] + ".tar")
        if os.path.exists(dst):
            skipped_existing += 1
        else:
            tasks.append((v, dst))

    print(f"To convert: {len(tasks)}, already done: {skipped_existing}")
    print(f"Quality: {args.quality} (q:v={qv}), workers: {args.workers}")
    print(f"Output: {dst_dir}")

    if not tasks:
        print("Nothing to do.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    done = 0
    failed = 0
    total_frames = 0
    total_src = 0
    total_tar = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for src, dst in tasks:
            futures[pool.submit(convert_one, src, dst, qv)] = src

        for fut in as_completed(futures):
            done += 1
            try:
                s = fut.result()
                if s.get("failed"):
                    failed += 1
                elif not s.get("skipped"):
                    total_frames += s["frames"]
                    total_src += s["src_bytes"]
                    total_tar += s["tar_bytes"]
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  FAIL: {futures[fut]}: {e}")

            if done % 1000 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(
                    f"  {done:>8}/{len(tasks)} | {rate:.0f} vid/s | "
                    f"ETA {eta/60:.0f}m | fail {failed}"
                )

    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print(f"  Conversion complete")
    print(f"{'=' * 60}")
    print(f"  Videos:       {done - failed} ok, {failed} failed, {skipped_existing} skipped")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Time:         {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Throughput:   {(done-failed)/elapsed:.0f} vid/s, {total_frames/elapsed:.0f} frames/s")
    if total_src > 0:
        print(f"  Source:       {total_src/1e9:.1f} GB")
        print(f"  Target:       {total_tar/1e9:.1f} GB")
        print(f"  Ratio:        {total_tar/total_src:.2f}x")


if __name__ == "__main__":
    main()
