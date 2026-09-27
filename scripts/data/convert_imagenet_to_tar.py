"""Convert ImageNet parquet files to tar-of-JPEG files.

Each output tar contains BATCH_SIZE images (default 64), resized so the
shorter edge is at most MAX_SHORT_EDGE (default 384). Images that are
already small enough are passed through without re-encoding.

Output structure per tar:
  meta.json         — {"count": N}
  000000.jpg        — image 0
  000001.jpg        — image 1
  ...

Usage:
  python scripts/data/convert_imagenet_to_tar.py \
    data/imagenet-1k \
    data/imagenet-1k-tar \
    --split train --workers 32 --batch-size 64 --max-short-edge 384
"""

import argparse
import glob
import io
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq
from PIL import Image


def resize_jpeg(jpeg_bytes, max_short_edge):
    """Resize image so shorter edge <= max_short_edge. Returns JPEG bytes."""
    img = Image.open(io.BytesIO(jpeg_bytes))
    w, h = img.size
    short_edge = min(w, h)
    if short_edge <= max_short_edge:
        # Already small enough — but ensure it's RGB JPEG
        if img.mode != "RGB":
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            return buf.getvalue()
        return jpeg_bytes

    # Resize so short edge = max_short_edge
    scale = max_short_edge / short_edge
    new_w = round(w * scale)
    new_h = round(h * scale)
    img = img.convert("RGB")
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def build_tar(images, tar_path):
    """Pack a list of JPEG bytes into a tar file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        meta = json.dumps({"count": len(images)}).encode()
        ti = tarfile.TarInfo(name="meta.json")
        ti.size = len(meta)
        tar.addfile(ti, io.BytesIO(meta))

        for i, jpeg_bytes in enumerate(images):
            ti = tarfile.TarInfo(name=f"{i:06d}.jpg")
            ti.size = len(jpeg_bytes)
            tar.addfile(ti, io.BytesIO(jpeg_bytes))

    with open(tar_path, "wb") as f:
        f.write(buf.getvalue())
    return len(images), buf.tell()


def process_parquet(parquet_path, out_dir, start_tar_idx, batch_size, max_short_edge):
    """Convert one parquet file into multiple tar files of batch_size images."""
    table = pq.read_table(parquet_path, columns=["image"])
    n = len(table)
    images_col = table.column("image")

    results = []
    batch = []
    tar_idx = start_tar_idx

    for i in range(n):
        jpeg_bytes = images_col[i].as_py()["bytes"]
        jpeg_bytes = resize_jpeg(jpeg_bytes, max_short_edge)
        batch.append(jpeg_bytes)

        if len(batch) == batch_size:
            tar_path = os.path.join(out_dir, f"batch_{tar_idx:06d}.tar")
            n_imgs, n_bytes = build_tar(batch, tar_path)
            results.append((tar_idx, n_imgs, n_bytes))
            batch = []
            tar_idx += 1

    # Leftover
    if batch:
        tar_path = os.path.join(out_dir, f"batch_{tar_idx:06d}.tar")
        n_imgs, n_bytes = build_tar(batch, tar_path)
        results.append((tar_idx, n_imgs, n_bytes))
        tar_idx += 1

    return results, tar_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="ImageNet HF dataset dir (contains data/)")
    parser.add_argument("dst_dir", help="Output tar directory")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-short-edge", type=int, default=384)
    args = parser.parse_args()

    pattern = os.path.join(args.src_dir, "data", f"{args.split}-*.parquet")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} parquet files for split={args.split}")

    os.makedirs(args.dst_dir, exist_ok=True)

    # Check what's already done
    existing = set(os.listdir(args.dst_dir))
    if existing:
        print(f"Output dir has {len(existing)} existing files, starting fresh conversion")

    t0 = time.perf_counter()
    total_images = 0
    total_bytes = 0
    total_tars = 0
    done_parquets = 0

    # Process parquet files in parallel
    # Each parquet produces multiple tars, so we need to assign tar indices sequentially
    # First pass: count images per parquet to pre-assign tar index ranges
    print("Counting images per parquet shard...")
    counts = []
    for f in files:
        meta = pq.read_metadata(f)
        counts.append(meta.num_rows)
    total_count = sum(counts)
    print(f"Total images: {total_count}, tars needed: ~{total_count // args.batch_size}")

    # Assign tar index ranges
    tar_starts = []
    cur = 0
    for c in counts:
        tar_starts.append(cur)
        n_tars = (c + args.batch_size - 1) // args.batch_size
        cur += n_tars

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, f in enumerate(files):
            fut = pool.submit(
                process_parquet, f, args.dst_dir,
                tar_starts[i], args.batch_size, args.max_short_edge,
            )
            futures[fut] = f

        for fut in as_completed(futures):
            done_parquets += 1
            try:
                results, _ = fut.result()
                for _, n_imgs, n_bytes in results:
                    total_images += n_imgs
                    total_bytes += n_bytes
                    total_tars += 1
            except Exception as e:
                print(f"  FAIL: {futures[fut]}: {e}")
            if done_parquets % 20 == 0 or done_parquets == len(files):
                elapsed = time.perf_counter() - t0
                print(
                    f"  [{done_parquets}/{len(files)}] "
                    f"{total_images} images, {total_tars} tars, "
                    f"{total_bytes/1e9:.1f} GB, {elapsed:.0f}s"
                )

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {total_images} images in {total_tars} tars, {total_bytes/1e9:.1f} GB, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
