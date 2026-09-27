"""Convert video dataset to tar-JPEG format for eval.

Each video becomes a .tar containing numbered JPEG frames + meta.json.
Supports AVI, MP4, WebM, etc. via OpenCV.

Usage:
  python scripts/data/benchmarks/convert_to_tar.py --input /path/to/videos --output /path/to/output-tar
  python scripts/data/benchmarks/convert_to_tar.py --input hmdb51/videos --output hmdb51-tar --preserve-structure
"""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2


def video_to_tar(video_path, tar_path):
    """Convert a single video to tar-JPEG format."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, str(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(jpg.tobytes())
    cap.release()

    if len(frames) == 0:
        return False, str(video_path)

    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    with tarfile.open(tar_path, 'w') as tar:
        # meta.json
        meta = json.dumps({"frame_count": len(frames), "fps": fps})
        meta_bytes = meta.encode()
        meta_info = tarfile.TarInfo("meta.json")
        meta_info.size = len(meta_bytes)
        tar.addfile(meta_info, io.BytesIO(meta_bytes))
        # frames
        for i, jpg_bytes in enumerate(frames):
            info = tarfile.TarInfo(f"{i:06d}.jpg")
            info.size = len(jpg_bytes)
            tar.addfile(info, io.BytesIO(jpg_bytes))

    return True, str(video_path)


def convert_dataset(input_dir, output_dir, preserve_structure=True, num_workers=16):
    """Convert all videos in input_dir to tar-JPEG in output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_exts = {'.avi', '.mp4', '.webm', '.mkv', '.mpg', '.mpeg', '.mov'}
    videos = [p for p in input_dir.rglob('*') if p.suffix.lower() in video_exts]
    print(f"Found {len(videos)} videos in {input_dir}")

    tasks = []
    for video_path in videos:
        if preserve_structure:
            rel = video_path.relative_to(input_dir)
            tar_path = output_dir / rel.with_suffix('.tar')
        else:
            tar_path = output_dir / (video_path.stem + '.tar')
        tasks.append((video_path, tar_path))

    done = 0
    failed = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(video_to_tar, v, t): v for v, t in tasks}
        for future in as_completed(futures):
            success, path = future.result()
            done += 1
            if not success:
                failed.append(path)
            if done % 500 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] converted, {len(failed)} failed")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed[:20]:
            print(f"  {f}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    print(f"\nDone: {done - len(failed)}/{len(tasks)} videos converted to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert videos to tar-JPEG format")
    parser.add_argument("--input", "-i", required=True, help="Input directory with videos")
    parser.add_argument("--output", "-o", required=True, help="Output directory for tar files")
    parser.add_argument("--preserve-structure", action="store_true", default=True,
                        help="Preserve subdirectory structure (default: True)")
    parser.add_argument("--flat", action="store_true", help="Flat output (no subdirs)")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()
    convert_dataset(args.input, args.output,
                    preserve_structure=not args.flat,
                    num_workers=args.workers)
