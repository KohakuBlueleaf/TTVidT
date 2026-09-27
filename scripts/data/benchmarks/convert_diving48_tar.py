"""Convert Diving48 MP4 videos to tar-JPEG organized by class label.

Reads Diving48_V2_{train,test}.json annotations and produces:
  <output>/<label_id>/<vid_name>.tar
"""
import argparse
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2


def video_to_tar(video_path: Path, tar_path: Path) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, str(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(jpg.tobytes())
    cap.release()

    if not frames:
        return False, str(video_path)

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        meta = json.dumps({"frame_count": len(frames), "fps": fps}).encode()
        info = tarfile.TarInfo("meta.json")
        info.size = len(meta)
        tar.addfile(info, io.BytesIO(meta))
        for i, jpg in enumerate(frames):
            info = tarfile.TarInfo(f"{i:06d}.jpg")
            info.size = len(jpg)
            tar.addfile(info, io.BytesIO(jpg))
    return True, str(video_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="eval-dataset/diving48")
    parser.add_argument("--output", default="eval-dataset/diving48-tar")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output)
    ann_dir = root / "annotations"
    video_dir = root / "videos"

    with open(ann_dir / "Diving48_V2_train.json") as f:
        train = json.load(f)
    with open(ann_dir / "Diving48_V2_test.json") as f:
        test = json.load(f)

    all_items = train + test
    print(f"Train: {len(train)}, Test: {len(test)}, Total: {len(all_items)}")

    tasks = []
    skipped = 0
    for item in all_items:
        vid = item["vid_name"]
        label = item["label"]
        video_path = video_dir / f"{vid}.mp4"
        if not video_path.exists():
            skipped += 1
            continue
        tar_path = out / str(label) / f"{vid}.tar"
        if tar_path.exists():
            continue
        tasks.append((video_path, tar_path))

    print(f"Tasks: {len(tasks)}, skipped (missing): {skipped}")

    done, failed = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(video_to_tar, v, t): v for v, t in tasks}
        for future in as_completed(futures):
            ok, path = future.result()
            done += 1
            if not ok:
                failed.append(path)
            if done % 500 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] converted, {len(failed)} failed")

    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed[:10]:
            print(f"  {f}")

    print(f"Done: {done - len(failed)}/{len(tasks)} -> {out}")


if __name__ == "__main__":
    main()
