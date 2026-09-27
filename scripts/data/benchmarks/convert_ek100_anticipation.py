"""
Extract EK100 anticipation observation windows from untrimmed videos.

For each action at start_frame:
  obs_end   = start_frame - tau * fps      (1s before action)
  obs_start = obs_end - obs_duration * fps  (4s observation window)

Saves as tar-JPEG: output_dir/{train,val}/<narration_id>.tar
Each tar includes meta.json with verb_class and noun_class.

Usage:
    python scripts/data/benchmarks/convert_ek100_anticipation.py \
        --ann-dir eval-dataset/ek100/annotations \
        --video-dir eval-dataset/ek100_untrimmed/videos \
        --output-dir eval-dataset/ek100_anticipation-tar \
        --tau 1.0 --obs-duration 4.0
"""
import argparse
import csv
import io
import json
import os
import tarfile
from pathlib import Path

import cv2
from tqdm import tqdm


def _process_one_video(vid, segs, video_dir, output_dir, tau, obs_duration):
    """Process all segments of a single video. Returns (written, skipped, missing)."""
    pid = segs[0]["participant_id"]
    vpath = find_video(Path(video_dir), vid, pid)
    if vpath is None:
        return 0, len(segs), 1

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return 0, len(segs), 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segs = sorted(segs, key=lambda s: s["start_frame"])
    written, skipped = 0, 0
    output_dir = Path(output_dir)
    for seg in segs:
        nid = seg["narration_id"]
        tar_path = output_dir / seg["split"] / f"{nid}.tar"
        if tar_path.exists():
            continue
        gap = int(fps * tau)
        obs_len = int(fps * obs_duration)
        obs_end = seg["start_frame"] - gap
        obs_start = max(0, obs_end - obs_len)
        if obs_end <= 0 or obs_start >= total_frames:
            skipped += 1
            continue
        obs_end = min(obs_end, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, obs_start)
        frames = []
        for _ in range(obs_end - obs_start):
            ret, frame = cap.read()
            if not ret:
                break
            _, jpg = cv2.imencode(".jpg", frame)
            frames.append(jpg.tobytes())
        if not frames:
            skipped += 1
            continue
        write_tar(tar_path, frames, fps, seg["verb_class"], seg["noun_class"])
        written += 1
    cap.release()
    return written, skipped, 0


def read_annotations(ann_dir: Path):
    segments = []
    for split_name, csv_file in [("train", "EPIC_100_train.csv"), ("validation", "EPIC_100_validation.csv")]:
        split_label = "train" if "train" in split_name else "val"
        with open(ann_dir / csv_file) as f:
            for row in csv.DictReader(f):
                segments.append({
                    "narration_id": row["narration_id"],
                    "video_id": row["video_id"],
                    "participant_id": row["participant_id"],
                    "start_frame": int(row["start_frame"]),
                    "stop_frame": int(row["stop_frame"]),
                    "verb_class": int(row["verb_class"]),
                    "noun_class": int(row["noun_class"]),
                    "split": split_label,
                })
    return segments


def find_video(video_dir: Path, video_id: str, participant_id: str) -> Path | None:
    candidates = [
        video_dir / participant_id / "videos" / f"{video_id}.MP4",
        video_dir / participant_id / "videos" / f"{video_id}.mp4",
        video_dir / participant_id / f"{video_id}.MP4",
        video_dir / participant_id / f"{video_id}.mp4",
        video_dir / f"{video_id}.MP4",
        video_dir / f"{video_id}.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def write_tar(tar_path: Path, frames: list[bytes], fps: float, verb_class: int, noun_class: int):
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "w") as tar:
        meta = json.dumps({
            "frame_count": len(frames), "fps": fps,
            "verb_class": verb_class, "noun_class": noun_class,
        })
        mi = tarfile.TarInfo("meta.json")
        mi.size = len(meta)
        tar.addfile(mi, io.BytesIO(meta.encode()))
        for i, jpg_bytes in enumerate(frames):
            fi = tarfile.TarInfo(f"{i:06d}.jpg")
            fi.size = len(jpg_bytes)
            tar.addfile(fi, io.BytesIO(jpg_bytes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--obs-duration", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    args = parser.parse_args()

    segments = read_annotations(args.ann_dir)
    print(f"Segments: {len(segments)} (tau={args.tau}s, obs={args.obs_duration}s)")

    # Group by video for efficient sequential reading
    by_video = {}
    for seg in segments:
        vid = seg["video_id"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(seg)

    # Stable order
    video_jobs = [(vid, segs) for vid, segs in sorted(by_video.items())]
    print(f"Videos to process: {len(video_jobs)}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    written = 0
    skipped = 0
    video_miss = 0

    n_workers = args.workers if args.workers else max(1, (os.cpu_count() or 1) - 1)
    print(f"Workers: {n_workers}")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_one_video, vid, segs, args.video_dir, args.output_dir,
                          args.tau, args.obs_duration): vid for vid, segs in video_jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Videos"):
            w, s, m = fut.result()
            written += w
            skipped += s
            video_miss += m

    print(f"\nDone: {written} written, {skipped} skipped, {video_miss} videos not found")
    for split in ["train", "val"]:
        count = len(list((args.output_dir / split).glob("*.tar")))
        print(f"  {split}: {count} tars")


if __name__ == "__main__":
    main()
