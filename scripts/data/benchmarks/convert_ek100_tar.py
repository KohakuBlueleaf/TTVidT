"""Convert EK100 trimmed mp4s to tar-JPEG, organized by split."""
import csv
import io
import json
import tarfile
from pathlib import Path
from tqdm import tqdm
import cv2

dst_dir = "eval-dataset/ek100"
tar_dir = "eval-dataset/ek100-tar"
ann_dir = f"{dst_dir}/annotations"
video_base = Path(dst_dir) / "trimmed_videos"

splits = {}
for split_name, csv_file in [("train", "EPIC_100_train.csv"), ("val", "EPIC_100_validation.csv")]:
    with open(f"{ann_dir}/{csv_file}") as f:
        for row in csv.DictReader(f):
            splits[row["narration_id"]] = {
                "split": split_name,
                "verb_class": int(row["verb_class"]),
                "noun_class": int(row["noun_class"]),
            }

print(f"Train: {sum(1 for v in splits.values() if v['split']=='train')}")
print(f"Val: {sum(1 for v in splits.values() if v['split']=='val')}")

for split in ["train", "val"]:
    vdir = video_base / split
    vfiles = list(vdir.rglob("*.mp4"))
    print(f"\n{split}: {len(vfiles)} videos")
    written = 0
    skipped = 0
    for vpath in tqdm(vfiles, desc=split):
        nid = vpath.stem
        if nid not in splits:
            skipped += 1
            continue
        info = splits[nid]
        tar_path = Path(tar_dir) / split / f"{nid}.tar"
        if tar_path.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            skipped += 1
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, jpg = cv2.imencode(".jpg", frame)
            frames.append(jpg.tobytes())
        cap.release()
        if not frames:
            skipped += 1
            continue
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(tar_path), "w") as tar:
            meta = json.dumps({
                "frame_count": len(frames), "fps": fps,
                "verb_class": info["verb_class"], "noun_class": info["noun_class"],
            })
            mi = tarfile.TarInfo("meta.json")
            mi.size = len(meta)
            tar.addfile(mi, io.BytesIO(meta.encode()))
            for i, jpg_bytes in enumerate(frames):
                fi = tarfile.TarInfo(f"{i:06d}.jpg")
                fi.size = len(jpg_bytes)
                tar.addfile(fi, io.BytesIO(jpg_bytes))
        written += 1
    total = len(list(Path(tar_dir, split).glob("*.tar")))
    print(f"{split}: {written} written, {skipped} skipped, {total} total tars")

print("\nDone!")
