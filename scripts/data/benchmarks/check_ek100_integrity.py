"""Check EK100 untrimmed video integrity with ffprobe. Parallelized."""
import subprocess
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def check_video(mp4_path):
    """Return (path, status) where status is 'ok', 'corrupt:<reason>', or 'missing'."""
    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        return str(mp4_path), "missing"

    # Use ffprobe to check the container and first packet
    try:
        # Check duration, codec, and count streams
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_entries", "format=duration,size:stream=codec_type,nb_frames",
             str(mp4_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            err = result.stderr.strip()[:100] or "ffprobe_fail"
            return str(mp4_path), f"corrupt:{err}"

        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        size = int(data.get("format", {}).get("size", 0))
        streams = data.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]

        if not video_streams:
            return str(mp4_path), "corrupt:no_video_stream"
        if duration < 1.0:
            return str(mp4_path), f"corrupt:short_duration_{duration:.2f}s"

        return str(mp4_path), "ok"
    except subprocess.TimeoutExpired:
        return str(mp4_path), "corrupt:timeout"
    except Exception as e:
        return str(mp4_path), f"corrupt:{type(e).__name__}"


def main():
    root = Path("eval-dataset/ek100_untrimmed")
    videos = list(root.rglob("*.MP4"))
    print(f"Checking {len(videos)} videos...")

    ok, corrupt, missing = 0, [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(check_video, str(v)): v for v in videos}
        for fut in tqdm(as_completed(futures), total=len(videos), desc="ffprobe"):
            path, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "missing":
                missing.append(path)
            else:
                corrupt.append((path, status))

    print(f"\n=== Results ===")
    print(f"OK: {ok}")
    print(f"Missing: {len(missing)}")
    print(f"Corrupt: {len(corrupt)}")

    if corrupt:
        print(f"\nCorrupt files:")
        for path, status in corrupt[:20]:
            print(f"  {path}: {status}")
        if len(corrupt) > 20:
            print(f"  ... +{len(corrupt)-20} more")

        # Save list for re-download
        with open("logs/ek100_corrupt_files.txt", "w") as f:
            for path, status in corrupt:
                f.write(f"{path}\t{status}\n")
        print(f"\nSaved to logs/ek100_corrupt_files.txt")


if __name__ == "__main__":
    main()
