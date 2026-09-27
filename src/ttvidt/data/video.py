"""
FolderVideoDataset - Memory-Safe Video Loading

Supports two backends:
- kohakuclip: Rust-based, statically links FFmpeg (no system FFmpeg needed)
- torchcodec: PyTorch-native, requires system FFmpeg

Backend is auto-detected (prefers kohakuclip), or set explicitly via backend param.
"""

import os
import random
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import transforms as trns
from tqdm import tqdm

Backend = Literal["torchcodec", "kohakuclip", "auto"]


# ============================================================================
# Backend detection
# ============================================================================


def _detect_backend() -> str:
    """Auto-detect available video backend. Prefers kohakuclip."""
    try:
        import kohakuclip

        return "kohakuclip"
    except ImportError:
        pass
    try:
        from torchcodec.decoders import VideoDecoder

        return "torchcodec"
    except ImportError:
        pass
    raise ImportError("No video backend available. Install kohakuclip or torchcodec.")


def _resolve_backend(backend: Backend) -> str:
    if backend != "auto":
        return backend
    return _detect_backend()


# ============================================================================
# Module-level metadata functions (must be picklable for multiprocessing)
# ============================================================================


def _get_meta_torchcodec(file_path: str) -> tuple[int, float]:
    """Get frame count and fps via TorchCodec."""
    from torchcodec.decoders import VideoDecoder

    try:
        decoder = VideoDecoder(file_path, seek_mode="approximate", num_ffmpeg_threads=1)
        meta = decoder.metadata
        frame_count = meta.num_frames
        if frame_count is None or frame_count == 0:
            frame_count = len(decoder)
        fps = meta.average_fps if meta.average_fps else 30.0
        return (frame_count if frame_count else 0, fps)
    except Exception:
        return (0, 0.0)


def _get_meta_kohakuclip(file_path: str) -> tuple[int, float]:
    """Get frame count via KohakuClip, fps via av."""
    try:
        from kohakuclip import KClip

        meta = KClip(file_path).meta()
        frame_count = meta.frame_count or 0
        # KohakuClip doesn't expose fps, use av as fallback
        fps = 30.0
        try:
            import av

            with av.open(file_path) as container:
                stream = container.streams.video[0]
                if stream.average_rate:
                    fps = float(stream.average_rate)
        except Exception:
            pass
        return (frame_count, fps)
    except Exception:
        return (0, 0.0)


class FolderVideoDataset(data.Dataset):
    """Dataset that lazily loads video files from a folder.

    Features:
    - Dual backend: kohakuclip (no FFmpeg dep) or torchcodec
    - Caches file list, frame lengths, and fps for efficient filtering
    - Filters out videos shorter than target_length
    - Multiple sampling modes (variable step, contiguous, fixed FPS)
    - Memory-safe for multiprocessing DataLoader
    """

    VIDEO_EXTENSIONS = ("*.mp4", "*.webm", "*.avi", "*.mkv", "*.mov")

    def __init__(
        self,
        folder_path: Path | str,
        *,
        backend: Backend = "auto",
        resize: tuple[int, int] | None = None,
        target_length: int | None = None,
        transform: Callable | None = None,
        cache_name: str = "dataset_tc",
        max_attempts: int = 8,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        allow_variable_step: bool = True,
        extensions: tuple[str, ...] | None = None,
        # TorchCodec specific (ignored by kohakuclip)
        seek_mode: str = "approximate",
        device: str = "cpu",
        num_ffmpeg_threads: int = 1,
        # Fixed FPS sampling
        sample_fps: float | None = None,
        # Cache building
        cache_workers: int | None = None,
        ) -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"{self.folder_path} does not exist")

        self._backend = _resolve_backend(backend)
        self.resize = resize
        self.target_length = target_length
        self.transform = transform
        self.cache_name = cache_name
        self.max_attempts = max_attempts
        self.dtype = dtype
        self.allow_variable_step = allow_variable_step
        self.extensions = extensions or self.VIDEO_EXTENSIONS
        self.seek_mode = seek_mode
        self.device = device
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.sample_fps = sample_fps
        self.cache_workers = cache_workers or min(32, os.cpu_count() or 4)
        self._rng = random.Random(seed)

        self._meta_fn = (
            _get_meta_torchcodec
            if self._backend == "torchcodec"
            else _get_meta_kohakuclip
        )

        # Cache paths
        self.files_cache = self.folder_path / f"{cache_name}.files.npy"
        self.lengths_cache = self.folder_path / f"{cache_name}.lengths.npy"
        self.fps_cache = self.folder_path / f"{cache_name}.fps.npy"

        # File lists and lengths - stored as numpy arrays for memory safety
        self._all_files: np.ndarray | None = None
        self._all_lengths: np.ndarray | None = None
        self._all_fps: np.ndarray | None = None
        self._valid_indices: np.ndarray | None = None

        # Build caches
        self._ensure_files_cache()
        self._ensure_lengths_cache()
        self._build_valid_indices()

        if len(self._valid_indices) == 0:
            raise ValueError(f"No valid video files found under {self.folder_path}")

        print(f"Using video backend: {self._backend}")

    def _ensure_files_cache(self) -> None:
        """Build cache of all video files if not exists."""
        if self.files_cache.exists():
            self._all_files = np.load(self.files_cache, allow_pickle=True)
            return

        print(f"Building file cache for {self.folder_path}...")
        all_files = []
        for ext in self.extensions:
            all_files.extend(str(p) for p in self.folder_path.rglob(ext) if p.is_file())

        all_files = sorted(set(all_files))
        self._all_files = np.array(all_files, dtype=object)
        np.save(self.files_cache, self._all_files)
        print(f"Found {len(all_files)} video files")

    def _ensure_lengths_cache(self) -> None:
        """Build cache of video frame lengths and fps with multiprocessing."""
        if self.lengths_cache.exists() and self.fps_cache.exists():
            self._all_lengths = np.load(self.lengths_cache)
            self._all_fps = np.load(self.fps_cache)
            if len(self._all_lengths) == len(self._all_files) and len(
                self._all_fps
            ) == len(self._all_files):
                return
            print("Cache size mismatch, rebuilding...")
        elif self.lengths_cache.exists():
            self._all_lengths = np.load(self.lengths_cache)
            if len(self._all_lengths) == len(self._all_files):
                print("Adding FPS cache (one-time migration)...")
                self._add_fps_cache()
                return
            print("Lengths cache size mismatch, rebuilding...")

        print(
            f"Scanning video metadata with {self.cache_workers} workers (one-time operation)..."
        )
        lengths = np.zeros(len(self._all_files), dtype=np.int32)
        fps_array = np.zeros(len(self._all_files), dtype=np.float32)

        file_paths = [str(fp) for fp in self._all_files]

        with ProcessPoolExecutor(max_workers=self.cache_workers) as executor:
            future_to_idx = {
                executor.submit(self._meta_fn, fp): i for i, fp in enumerate(file_paths)
            }

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(self._all_files),
                desc="Scanning video metadata",
            ):
                idx = future_to_idx[future]
                try:
                    frame_count, fps = future.result()
                    lengths[idx] = frame_count
                    fps_array[idx] = fps
                except Exception:
                    lengths[idx] = 0
                    fps_array[idx] = 0.0

        self._all_lengths = lengths
        self._all_fps = fps_array
        np.save(self.lengths_cache, lengths)
        np.save(self.fps_cache, fps_array)

        valid_count = np.sum(lengths > 0)
        print(f"Scanned {len(lengths)} videos, {valid_count} have valid frame counts")

    def _add_fps_cache(self) -> None:
        """Add FPS cache for existing lengths cache (migration helper)."""
        fps_array = np.zeros(len(self._all_files), dtype=np.float32)
        file_paths = [str(fp) for fp in self._all_files]

        with ProcessPoolExecutor(max_workers=self.cache_workers) as executor:
            future_to_idx = {
                executor.submit(self._meta_fn, fp): i for i, fp in enumerate(file_paths)
            }

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(self._all_files),
                desc="Scanning video FPS",
            ):
                idx = future_to_idx[future]
                try:
                    _, fps = future.result()
                    fps_array[idx] = fps
                except Exception:
                    fps_array[idx] = 0.0

        self._all_fps = fps_array
        np.save(self.fps_cache, fps_array)

    def _build_valid_indices(self) -> None:
        """Build indices of valid files based on target_length and sample_fps."""
        if self.sample_fps is not None and self.target_length is not None:
            required_duration = self.target_length / self.sample_fps
            min_frames_needed = required_duration * self._all_fps
            valid_mask = (self._all_lengths >= min_frames_needed) & (self._all_fps > 0)
            self._valid_indices = np.where(valid_mask)[0]
            print(
                f"Valid videos for {self.target_length} frames @ {self.sample_fps} FPS "
                f"(need {required_duration:.2f}s): {len(self._valid_indices)}"
            )
        elif self.target_length is not None:
            self._valid_indices = np.where(self._all_lengths >= self.target_length)[0]
            print(
                f"Valid videos for target_length={self.target_length}: {len(self._valid_indices)}"
            )
        else:
            self._valid_indices = np.where(self._all_lengths > 0)[0]
            print(f"Valid videos: {len(self._valid_indices)}")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if len(self._valid_indices) == 0:
            raise RuntimeError("Dataset is empty")

        attempts = 0
        index = idx % len(self._valid_indices)

        while attempts < self.max_attempts:
            file_idx = self._valid_indices[index]
            file_path = str(self._all_files[file_idx])
            frame_count = int(self._all_lengths[file_idx])
            video_fps = float(self._all_fps[file_idx])
            try:
                return self._load_clip(file_path, frame_count, video_fps)
            except Exception as e:
                attempts += 1
                index = self._rng.randint(0, len(self._valid_indices) - 1)
                if attempts >= self.max_attempts:
                    raise e

    def _load_clip(
        self,
        file_path: str,
        frame_count: int,
        video_fps: float,
        video_bytes: bytes | None = None,
    ) -> torch.Tensor:
        if self._backend == "torchcodec":
            return self._load_clip_torchcodec(
                file_path, frame_count, video_fps, video_bytes=video_bytes
            )
        else:
            return self._load_clip_kohakuclip(file_path, frame_count, video_fps)

    def _load_clip_torchcodec(
        self,
        file_path: str,
        frame_count: int,
        video_fps: float,
        video_bytes: bytes | None = None,
    ) -> torch.Tensor:
        from torchcodec.decoders import VideoDecoder

        # Read entire file into memory first — converts many small random NFS
        # reads (seeks during demux/decode) into a single sequential read.
        # Typical video files are 100-500KB, so the memory cost is negligible.
        if video_bytes is None:
            with open(file_path, "rb") as f:
                video_bytes = f.read()

        decoder = VideoDecoder(
            video_bytes,
            device=self.device,
            seek_mode=self.seek_mode,
            dimension_order="NCHW",
            num_ffmpeg_threads=self.num_ffmpeg_threads,
        )

        actual_frame_count = len(decoder)
        if frame_count <= 0 or frame_count > actual_frame_count:
            frame_count = actual_frame_count
        if video_fps <= 0:
            meta = decoder.metadata
            video_fps = meta.average_fps if meta.average_fps else 30.0

        if self.target_length is not None:
            if self.sample_fps is not None:
                indices = self._sample_fixed_fps(
                    frame_count, video_fps, self.target_length, self.sample_fps
                )
                indices = [min(max(0, i), frame_count - 1) for i in indices]
                frame_batch = decoder.get_frames_at(indices=indices)
            else:
                if frame_count < self.target_length:
                    raise ValueError(
                        f"Video {file_path} has {frame_count} frames, "
                        f"shorter than target {self.target_length}"
                    )
                start, step = self._sample_range(frame_count, self.target_length)
                stop = min(start + self.target_length * step, frame_count)
                frame_batch = decoder.get_frames_in_range(start, stop, step)

            tensor = frame_batch.data  # (T, C, H, W)
        else:
            frame_batch = decoder.get_frames_in_range(0, frame_count)
            tensor = frame_batch.data

        tensor = tensor.to(self.dtype) / 255.0
        return self._post_process(tensor)

    def _load_clip_kohakuclip(
        self, file_path: str, frame_count: int, video_fps: float
    ) -> torch.Tensor:
        from kohakuclip import KClip

        clip = KClip(file_path)

        if frame_count <= 0:
            meta = clip.meta()
            frame_count = meta.frame_count or 0

        if self.target_length is not None:
            if self.sample_fps is not None:
                # Fixed FPS: compute indices, load covering range, then pick
                indices = self._sample_fixed_fps(
                    frame_count, video_fps, self.target_length, self.sample_fps
                )
                indices = [min(max(0, i), frame_count - 1) for i in indices]
                min_i, max_i = min(indices), max(indices)
                frames = clip.range(start=min_i, end=max_i + 1).to_array()
                local_indices = [i - min_i for i in indices]
                frames = frames[local_indices]
            else:
                if frame_count < self.target_length:
                    raise ValueError(
                        f"Video {file_path} has {frame_count} frames, "
                        f"shorter than target {self.target_length}"
                    )
                start, step = self._sample_range(frame_count, self.target_length)
                end = min(start + self.target_length * step, frame_count)
                frames = clip.range(start=start, end=end, step=step).to_array()
        else:
            frames = clip.to_array()

        # NHWC uint8 numpy → NCHW float tensor
        tensor = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(self.dtype)
            / 255.0
        )
        return self._post_process(tensor)

    def _post_process(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply resize and transform. Shared by both backends."""
        if self.resize is not None:
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=self.resize[::-1],  # (H, W) format
                mode="bilinear",
                align_corners=False,
            )
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor

    def _sample_fixed_fps(
        self,
        frame_count: int,
        video_fps: float,
        target_length: int,
        sample_fps: float,
    ) -> list[int]:
        """Sample frames at a fixed FPS rate."""
        step = video_fps / sample_fps
        total_span = (target_length - 1) * step

        max_start = frame_count - 1 - total_span
        if max_start < 0:
            max_start = 0
            step = (frame_count - 1) / max(target_length - 1, 1)

        start = self._rng.uniform(0, max(0, max_start))
        indices = [int(round(start + i * step)) for i in range(target_length)]
        return indices

    def _sample_range(self, frame_count: int, target_length: int) -> tuple[int, int]:
        """Determine start position and step for frame sampling."""
        if not self.allow_variable_step or frame_count < 2 * target_length:
            max_offset = frame_count - target_length
            start = self._rng.randint(0, max_offset)
            return start, 1

        max_step = frame_count // target_length
        step = self._rng.randint(1, max_step)
        required_length = target_length * step
        max_offset = frame_count - required_length

        if max_offset < 0:
            step = 1
            max_offset = frame_count - target_length

        start = self._rng.randint(0, max_offset)
        return start, step

    def metadata(self, idx: int):
        """Get metadata for a specific video."""
        file_idx = self._valid_indices[idx % len(self._valid_indices)]
        file_path = str(self._all_files[file_idx])
        if self._backend == "torchcodec":
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(
                file_path, seek_mode="approximate", num_ffmpeg_threads=1
            )
            return decoder.metadata
        else:
            from kohakuclip import KClip

            return KClip(file_path).meta()

    @property
    def files(self) -> np.ndarray:
        """Return list of valid file paths."""
        return self._all_files[self._valid_indices]

    @property
    def lengths(self) -> np.ndarray:
        """Return frame lengths of valid files."""
        return self._all_lengths[self._valid_indices]

    def rebuild_cache(self, lengths_only: bool = False) -> None:
        """Rebuild the file cache(s)."""
        if not lengths_only:
            if self.files_cache.exists():
                self.files_cache.unlink()
            self._all_files = None
            self._ensure_files_cache()

        if self.lengths_cache.exists():
            self.lengths_cache.unlink()
        if self.fps_cache.exists():
            self.fps_cache.unlink()
        self._all_lengths = None
        self._all_fps = None
        self._ensure_lengths_cache()
        self._build_valid_indices()

    @property
    def fps(self) -> np.ndarray:
        """Return FPS values of valid files."""
        return self._all_fps[self._valid_indices]


# Utility functions


class StreamingVideoDataset(data.IterableDataset):
    """IterableDataset that batch-prefetches video files for NFS performance.

    Each DataLoader worker maintains its own prefetch buffer:
    1. Shuffles the full file list into an epoch order
    2. Splits work across workers (worker_id-based sharding)
    3. Reads prefetch_batch files at once via ThreadPoolExecutor
    4. Yields decoded clips one by one from the buffer
    5. Refills when buffer runs low

    This avoids the idx-mismatch problem of map-style datasets with prefetch,
    and naturally pipelines NFS I/O with GPU-bound decode/transforms.
    """

    VIDEO_EXTENSIONS = ("*.mp4", "*.webm", "*.avi", "*.mkv", "*.mov")

    def __init__(
        self,
        folder_path: Path | str,
        *,
        backend: Backend = "auto",
        resize: tuple[int, int] | None = None,
        target_length: int | None = None,
        transform: Callable | None = None,
        cache_name: str = "dataset_tc",
        max_attempts: int = 8,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        allow_variable_step: bool = True,
        extensions: tuple[str, ...] | None = None,
        seek_mode: str = "approximate",
        device: str = "cpu",
        num_ffmpeg_threads: int = 1,
        sample_fps: float | None = None,
        cache_workers: int | None = None,
        # Prefetch settings
        prefetch_batch: int = 200,
        prefetch_workers: int = 16,
    ) -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"{self.folder_path} does not exist")

        self._backend = _resolve_backend(backend)
        self.resize = resize
        self.target_length = target_length
        self.transform = transform
        self.cache_name = cache_name
        self.max_attempts = max_attempts
        self.dtype = dtype
        self.allow_variable_step = allow_variable_step
        self.extensions = extensions or self.VIDEO_EXTENSIONS
        self.seek_mode = seek_mode
        self.device = device
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.sample_fps = sample_fps
        self.cache_workers = cache_workers or min(32, os.cpu_count() or 4)
        self.prefetch_batch = prefetch_batch
        self.prefetch_workers = prefetch_workers
        self._seed = seed
        self._rng = random.Random(seed)

        self._meta_fn = (
            _get_meta_torchcodec
            if self._backend == "torchcodec"
            else _get_meta_kohakuclip
        )

        # Cache paths — shared with FolderVideoDataset
        self.files_cache = self.folder_path / f"{cache_name}.files.npy"
        self.lengths_cache = self.folder_path / f"{cache_name}.lengths.npy"
        self.fps_cache = self.folder_path / f"{cache_name}.fps.npy"

        self._all_files: np.ndarray | None = None
        self._all_lengths: np.ndarray | None = None
        self._all_fps: np.ndarray | None = None
        self._valid_indices: np.ndarray | None = None

        self._ensure_files_cache()
        self._ensure_lengths_cache()
        self._build_valid_indices()

        if len(self._valid_indices) == 0:
            raise ValueError(f"No valid video files found under {self.folder_path}")

        print(f"StreamingVideoDataset: {len(self._valid_indices)} valid videos, "
              f"backend={self._backend}, prefetch_batch={prefetch_batch}")

    # Reuse FolderVideoDataset's cache building (identical logic)
    _ensure_files_cache = FolderVideoDataset._ensure_files_cache
    _ensure_lengths_cache = FolderVideoDataset._ensure_lengths_cache
    _add_fps_cache = FolderVideoDataset._add_fps_cache
    _build_valid_indices = FolderVideoDataset._build_valid_indices

    # Reuse clip loading and sampling
    _load_clip = FolderVideoDataset._load_clip
    _load_clip_torchcodec = FolderVideoDataset._load_clip_torchcodec
    _load_clip_kohakuclip = FolderVideoDataset._load_clip_kohakuclip
    _post_process = FolderVideoDataset._post_process
    _sample_fixed_fps = FolderVideoDataset._sample_fixed_fps
    _sample_range = FolderVideoDataset._sample_range

    def __len__(self) -> int:
        return len(self._valid_indices)

    def _read_file(self, file_idx: int) -> tuple[int, bytes | None]:
        try:
            with open(str(self._all_files[file_idx]), "rb") as f:
                return (file_idx, f.read())
        except Exception:
            return (file_idx, None)

    def __iter__(self):
        # Per-worker RNG for independent shuffling
        worker_info = data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            rng = random.Random((self._seed or 0) + worker_id)
        else:
            worker_id = 0
            num_workers = 1
            rng = self._rng

        # Shuffle full list, then shard by worker
        order = list(range(len(self._valid_indices)))
        rng.shuffle(order)
        my_order = order[worker_id::num_workers]

        pool = ThreadPoolExecutor(max_workers=self.prefetch_workers)

        # Submit first batch
        cursor = 0
        pending: dict[int, "Future"] = {}  # file_idx → Future[bytes|None]

        def _submit_batch():
            nonlocal cursor
            end = min(cursor + self.prefetch_batch, len(my_order))
            for i in my_order[cursor:end]:
                fi = int(self._valid_indices[i])
                pending[fi] = pool.submit(self._read_file, fi)
            cursor = end

        _submit_batch()

        yielded = 0
        while pending:
            # Refill when half consumed
            if len(pending) <= self.prefetch_batch // 2 and cursor < len(my_order):
                _submit_batch()

            # Pop one completed future, decode, yield
            fi, fut = pending.popitem()
            try:
                _, video_bytes = fut.result()
            except Exception:
                continue
            if video_bytes is None:
                continue

            file_path = str(self._all_files[fi])
            frame_count = int(self._all_lengths[fi])
            video_fps = float(self._all_fps[fi])

            try:
                yield self._load_clip(
                    file_path, frame_count, video_fps, video_bytes=video_bytes
                )
                yielded += 1
            except Exception:
                continue

        pool.shutdown(wait=False)

    @property
    def files(self) -> np.ndarray:
        return self._all_files[self._valid_indices]

    @property
    def lengths(self) -> np.ndarray:
        return self._all_lengths[self._valid_indices]

    @property
    def fps(self) -> np.ndarray:
        return self._all_fps[self._valid_indices]


def random_truncate(length):
    """Create a transform that randomly truncates videos to specified length."""

    def inner(video):
        t, c, h, w = video.shape
        if t <= length:
            return video
        truncate_offset = torch.randint(0, t - length, (1,))
        return video[truncate_offset : truncate_offset + length]

    return inner


def save_video(video: torch.Tensor, path: str, fps: int = 30):
    """Save a video tensor to file."""
    import av

    assert video.ndim == 4 and video.shape[1] == 3, "[T, 3, H, W]"

    video = (video.clamp(-1, 1) * 127.5 + 127.5).byte()
    video = video.permute(0, 2, 3, 1).cpu().numpy()

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = video.shape[2]
    stream.height = video.shape[1]
    stream.pix_fmt = "yuv420p"

    for frame in video:
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


if __name__ == "__main__":
    import gc
    import psutil

    def format_bytes(num_bytes: float) -> str:
        """Format bytes into human readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(num_bytes) < 1024.0:
                return (
                    f"{num_bytes:+.1f}{unit}"
                    if num_bytes != abs(num_bytes)
                    else f"{num_bytes:.1f}{unit}"
                )
            num_bytes /= 1024.0
        return f"{num_bytes:.1f}PB"

    # Get process for memory monitoring
    process = psutil.Process()
    # Record initial memory before dataset creation
    gc.collect()
    mem_before_dataset = process.memory_info().rss

    # Example usage - drop-in replacement for your existing code
    FRAME_COUNT = 49
    BATCH_SIZE = 16
    NUM_WORKERS = 16
    TEST_BATCH_COUNT = 1000

    dataset = FolderVideoDataset(
        "data/openvid384/",
        transform=trns.Compose(
            [
                trns.Resize(256),
                trns.RandomCrop(256),
                trns.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        target_length=FRAME_COUNT,
        allow_variable_step=True,
        seek_mode="approximate",
        cache_name="dataset_tc",
    )
    print(f"Dataset size: {len(dataset)}")

    # Test loading
    gc.collect()
    mem_after_dataset = process.memory_info().rss
    print(
        f"Memory after dataset creation: {format_bytes(mem_after_dataset)} "
        f"(+{format_bytes(mem_after_dataset - mem_before_dataset)} from start)"
    )

    loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Record memory at the start of iteration (after workers spawn)
    # Do a warmup iteration first to let workers initialize
    print("\nWarming up workers...")
    warmup_iter = iter(loader)
    _ = next(warmup_iter)
    del warmup_iter
    gc.collect()

    # Now record baseline after warmup
    mem_baseline = process.memory_info().rss
    print(
        f"Memory after warmup: {format_bytes(mem_baseline)} "
        f"(+{format_bytes(mem_baseline - mem_before_dataset)} from start)"
    )

    print(f"\nTesting {TEST_BATCH_COUNT} batches with memory monitoring...")
    print(f"Baseline memory: {format_bytes(mem_baseline)}")
    print("-" * 60)

    # Test loading with memory monitoring
    pbar = tqdm(
        enumerate(loader),
        total=min(len(loader), TEST_BATCH_COUNT),
        desc="Loading batches",
    )

    batch = None
    max_mem_diff = 0

    for idx, batch in pbar:
        # Get current memory
        current_mem = process.memory_info().rss
        mem_diff = current_mem - mem_baseline
        max_mem_diff = max(max_mem_diff, mem_diff)

        # Update progress bar with memory info
        pbar.set_postfix(
            {
                "mem": format_bytes(current_mem),
                "diff": format_bytes(mem_diff),
                "max_diff": format_bytes(max_mem_diff),
            }
        )

        if idx >= TEST_BATCH_COUNT - 1:
            break

    # Final stats
    gc.collect()
    final_mem = process.memory_info().rss

    print("-" * 60)
    print(f"\n=== Memory Test Results ===")
    print(f"Initial memory (before dataset): {format_bytes(mem_before_dataset)}")
    print(
        f"After dataset creation: {format_bytes(mem_after_dataset)} (+{format_bytes(mem_after_dataset - mem_before_dataset)})"
    )
    print(
        f"After warmup (baseline): {format_bytes(mem_baseline)} (+{format_bytes(mem_baseline - mem_before_dataset)})"
    )
    print(
        f"Final memory: {format_bytes(final_mem)} (+{format_bytes(final_mem - mem_before_dataset)} from start)"
    )
    print(f"Memory growth during iteration: {format_bytes(final_mem - mem_baseline)}")
    print(f"Max memory diff from baseline: {format_bytes(max_mem_diff)}")

    if batch is not None:
        print(f"\nBatch shape: {batch.shape}")  # Should be [B, T, C, H, W]

    # Memory stability check
    growth = final_mem - mem_baseline
    if growth > 500 * 1024 * 1024:  # 500 MB threshold
        print("\nWARNING: Significant memory growth detected!")
    else:
        print("\nMemory usage appears stable")
