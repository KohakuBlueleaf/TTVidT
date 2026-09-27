"""TarVideoDataset — load pre-decoded JPEG frames from tar files.

Each .tar file contains:
  meta.json  — {"fps", "frame_count", "width", "height"}
  000000.jpg — frame 0, 000001.jpg — frame 1, ...

Loading strategy:
  1. Read entire tar file to memory (single sequential NFS read, ~1-5MB)
  2. Parse tar TOC in memory (no disk seeks)
  3. Sample frame indices based on target_length / sample_fps
  4. JPEG-decode only the needed frames (skip unneeded ones)

JPEG decode is ~10-50x faster than H.264, and the read pattern is perfectly
sequential — ideal for NFS.
"""

import io
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import transforms as trns


class TarVideoDataset(data.Dataset):
    """Map-style dataset loading video frames from tar-of-JPEG files.

    Designed for maximum NFS throughput:
    - Single sequential read per video (read entire tar to bytes)
    - In-memory tar parsing (no seeks)
    - Only JPEG-decode sampled frames
    """

    def __init__(
        self,
        folder_path: str | Path,
        *,
        target_length: int | None = None,
        transform: Callable | None = None,
        max_attempts: int = 8,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        allow_variable_step: bool = True,
        sample_fps: float | None = None,
        cache_name: str = "tarvidds",
        decode_device: str = "cpu",
        max_gap_sec: float | None = None,
    ):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"{self.folder_path} does not exist")

        self.target_length = target_length
        self.transform = transform
        self.max_attempts = max_attempts
        self.dtype = dtype
        self.allow_variable_step = allow_variable_step
        self.sample_fps = sample_fps
        self.decode_device = decode_device
        self.max_gap_sec = max_gap_sec
        self._rng = random.Random(seed)

        # Build or load file list + metadata cache
        self._files_cache = self.folder_path / f"{cache_name}.files.npy"
        self._meta_cache = self.folder_path / f"{cache_name}.meta.npy"

        self._build_cache()
        self._build_valid_indices()

        if len(self._valid_indices) == 0:
            raise ValueError(f"No valid tar files found under {self.folder_path}")

        print(f"TarVideoDataset: {len(self._valid_indices)} valid videos from {self.folder_path}")

    def _build_cache(self):
        if self._files_cache.exists() and self._meta_cache.exists():
            self._all_files = np.load(self._files_cache, allow_pickle=True)
            self._all_meta = np.load(self._meta_cache, allow_pickle=True)
            if len(self._all_files) == len(self._all_meta):
                return

        print(f"Scanning tar files in {self.folder_path}...")
        tar_files = sorted(str(p) for p in self.folder_path.rglob("*.tar") if p.is_file())
        print(f"Found {len(tar_files)} tar files")

        # Read metadata from each tar (just meta.json, tiny)
        meta_list = []
        for tf in tar_files:
            try:
                m = self._read_meta_only(tf)
                meta_list.append(m)
            except Exception:
                meta_list.append({"frame_count": 0, "fps": 0.0})

        self._all_files = np.array(tar_files, dtype=object)
        self._all_meta = np.array(meta_list, dtype=object)
        np.save(self._files_cache, self._all_files)
        np.save(self._meta_cache, self._all_meta)

    @staticmethod
    def _read_meta_only(tar_path: str) -> dict:
        """Read just meta.json from a tar file."""
        with open(tar_path, "rb") as f:
            tar_bytes = f.read()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            meta_member = tar.getmember("meta.json")
            meta_f = tar.extractfile(meta_member)
            return json.loads(meta_f.read())

    def _build_valid_indices(self):
        counts = np.array([m.get("frame_count", 0) for m in self._all_meta])
        fps_arr = np.array([m.get("fps", 30.0) for m in self._all_meta])

        if self.sample_fps is not None and self.target_length is not None:
            required_duration = self.target_length / self.sample_fps
            min_frames = required_duration * fps_arr
            self._valid_indices = np.where((counts >= min_frames) & (fps_arr > 0))[0]
        elif self.target_length is not None:
            self._valid_indices = np.where(counts >= self.target_length)[0]
        else:
            self._valid_indices = np.where(counts > 0)[0]

        self._counts = counts
        self._fps_arr = fps_arr

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        attempts = 0
        index = idx % len(self._valid_indices)

        while attempts < self.max_attempts:
            file_idx = self._valid_indices[index]
            tar_path = str(self._all_files[file_idx])
            frame_count = int(self._counts[file_idx])
            fps = float(self._fps_arr[file_idx])
            try:
                return self._load_from_tar(tar_path, frame_count, fps)
            except Exception:
                attempts += 1
                index = self._rng.randint(0, len(self._valid_indices) - 1)

        raise RuntimeError(f"Failed after {self.max_attempts} attempts")

    def _load_from_tar(self, tar_path: str, frame_count: int, fps: float) -> torch.Tensor:
        # Step 1: Single sequential read of entire tar
        with open(tar_path, "rb") as f:
            tar_bytes = f.read()

        # Step 2: Determine which frames to sample
        if self.target_length is not None:
            if self.sample_fps is not None:
                indices = self._sample_fixed_fps(frame_count, fps, self.target_length, self.sample_fps)
            else:
                max_step = None
                if self.max_gap_sec is not None and fps > 0:
                    max_step = max(1, int(fps * self.max_gap_sec))
                start, step = self._sample_range(frame_count, self.target_length, max_step=max_step)
                indices = list(range(start, start + self.target_length * step, step))
        else:
            indices = list(range(frame_count))

        needed = set(indices)

        # Step 3: Parse tar in memory, extract only needed JPEG bytes
        frame_bytes = {}
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".jpg"):
                    continue
                frame_idx = int(member.name.split(".")[0])
                if frame_idx in needed:
                    f = tar.extractfile(member)
                    frame_bytes[frame_idx] = f.read()

        # Step 4: JPEG decode only sampled frames via torchvision.io.decode_jpeg
        from torchvision.io import decode_jpeg

        # Collect JPEG byte tensors in order
        jpg_tensors = []
        for i in indices:
            raw = frame_bytes.get(i)
            if raw is None:
                nearest = min(frame_bytes.keys(), key=lambda k: abs(k - i))
                raw = frame_bytes[nearest]
            jpg_tensors.append(torch.frombuffer(bytearray(raw), dtype=torch.uint8))

        # Batched decode — decode_jpeg accepts a list for GPU batched decode
        if self.decode_device != "cpu":
            decoded = decode_jpeg(jpg_tensors, device=self.decode_device)
            tensor = torch.stack(decoded).float() / 255.0  # (T, C, H, W) on GPU
            tensor = tensor.cpu()
        else:
            decoded = [decode_jpeg(j) for j in jpg_tensors]
            tensor = torch.stack(decoded).to(self.dtype) / 255.0  # (T, C, H, W)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def _sample_fixed_fps(self, frame_count, video_fps, target_length, sample_fps):
        step = video_fps / sample_fps
        total_span = (target_length - 1) * step
        max_start = frame_count - 1 - total_span
        if max_start < 0:
            max_start = 0
            step = (frame_count - 1) / max(target_length - 1, 1)
        start = self._rng.uniform(0, max(0, max_start))
        indices = [min(max(0, int(round(start + i * step))), frame_count - 1) for i in range(target_length)]
        return indices

    def _sample_range(self, frame_count, target_length, max_step=None):
        if not self.allow_variable_step or frame_count < 2 * target_length:
            max_offset = frame_count - target_length
            start = self._rng.randint(0, max(0, max_offset))
            return start, 1
        computed_max = frame_count // target_length
        if max_step is not None:
            computed_max = min(computed_max, max_step)
        computed_max = max(1, computed_max)
        step = self._rng.randint(1, computed_max)
        required = target_length * step
        max_offset = frame_count - required
        if max_offset < 0:
            step = 1
            max_offset = frame_count - target_length
        start = self._rng.randint(0, max(0, max_offset))
        return start, step

    @property
    def files(self):
        return self._all_files[self._valid_indices]

    def rebuild_cache(self):
        for f in [self._files_cache, self._meta_cache]:
            if f.exists():
                f.unlink()
        self._build_cache()
        self._build_valid_indices()
