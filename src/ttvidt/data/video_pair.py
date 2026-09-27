"""Video pair dataset for frame prediction pretraining.

Wraps TarVideoDataset to get frame pairs (T, T+dT) with known temporal gap.
Returns (frame_T, frame_TdT, dt_seconds) where dt_seconds is the actual
temporal distance in seconds between the two frames.

dT is sampled uniformly from [min_gap_sec, max_gap_sec].
"""

import random

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms as trns

from ttvidt.data.tar_video import TarVideoDataset


class VideoPairDataset(data.Dataset):
    """Load random frame pairs from tar-JPEG video files with known dT.

    Returns (frame_T, frame_TdT, dt_seconds).
    """

    def __init__(
        self,
        folder_paths,
        transform=None,
        image_size=256,
        min_gap_sec=1 / 6,
        max_gap_sec=0.5,
    ):
        self.image_size = image_size
        self.transform = transform
        self.min_gap_sec = min_gap_sec
        self.max_gap_sec = max_gap_sec
        self._rng = random.Random()

        # Load datasets — target_length=None so we control frame sampling ourselves
        # But we still need the metadata (fps, frame_count) for gap calculation
        # Use target_length=2 with max_gap_sec to filter out too-short videos
        self._datasets = []
        for folder in folder_paths:
            ds = TarVideoDataset(
                folder,
                target_length=2,
                transform=None,
                max_gap_sec=max_gap_sec,
            )
            self._datasets.append(ds)

        self._concat = data.ConcatDataset(self._datasets)
        print(f"[VideoPair] Total: {len(self._concat)} valid videos, "
              f"dT range: [{min_gap_sec:.3f}, {max_gap_sec:.3f}] sec")

    def __len__(self):
        return len(self._concat)

    def __getitem__(self, idx):
        for attempt in range(8):
            try:
                return self._load_pair(idx if attempt == 0 else self._rng.randint(0, len(self._concat) - 1))
            except Exception:
                pass
        S = self.image_size
        return torch.zeros(3, S, S), torch.zeros(3, S, S), torch.tensor(self.min_gap_sec)

    def _load_pair(self, idx):
        # Resolve which sub-dataset and local index
        ds_idx = idx
        for i, ds in enumerate(self._datasets):
            if ds_idx < len(ds):
                tar_ds = ds
                local_idx = ds_idx
                break
            ds_idx -= len(ds)
        else:
            raise IndexError(f"Index {idx} out of range")

        # Get video metadata
        file_idx = tar_ds._valid_indices[local_idx % len(tar_ds._valid_indices)]
        fps = float(tar_ds._fps_arr[file_idx])
        frame_count = int(tar_ds._counts[file_idx])

        # Sample dT uniformly in [min_gap, max_gap] seconds
        dt_sec = self._rng.uniform(self.min_gap_sec, self.max_gap_sec)
        frame_gap = max(1, round(dt_sec * fps))

        # Clamp frame_gap to valid range
        frame_gap = min(frame_gap, frame_count - 1)
        # Actual dT after clamping
        actual_dt = frame_gap / fps

        # Sample start frame
        max_start = frame_count - 1 - frame_gap
        start = self._rng.randint(0, max(0, max_start))

        # Load the two specific frames from the tar
        import io
        import tarfile
        tar_path = str(tar_ds._all_files[file_idx])

        with open(tar_path, "rb") as f:
            tar_bytes = f.read()

        needed = {start, start + frame_gap}
        frame_bytes = {}
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".jpg"):
                    continue
                fidx = int(member.name.split(".")[0])
                if fidx in needed:
                    frame_bytes[fidx] = tar.extractfile(member).read()

        from torchvision.io import decode_jpeg, decode_image

        def decode(raw):
            buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            try:
                img = decode_jpeg(buf)
            except RuntimeError:
                img = decode_image(buf)
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif img.shape[0] == 4:
                img = img[:3]
            return img.float() / 255.0

        img_T = decode(frame_bytes[start])
        img_TdT = decode(frame_bytes[start + frame_gap])

        if self.transform:
            S = self.image_size
            img_T = TF.resize(img_T, S, antialias=True)
            img_TdT = TF.resize(img_TdT, S, antialias=True)

            i, j, h, w = trns.RandomCrop.get_params(img_T, (S, S))
            img_T = TF.crop(img_T, i, j, h, w)
            img_TdT = TF.crop(img_TdT, i, j, h, w)

            if self._rng.random() < 0.5:
                img_T = TF.hflip(img_T)
                img_TdT = TF.hflip(img_TdT)

            img_T = TF.normalize(img_T, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            img_TdT = TF.normalize(img_TdT, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        return img_T, img_TdT, torch.tensor(actual_dt, dtype=torch.float32)
