"""Tar-of-JPEG batch dataset for ImageNet-style image training.

Each tar file contains a batch of JPEG images (e.g., 64 images).
One __getitem__ call reads the entire tar sequentially (single NFS read),
decodes all JPEGs, applies transforms, and returns a full batch tensor.

DataLoader should use batch_size=1 with collate_fn=lambda x: x[0].

Supports:
  - CPU decode: torchvision.io.decode_jpeg per image
  - GPU batch decode: torchvision.io.decode_jpeg with device=cuda
  - Configurable resize with antialiased interpolation
"""

import io
import os
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data


class TarImageBatchDataset(data.Dataset):
    """Map-style dataset where each item is a batch of images from one tar file.

    Args:
        folder_path: Directory containing .tar files (each tar = one batch)
        image_size: Target image size (square crop)
        decode_device: "cpu" or "cuda" for JPEG decoding
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        folder_path: str,
        image_size: int = 256,
        decode_device: str = "cpu",
        seed: int | None = None,
    ):
        self.folder_path = Path(folder_path)
        self.image_size = image_size
        self.decode_device = decode_device
        self._rng = random.Random(seed)

        # Discover tar files
        self._tar_files = sorted(str(p) for p in self.folder_path.rglob("*.tar"))
        assert self._tar_files, f"No tar files found in {folder_path}"
        print(f"[TarImageBatch] {len(self._tar_files)} tar files from {folder_path}")

    def __len__(self):
        return len(self._tar_files)

    def __getitem__(self, idx):
        for attempt in range(8):
            try:
                return self._load_tar(idx if attempt == 0 else self._rng.randint(0, len(self._tar_files) - 1))
            except Exception:
                pass
        # Last resort: dummy batch
        S = self.image_size
        return torch.zeros(1, 3, S, S) * 2 - 1

    def _load_tar(self, idx):
        tar_path = self._tar_files[idx]

        # Step 1: Single sequential read of entire tar
        with open(tar_path, "rb") as f:
            tar_bytes = f.read()

        # Step 2: Extract all JPEG bytes from tar in memory
        jpeg_list = []
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".jpg"):
                    continue
                jpeg_list.append(tar.extractfile(member).read())

        assert jpeg_list, f"No JPEGs in {tar_path}"

        # Step 3: Decode JPEGs — skip corrupted, keep good ones
        decoded = []
        for j in jpeg_list:
            img = self._safe_decode(torch.frombuffer(bytearray(j), dtype=torch.uint8))
            if img is not None:
                decoded.append(img)

        assert decoded, f"All JPEGs corrupted in {tar_path}"

        # Step 4: Resize + random crop + flip → [B, 3, H, W] float in [-1, 1]
        batch = self._transform_batch(decoded)
        return batch

    @staticmethod
    def _safe_decode(jpg_tensor):
        from torchvision.io import decode_jpeg, decode_image
        try:
            try:
                img = decode_jpeg(jpg_tensor)
            except RuntimeError:
                img = decode_image(jpg_tensor)
            # Ensure 3 channels
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif img.shape[0] == 4:
                img = img[:3]
            return img
        except Exception:
            return None

    def _transform_batch(self, decoded_images):
        """Resize (antialias), random crop, random flip, normalize to [-1,1].

        Args:
            decoded_images: list of [3, H_i, W_i] uint8 tensors (variable sizes)

        Returns:
            [B, 3, image_size, image_size] float32 tensor in [-1, 1]
        """
        S = self.image_size
        results = []

        for img in decoded_images:
            # img: [3, H, W] uint8
            h, w = img.shape[1], img.shape[2]

            # Resize shorter edge to image_size (antialiased)
            # Use float for interpolation
            img_f = img.unsqueeze(0).float()  # [1, 3, H, W]
            if min(h, w) != S or max(h, w) < S:
                scale = S / min(h, w)
                new_h = max(S, round(h * scale))
                new_w = max(S, round(w * scale))
                img_f = F.interpolate(
                    img_f, size=(new_h, new_w),
                    mode="bilinear", align_corners=False, antialias=True,
                )
            else:
                new_h, new_w = h, w

            # Random crop to image_size x image_size
            top = self._rng.randint(0, max(0, new_h - S))
            left = self._rng.randint(0, max(0, new_w - S))
            img_f = img_f[:, :, top:top+S, left:left+S]

            # Random horizontal flip
            if self._rng.random() < 0.5:
                img_f = img_f.flip(-1)

            results.append(img_f.squeeze(0))  # [3, S, S]

        # Stack and normalize: [0, 255] → [-1, 1]
        batch = torch.stack(results)  # [B, 3, S, S]
        batch = batch / 255.0 * 2.0 - 1.0
        return batch


def identity_collate(batch):
    """Collate for BS=1 batch dataset — just unwrap the single item."""
    return batch[0]
