"""
Dual augmentation dataset wrapper for TT-VidT.

Wraps any video dataset to support separate encoder/decoder augmentations.
Both transforms operate in pixel space [T, C, H, W] before any AE encoding.

Usage:
    # Single augmentation (default, current behavior)
    dataset = TarVideoDataset(folder, transform=transform, ...)

    # Dual augmentation (opt-in)
    dataset = DualAugDataset(
        TarVideoDataset(folder, transform=None, ...),  # no transform on base dataset
        encoder_transform=aggressive_transform,
        decoder_transform=mild_transform,
    )
    # Returns (enc_video, dec_video) tuple instead of single tensor

    # Single augmentation via wrapper (equivalent to base dataset with transform)
    dataset = DualAugDataset(
        TarVideoDataset(folder, transform=None, ...),
        encoder_transform=transform,
        decoder_transform=None,  # falls back to encoder_transform
    )
    # Returns single tensor (not tuple)
"""

import torch
from torch.utils.data import Dataset


class DualAugDataset(Dataset):
    """
    Wraps a video dataset to apply separate transforms for encoder and decoder.

    When decoder_transform is None, behaves identically to the base dataset
    with encoder_transform applied — returns a single tensor.

    When decoder_transform is set, returns a tuple (enc_video, dec_video)
    where both are different augmentations of the same raw video.

    The base dataset MUST return raw (untransformed) video tensors.
    Set transform=None on the base dataset.

    Args:
        base_dataset: Video dataset that returns [T, C, H, W] tensors (no transform)
        encoder_transform: Transform for encoder input (aggressive augmentation)
        decoder_transform: Transform for decoder target/source (mild augmentation).
                          If None, falls back to encoder_transform (single-aug mode).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        encoder_transform=None,
        decoder_transform=None,
    ):
        self.base_dataset = base_dataset
        self.encoder_transform = encoder_transform
        self.decoder_transform = decoder_transform
        self._dual_mode = decoder_transform is not None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        raw_video = self.base_dataset[idx]  # [T, C, H, W] raw pixels

        if self._dual_mode:
            # Dual augmentation: two different transforms on same raw frames
            enc_video = self.encoder_transform(raw_video) if self.encoder_transform else raw_video
            dec_video = self.decoder_transform(raw_video)
            return enc_video, dec_video
        else:
            # Single augmentation (default behavior)
            if self.encoder_transform is not None:
                raw_video = self.encoder_transform(raw_video)
            return raw_video
