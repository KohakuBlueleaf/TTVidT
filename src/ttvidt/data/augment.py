"""Train-time video transforms for encoder pretraining.

Two pipelines are used in the paper:

* **Default** (every sweep entry): resize to 256, random 256x256 crop, random
  horizontal and vertical flip, normalise to [-1, 1]. Encoder input and
  reconstruction target are the same view.
* **Dual augmentation** (DisMo's native recipe, the "DisMo with dual
  augmentation" row): the encoder sees an aggressively augmented view and the
  decoder reconstructs a mildly augmented view of the same clip. Spatial
  parameters are shared across all frames of a clip so the augmentation never
  introduces motion; no flips, since a flip reverses motion direction.
"""

from torchvision.transforms import transforms as trns

NORMALIZE = trns.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


class TemporalConsistentTransform:
    """Apply spatial transforms with one set of random parameters per clip.

    Frames ``[T, C, H, W]`` are reshaped to ``[1, T*C, H, W]`` for the spatial
    ops so every frame receives the same crop / rotation, then reshaped back.
    Photometric transforms are applied afterwards.
    """

    def __init__(self, spatial_transforms, photometric_transforms=None, normalize=None):
        self.spatial = trns.Compose(spatial_transforms) if spatial_transforms else None
        self.photometric = trns.Compose(photometric_transforms) if photometric_transforms else None
        self.normalize = normalize

    def __call__(self, frames):
        T, C, H, W = frames.shape
        if self.spatial:
            out = self.spatial(frames.reshape(1, T * C, H, W))
            frames = out.reshape(T, C, out.shape[2], out.shape[3])
        if self.photometric:
            frames = self.photometric(frames)
        if self.normalize:
            frames = self.normalize(frames)
        return frames

    def __repr__(self):
        return (f"{type(self).__name__}(spatial={self.spatial}, "
                f"photometric={self.photometric}, normalize={self.normalize})")


def default_transform(size: int = 256):
    """Pipeline used by every entry of the architecture-objective sweep."""
    return trns.Compose([
        trns.Resize(size),
        trns.RandomCrop(size),
        trns.RandomHorizontalFlip(),
        trns.RandomVerticalFlip(),
        NORMALIZE,
    ])


def dual_aug_transforms(size: int = 256):
    """(encoder_transform, decoder_transform) for DisMo-style dual augmentation."""
    encoder_transform = TemporalConsistentTransform(
        spatial_transforms=[
            trns.RandomResizedCrop(size, scale=(0.25, 1.0), ratio=(0.666, 1.5)),
            trns.RandomAffine(degrees=30, translate=None, scale=None),
        ],
        photometric_transforms=[trns.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)],
        normalize=NORMALIZE,
    )
    decoder_transform = TemporalConsistentTransform(
        spatial_transforms=[trns.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1))],
        photometric_transforms=[trns.ColorJitter(brightness=0.2, saturation=0.2)],
        normalize=NORMALIZE,
    )
    return encoder_transform, decoder_transform
