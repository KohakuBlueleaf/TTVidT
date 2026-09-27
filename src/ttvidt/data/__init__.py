"""Pretraining data pipeline.

- ``tar_video``  ``TarVideoDataset``: videos stored as one tar of JPEG frames each
                 (the format all pretraining and benchmark data is converted to)
- ``video``      ``FolderVideoDataset`` for raw video files (torchcodec backend),
                 plus ``save_video`` used for logging reconstructions
- ``augment``    train-time transforms, including DisMo-style dual augmentation
- ``dual_aug``   ``DualAugDataset``: separate encoder / decoder views of one clip
- ``tar_image``  ImageNet tar shards for decoder pretraining
- ``video_pair`` frame-pair sampling for video decoder pretraining
"""
