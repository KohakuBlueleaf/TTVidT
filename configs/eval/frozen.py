"""Frozen-feature extraction for the benchmarks in the paper (Tables 1, 2, 4).

8 frames sampled uniformly over each clip, resized to 256x256, pixels in [-1, 1];
features are the per-frame motion tokens of the frozen encoder.

Build a config on top of this one that sets the checkpoint::

    from kohakuengine import use_config

    use_config("frozen.py")
    CHECKPOINT_PATH = "ttvidt/<run_id>/checkpoints/epoch=7.ckpt"   # or exported model dir / HF repo id
    MODEL_NAME = "my_model"

    kogine run scripts/eval/extract_features.py -c configs/eval/my_model.py
"""

CHECKPOINT_PATH = None
DATASETS = "hmdb51,arid,iard,jester,sthsthv2,diving48,ek100_verb"
DATA_ROOT = "eval-dataset"
DATASET_FORMAT = "tar"
TARGET_LENGTH = 8
TARGET_FPS = None
TARGET_DURATION = None
SAMPLING_MODE = "uniform"
RESIZE = (256, 256)
BATCH_SIZE = 64
NUM_WORKERS = 16
FEATURE_DIR = "features"
HMDB_SPLIT_ID = 1
ARID_SPLIT_ID = 0
IARD_SPLIT_BY = "actor"      # held-out actors (train_ratio of actors for training)
IARD_TRAIN_RATIO = 0.8
