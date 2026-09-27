"""EPIC-Kitchens-100 verb anticipation (Table 4, "EK-V Antic.").

Clips are the untrimmed observation windows ending 1 s before each action
(``scripts/data/benchmarks/setup_ek100_anticipation.sh``); 8 frames at 6 fps,
centered. Probe the features with ``scripts/eval/probe.py --datasets ek100_verb_anticip``.
"""

from ttvidt.config import Config

CHECKPOINT_PATH = None
DATASETS = "ek100_verb_anticip"
DATA_ROOT = "eval-dataset"
DATASET_FORMAT = "tar"
TARGET_LENGTH = 8
TARGET_FPS = 6.0
TARGET_DURATION = None
SAMPLING_MODE = "center"
RESIZE = (256, 256)
BATCH_SIZE = 64
NUM_WORKERS = 16
FEATURE_DIR = "features"


def config_gen():
    return Config.from_globals()
