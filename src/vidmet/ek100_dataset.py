"""
EK100 Dataset for action recognition / anticipation evaluation.

Supports two layouts:
  1. Flat split dirs: ek100-tar/{train,val}/<narration_id>.tar
     Labels read from annotation CSVs. label_type selects verb or noun.
  2. Class-organized: ek100_verb-tar/<verb_class>/<narration_id>.tar

Usage:
    from vidmet.ek100_dataset import EK100AnticipationDataset
    ds = EK100AnticipationDataset(root="eval-dataset/ek100-tar", split="train", label_type="verb")
"""
import csv
import json
from pathlib import Path

from vidmet.datasets import BaseVideoDataset


class EK100AnticipationDataset(BaseVideoDataset):
    """
    EPIC-Kitchens-100 dataset with verb/noun label modes.

    Flat layout (preferred):
        <root>/
            train/          <narration_id>.tar
            val/            <narration_id>.tar
            annotations/    EPIC_100_train.csv, EPIC_100_validation.csv

    Class-organized layout:
        <root>/
            <class_id>/
                <narration_id>.tar
    """

    # Action label encoding for "action" mode: action_id = verb_class * 1000 + noun_class
    # Then we re-map to a contiguous index built at load time.
    NUM_VERB_CLASSES = 97
    NUM_NOUN_CLASSES = 300

    def __init__(self, label_type: str = "verb", ann_dir: str | Path | None = None, **kwargs):
        assert label_type in ("verb", "noun", "action")
        self.label_type = label_type
        self.ann_dir = Path(ann_dir) if ann_dir else None
        self._action_to_idx: dict[int, int] = {}
        super().__init__(**kwargs)

    def _load_annotations(self):
        if self.label_type == "verb":
            num_classes = self.NUM_VERB_CLASSES
        elif self.label_type == "noun":
            num_classes = self.NUM_NOUN_CLASSES
        else:
            # "action" classes are built dynamically from observed (verb,noun) pairs
            num_classes = self.NUM_VERB_CLASSES * self.NUM_NOUN_CLASSES  # upper bound
        self.classes = [str(i) for i in range(num_classes)]
        self.label_to_idx = {name: int(name) for name in self.classes}

        # Detect layout
        split_dir = self.root / ("val" if self.split == "validation" else self.split)
        if split_dir.is_dir() and any(split_dir.glob("*.tar")):
            # Flat layout: root/{train,val}/*.tar — need annotations for labels
            self._load_flat(split_dir)
        else:
            # Class-organized: root/<class_id>/*.tar
            self._load_class_organized()

    def _find_ann_dir(self) -> Path | None:
        if self.ann_dir and (self.ann_dir / "EPIC_100_train.csv").exists():
            return self.ann_dir
        for candidate in [
            self.root / "annotations",
            self.root.parent / "ek100" / "annotations",
            self.root.parent / "ek100-tar" / "annotations",
        ]:
            if (candidate / "EPIC_100_train.csv").exists():
                return candidate
        return None

    def _load_flat(self, split_dir: Path):
        """Load from flat split dir with labels from CSV annotations."""
        ann_dir = self._find_ann_dir()
        if ann_dir is None:
            # Try reading labels from meta.json inside each tar
            self._load_flat_from_meta(split_dir)
            return

        split_map = {"train": "EPIC_100_train.csv", "validation": "EPIC_100_validation.csv"}
        csv_file = split_map.get(self.split)
        if csv_file is None:
            raise ValueError(f"Unknown split: {self.split}. Use 'train' or 'validation'.")

        # Build narration_id -> label mapping
        nid_to_label = {}
        with open(ann_dir / csv_file) as f:
            for row in csv.DictReader(f):
                nid = row["narration_id"]
                if self.label_type == "verb":
                    nid_to_label[nid] = int(row["verb_class"])
                elif self.label_type == "noun":
                    nid_to_label[nid] = int(row["noun_class"])
                else:
                    nid_to_label[nid] = int(row["verb_class"]) * 1000 + int(row["noun_class"])

        # For "action" mode, remap raw verb*1000+noun ids to a contiguous index
        if self.label_type == "action":
            unique = sorted(set(nid_to_label.values()))
            self._action_to_idx = {a: i for i, a in enumerate(unique)}
            nid_to_label = {nid: self._action_to_idx[v] for nid, v in nid_to_label.items()}

        for tar_path in sorted(split_dir.glob("*.tar")):
            nid = tar_path.stem
            if nid in nid_to_label:
                self.samples.append((tar_path, nid_to_label[nid]))

    def _load_flat_from_meta(self, split_dir: Path):
        """Load labels from meta.json embedded in each tar (fallback)."""
        import tarfile, io
        label_key = "verb_class" if self.label_type == "verb" else "noun_class"

        for tar_path in sorted(split_dir.glob("*.tar")):
            try:
                with tarfile.open(str(tar_path), "r") as tar:
                    meta_member = tar.getmember("meta.json")
                    meta = json.load(tar.extractfile(meta_member))
                    if label_key in meta:
                        self.samples.append((tar_path, meta[label_key]))
            except (KeyError, json.JSONDecodeError):
                continue

    def _load_class_organized(self):
        """Load from class-organized dirs: root/<class_id>/*.tar"""
        ann_dir = self._find_ann_dir()

        class_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

        if not class_dirs:
            return

        # If we have annotations, filter by split
        split_nids = None
        if ann_dir:
            split_map = {"train": "EPIC_100_train.csv", "validation": "EPIC_100_validation.csv"}
            csv_file = split_map.get(self.split)
            if csv_file and (ann_dir / csv_file).exists():
                split_nids = set()
                with open(ann_dir / csv_file) as f:
                    for row in csv.DictReader(f):
                        split_nids.add(row["narration_id"])

        for class_dir in class_dirs:
            label = int(class_dir.name)
            for tar_path in sorted(class_dir.glob("*.tar")):
                if split_nids is None or tar_path.stem in split_nids:
                    self.samples.append((tar_path, label))
