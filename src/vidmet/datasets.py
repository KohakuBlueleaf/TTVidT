"""
Dataset loaders for video motion benchmarks.

Assumes data is pre-downloaded and converted; see docs/data.md.
Tar-of-JPEG clips need no video decoder; raw video files use ttvidt.data.video.
"""

import io
import json
import logging
import random
import tarfile
import warnings
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from ttvidt.data.video import Backend, _resolve_backend

logger = logging.getLogger(__name__)

# Sampling modes for video loading
SamplingMode = Literal["center", "uniform", "fixed_fps", "random_duration"]


class VideoLoadError(Exception):
    """Raised when a video fails to load or has invalid dimensions."""

    pass


class BaseVideoDataset(Dataset):
    """Base class for video classification datasets.

    Uses TorchCodec for efficient video decoding.

    Supports multiple sampling modes:
    - "center": Crop from center (for longer videos), pad last frame (for shorter)
    - "uniform": Sample uniformly spaced frames across the video
    - "fixed_fps": Sample at a fixed FPS (e.g., 8 FPS), with random start position
    - "random_duration": Random start position with variable step (for training augmentation)

    This ensures all videos have exactly target_length frames for batching.
    Padding is safe since TT-VidT uses causal attention.
    """

    # Maximum number of fallback attempts to prevent infinite recursion
    MAX_FALLBACK_ATTEMPTS = 10

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        target_length: int | None = None,
        num_clips: int = 1,
        # Sampling options
        sampling_mode: SamplingMode = "center",
        sample_fps: float | None = None,  # Used when sampling_mode="fixed_fps"
        seed: int | None = None,  # For reproducible random sampling
        # FPS + Duration mode (alternative to target_length)
        target_fps: float | None = None,  # Target sampling FPS (e.g., 8.0)
        target_duration: float | None = None,  # Target duration in seconds (e.g., 1.0)
        # Video backend
        backend: Backend = "auto",
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.num_clips = num_clips
        self._rng = random.Random(seed)
        try:
            self._backend = _resolve_backend(backend)
        except ImportError:
            self._backend = None  # tar-only mode, no video backend needed
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        # Track failed samples to avoid repeated attempts
        self._failed_indices: set[int] = set()

        # Handle different sampling modes based on which parameters are set
        if target_fps is not None and target_duration is not None:
            # Mode 2: FPS + Duration → calculate frame count
            self.target_length = int(target_fps * target_duration)
            self.sampling_mode = "fixed_fps"
            self.sample_fps = target_fps
            logger.info(
                f"FPS+Duration mode: {target_fps} FPS × {target_duration}s = "
                f"{self.target_length} frames"
            )
        elif target_fps is not None and target_length is not None:
            # Mode 3: FPS + Frame count → duration is implicit
            self.target_length = target_length
            self.sampling_mode = "fixed_fps"
            self.sample_fps = target_fps
            implicit_duration = target_length / target_fps
            logger.info(
                f"FPS+Length mode: {target_fps} FPS × {target_length} frames = "
                f"{implicit_duration:.3f}s duration"
            )
        else:
            # Mode 1: Traditional mode - use target_length directly
            self.target_length = target_length
            self.sampling_mode = sampling_mode
            self.sample_fps = sample_fps

        self.target_fps = target_fps
        self.target_duration = target_duration

        self._load_annotations()

    def _load_annotations(self):
        """Override in subclass to load dataset-specific annotations."""
        raise NotImplementedError

    def _validate_tensor(self, tensor: torch.Tensor, path: Path) -> None:
        """Validate that the loaded tensor has correct dimensions.

        Args:
            tensor: Video tensor with shape [T, C, H, W]
            path: Path to the video file (for error messages)

        Raises:
            VideoLoadError: If tensor has invalid dimensions
        """
        if tensor.ndim != 4:
            raise VideoLoadError(
                f"Invalid tensor dimensions: expected 4D [T, C, H, W], "
                f"got {tensor.ndim}D with shape {tensor.shape} for {path}"
            )

        T, C, H, W = tensor.shape

        if T == 0:
            raise VideoLoadError(f"Empty video (0 frames) for {path}")

        if C != 3:
            raise VideoLoadError(
                f"Invalid channel count: expected 3, got {C} for {path}"
            )

        if H == 0 or W == 0:
            raise VideoLoadError(f"Invalid spatial dimensions: {H}x{W} for {path}")

        if self.target_length is not None and T != self.target_length:
            raise VideoLoadError(
                f"Frame count mismatch: expected {self.target_length}, "
                f"got {T} for {path}"
            )

        if not torch.isfinite(tensor).all():
            raise VideoLoadError(f"Tensor contains NaN or Inf values for {path}")

    def _load_video(self, path: Path) -> torch.Tensor:
        """Load video frames from path.

        Supports video files (torchcodec/kohakuclip) and tar-of-JPEG files.
        Always returns exactly target_length frames for batching.

        Raises:
            VideoLoadError: If video cannot be loaded or has invalid dimensions
        """
        if path.suffix == ".tar":
            tensor = self._load_video_tar(path)
        elif self._backend == "torchcodec":
            tensor = self._load_video_torchcodec(path)
        elif self._backend == "kohakuclip":
            tensor = self._load_video_kohakuclip(path)
        else:
            raise VideoLoadError(
                f"No video backend available to load {path}. "
                "Install kohakuclip or torchcodec, or use tar-of-JPEG format."
            )

        # Ensure exactly target_length frames (pad if needed)
        if self.target_length is not None:
            if tensor.shape[0] < self.target_length:
                pad_count = self.target_length - tensor.shape[0]
                padding = tensor[-1:].repeat(pad_count, 1, 1, 1)
                tensor = torch.cat([tensor, padding], dim=0)
            elif tensor.shape[0] > self.target_length:
                tensor = tensor[: self.target_length]

        return tensor

    def _load_video_torchcodec(self, path: Path) -> torch.Tensor:
        from torchcodec.decoders import VideoDecoder

        try:
            # Read entire file into memory first — converts many small random
            # NFS reads (seeks during demux/decode) into a single sequential
            # read. Typical video files are 100-500KB, so memory cost is negligible.
            with open(path, "rb") as f:
                video_bytes = f.read()

            decoder = VideoDecoder(
                video_bytes,
                seek_mode="approximate",
                dimension_order="NCHW",
                num_ffmpeg_threads=4,
            )
            frame_count = len(decoder)
            meta = decoder.metadata
            video_fps = meta.average_fps if meta.average_fps else 30.0
        except Exception as e:
            raise VideoLoadError(f"Failed to open video {path}: {e}") from e

        if frame_count == 0:
            raise VideoLoadError(f"Video has 0 frames: {path}")

        if self.target_length is not None and frame_count > 0:
            indices = self._get_frame_indices(frame_count, video_fps)
            indices = [min(max(0, i), frame_count - 1) for i in indices]
            try:
                frame_batch = decoder.get_frames_at(indices=indices)
                tensor = frame_batch.data
            except Exception as e:
                raise VideoLoadError(f"Failed to decode frames from {path}: {e}") from e
        else:
            try:
                frames = []
                for frame in decoder:
                    frames.append(frame)
                tensor = torch.stack(frames)
            except Exception as e:
                raise VideoLoadError(f"Failed to decode frames from {path}: {e}") from e

        if tensor.numel() == 0:
            raise VideoLoadError(f"Empty frame tensor from {path}")

        return tensor.float() / 255.0

    def _load_video_kohakuclip(self, path: Path) -> torch.Tensor:
        from kohakuclip import KClip

        try:
            clip = KClip(str(path))
            meta = clip.meta()
            frame_count = meta.frame_count or 0
            # Get fps via av
            video_fps = 30.0
            try:
                import av

                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    if stream.average_rate:
                        video_fps = float(stream.average_rate)
            except Exception:
                pass
        except Exception as e:
            raise VideoLoadError(f"Failed to open video {path}: {e}") from e

        if frame_count == 0:
            raise VideoLoadError(f"Video has 0 frames: {path}")

        if self.target_length is not None and frame_count > 0:
            indices = self._get_frame_indices(frame_count, video_fps)
            indices = [min(max(0, i), frame_count - 1) for i in indices]
            try:
                # Load covering range and pick indices
                min_i, max_i = min(indices), max(indices)
                frames = clip.range(start=min_i, end=max_i + 1).to_array()
                local_indices = [i - min_i for i in indices]
                frames = frames[local_indices]
            except Exception as e:
                raise VideoLoadError(f"Failed to decode frames from {path}: {e}") from e
        else:
            try:
                frames = clip.to_array()
            except Exception as e:
                raise VideoLoadError(f"Failed to decode frames from {path}: {e}") from e

        if frames.size == 0:
            raise VideoLoadError(f"Empty frame tensor from {path}")

        # NHWC uint8 numpy → NCHW float tensor
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float() / 255.0

    def _load_video_tar(self, path: Path) -> torch.Tensor:
        """Load video from a tar-of-JPEG file.

        Reads the entire tar into memory, parses meta.json for fps/frame_count,
        samples frame indices, and JPEG-decodes only the needed frames.
        """
        try:
            with open(path, "rb") as f:
                tar_bytes = f.read()
        except Exception as e:
            raise VideoLoadError(f"Failed to read tar file {path}: {e}") from e

        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
                # Read metadata
                meta_f = tar.extractfile("meta.json")
                meta = json.loads(meta_f.read())
                frame_count = meta["frame_count"]
                video_fps = meta.get("fps", 30.0)

                # Determine which frames to sample
                if self.target_length is not None and frame_count > 0:
                    indices = self._get_frame_indices(frame_count, video_fps)
                    indices = [min(max(0, i), frame_count - 1) for i in indices]
                else:
                    indices = list(range(frame_count))

                needed = set(indices)

                # Extract only needed JPEG bytes
                frame_bytes = {}
                for member in tar.getmembers():
                    if not member.name.endswith(".jpg"):
                        continue
                    frame_idx = int(member.name.split(".")[0])
                    if frame_idx in needed:
                        frame_bytes[frame_idx] = tar.extractfile(member).read()
        except Exception as e:
            if isinstance(e, VideoLoadError):
                raise
            raise VideoLoadError(f"Failed to parse tar {path}: {e}") from e

        if not frame_bytes:
            raise VideoLoadError(f"No valid frames in tar {path}")

        # JPEG decode
        from torchvision.io import decode_jpeg

        decoded = []
        for i in indices:
            raw = frame_bytes.get(i)
            if raw is None:
                nearest = min(frame_bytes.keys(), key=lambda k: abs(k - i))
                raw = frame_bytes[nearest]
            jpg_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            decoded.append(decode_jpeg(jpg_tensor))

        tensor = torch.stack(decoded).float() / 255.0  # [T, C, H, W]
        return tensor

    def _get_frame_indices(self, frame_count: int, video_fps: float) -> list[int]:
        """Get frame indices based on sampling mode.

        Args:
            frame_count: Total number of frames in video
            video_fps: Original video FPS

        Returns:
            List of frame indices to sample
        """
        target = self.target_length

        if self.sampling_mode == "center":
            # Center crop for longer videos
            if frame_count >= target:
                start = (frame_count - target) // 2
                return list(range(start, start + target))
            else:
                # Return all frames, will be padded later
                return list(range(frame_count))

        elif self.sampling_mode == "uniform":
            # Uniformly sample across the video
            if frame_count >= target:
                indices = np.linspace(0, frame_count - 1, target, dtype=int)
                return indices.tolist()
            else:
                return list(range(frame_count))

        elif self.sampling_mode == "fixed_fps":
            # Sample at fixed FPS
            if self.sample_fps is None:
                raise ValueError("sample_fps must be set when using fixed_fps mode")
            return self._sample_fixed_fps(frame_count, video_fps)

        elif self.sampling_mode == "random_duration":
            # Random start position with variable step (training augmentation)
            return self._sample_random_duration(frame_count)

        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

    def _sample_fixed_fps(self, frame_count: int, video_fps: float) -> list[int]:
        """Sample frames at a fixed FPS rate with random start."""
        target = self.target_length
        sample_fps = self.sample_fps

        # Calculate step in original video frames
        step = video_fps / sample_fps

        # Total span needed
        total_span = (target - 1) * step

        # Maximum valid start position
        max_start = frame_count - 1 - total_span
        if max_start < 0:
            # Video too short, use smaller step
            max_start = 0
            step = (frame_count - 1) / max(target - 1, 1)

        # Random start position
        start = self._rng.uniform(0, max(0, max_start))

        # Generate indices
        indices = [int(round(start + i * step)) for i in range(target)]
        return indices

    def _sample_random_duration(self, frame_count: int) -> list[int]:
        """Random start position with variable step (for training augmentation)."""
        target = self.target_length

        if frame_count < target:
            return list(range(frame_count))

        if frame_count < 2 * target:
            # Not enough for variable step, just random start
            max_offset = frame_count - target
            start = self._rng.randint(0, max_offset)
            return list(range(start, start + target))

        # Variable step sampling
        max_step = frame_count // target
        step = self._rng.randint(1, max_step)
        required_length = target * step
        max_offset = frame_count - required_length

        if max_offset < 0:
            step = 1
            max_offset = frame_count - target

        start = self._rng.randint(0, max_offset)
        return [start + i * step for i in range(target)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._getitem_with_fallback(idx, attempt=0)

    def _getitem_with_fallback(
        self, idx: int, attempt: int
    ) -> tuple[torch.Tensor, int]:
        """Load item with fallback mechanism for corrupted videos.

        Args:
            idx: Sample index
            attempt: Current fallback attempt number

        Returns:
            Tuple of (video_tensor, label)

        Raises:
            RuntimeError: If maximum fallback attempts exceeded
        """
        if attempt >= self.MAX_FALLBACK_ATTEMPTS:
            raise RuntimeError(
                f"Failed to load any valid sample after {self.MAX_FALLBACK_ATTEMPTS} "
                f"attempts. Last tried index: {idx}. "
                f"Total failed samples: {len(self._failed_indices)}"
            )

        path, label = self.samples[idx]

        try:
            video = self._load_video(path)

            if self.transform is not None:
                video = self.transform(video)

            # Validate final tensor dimensions
            self._validate_tensor(video, path)

            return video, label

        except (VideoLoadError, Exception) as e:
            # Log the failure
            if idx not in self._failed_indices:
                self._failed_indices.add(idx)
                logger.warning(
                    f"Failed to load video at index {idx} ({path}): {e}. "
                    f"Falling back to another sample."
                )

            # Select a random index that hasn't failed yet
            valid_indices = [
                i for i in range(len(self.samples)) if i not in self._failed_indices
            ]

            if not valid_indices:
                raise RuntimeError(
                    f"All {len(self.samples)} samples have failed to load. "
                    "Check dataset integrity."
                )

            new_idx = valid_indices[np.random.randint(0, len(valid_indices))]
            return self._getitem_with_fallback(new_idx, attempt + 1)


class SSv2Dataset(BaseVideoDataset):
    """
    Something-Something V2 dataset.

    Expected structure:
        root/
            videos/
                12345.webm
                ...
            labels/
                train.json
                validation.json
                labels.json
    """

    def _load_annotations(self):
        labels_file = self.root / "labels" / "labels.json"
        if labels_file.exists():
            with open(labels_file) as f:
                label_map = json.load(f)
            self.classes = list(label_map.keys())
            self.label_to_idx = {name: int(idx) for name, idx in label_map.items()}
        else:
            self.classes = []
            self.label_to_idx = {}

        split_name = "validation" if self.split == "val" else self.split
        split_file = self.root / "labels" / f"{split_name}.json"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            annotations = json.load(f)

        video_dir = self.root / "videos"
        for item in annotations:
            video_id = item["id"]
            template = item.get("template", item.get("label", ""))
            # Clean template: "[Something] [something]" -> "Something something"
            template = template.replace("[", "").replace("]", "")

            if template in self.label_to_idx:
                label = self.label_to_idx[template]
            else:
                continue

            # Check tar first (pre-converted), then webm (original)
            tar_path = video_dir / f"{video_id}.tar"
            video_path = video_dir / f"{video_id}.webm"
            if tar_path.exists():
                self.samples.append((tar_path, label))
            elif video_path.exists():
                self.samples.append((video_path, label))


class JesterDataset(BaseVideoDataset):
    """
    Jester gesture recognition dataset.

    Expected structure:
        root/
            videos/
                1/
                    00001.jpg, 00002.jpg, ...
                2/
                    ...
            jester-v1-train.csv
            jester-v1-validation.csv
            jester-v1-labels.csv
    """

    def _load_annotations(self):
        labels_file = self.root / "jester-v1-labels.csv"
        if labels_file.exists():
            with open(labels_file) as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.label_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        else:
            self.classes = []
            self.label_to_idx = {}

        split_file = self.root / f"jester-v1-{self.split}.csv"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            for line in f:
                parts = line.strip().split(";")
                if len(parts) >= 2:
                    video_id, label_name = parts[0], parts[1]
                    if label_name in self.label_to_idx:
                        # Check for tar file first, then frame directory
                        tar_path = self.root / "videos" / f"{video_id}.tar"
                        video_dir = self.root / "videos" / video_id
                        if tar_path.exists():
                            self.samples.append(
                                (tar_path, self.label_to_idx[label_name])
                            )
                        elif video_dir.exists():
                            self.samples.append(
                                (video_dir, self.label_to_idx[label_name])
                            )

    def _load_video(self, path: Path) -> torch.Tensor:
        """Load video from frame images or tar-of-JPEG.

        Always returns exactly target_length frames for batching.
        Supports multiple sampling modes based on self.sampling_mode.

        Raises:
            VideoLoadError: If video cannot be loaded or has invalid dimensions
        """
        # Dispatch tar files to base class tar loader
        if path.suffix == ".tar":
            return super()._load_video(path)

        from PIL import Image

        frame_files = sorted(path.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(path.glob("*.png"))

        frame_count = len(frame_files)

        if frame_count == 0:
            raise VideoLoadError(f"No frame images found in {path}")

        if self.target_length is not None:
            # Get frame indices based on sampling mode
            # For image sequences, assume ~12 FPS (typical for Jester)
            indices = self._get_frame_indices(frame_count, video_fps=12.0)
            # Clamp indices
            indices = [min(max(0, i), frame_count - 1) for i in indices]
            frame_files = [frame_files[i] for i in indices]

        frames = []
        first_frame_shape = None
        for f in frame_files:
            try:
                img = Image.open(f).convert("RGB")
                frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

                # Check for consistent frame shapes
                if first_frame_shape is None:
                    first_frame_shape = frame.shape
                elif frame.shape != first_frame_shape:
                    raise VideoLoadError(
                        f"Inconsistent frame shapes in {path}: "
                        f"first frame {first_frame_shape}, current frame {frame.shape}"
                    )

                frames.append(frame)
            except Exception as e:
                if isinstance(e, VideoLoadError):
                    raise
                raise VideoLoadError(f"Failed to load frame {f}: {e}") from e

        if not frames:
            raise VideoLoadError(f"No valid frames could be loaded from {path}")

        tensor = torch.stack(frames)

        # Ensure exactly target_length frames (pad if needed)
        if self.target_length is not None:
            if tensor.shape[0] < self.target_length:
                # Pad by repeating last frame
                pad_count = self.target_length - tensor.shape[0]
                padding = tensor[-1:].repeat(pad_count, 1, 1, 1)
                tensor = torch.cat([tensor, padding], dim=0)

        return tensor


class HMDB51Dataset(BaseVideoDataset):
    """
    HMDB51 action recognition dataset.

    Expected structure:
        root/
            videos/
                brush_hair/
                    video1.avi
                    ...
                cartwheel/
                    ...
            splits/
                brush_hair_test_split1.txt
                ...
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_id: int = 1,
        **kwargs,
    ):
        self.split_id = split_id
        super().__init__(root, split, **kwargs)

    def _load_annotations(self):
        video_dir = self.root / "videos"
        splits_dir = self.root / "splits"

        if not video_dir.exists():
            # Try alternate structure
            video_dir = self.root

        # a class is a directory with an official split file; this skips the
        # splits/ folder itself and any other helper directories
        self.classes = sorted(
            d.name
            for d in video_dir.iterdir()
            if d.is_dir() and (splits_dir / f"{d.name}_test_split{self.split_id}.txt").exists()
        )
        self.label_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Load split files
        for class_name in self.classes:
            split_file = splits_dir / f"{class_name}_test_split{self.split_id}.txt"
            if not split_file.exists():
                continue

            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        video_name, split_label = parts[0], int(parts[1])
                        # 1 = train, 2 = test, 0 = not used
                        if (self.split == "train" and split_label == 1) or (
                            self.split in ["test", "val"] and split_label == 2
                        ):
                            video_path = video_dir / class_name / video_name
                            # Also check for tar-of-JPEG version
                            tar_path = video_path.with_suffix(".tar")
                            if tar_path.exists():
                                self.samples.append(
                                    (tar_path, self.label_to_idx[class_name])
                                )
                            elif video_path.exists():
                                self.samples.append(
                                    (video_path, self.label_to_idx[class_name])
                                )


class IARDDataset(BaseVideoDataset):
    """
    IARD (Invariant Action Recognition Dataset) from CBMM/MIT.

    A controlled dataset for studying view/actor invariance in action recognition.
    5 actors × 5 actions × 5 views = controlled experimental setup.

    Expected structure:
        root/
            videos/
                drink/
                    drink_actor_background_view_variant_frame.avi
                eat/
                    ...
                jump/
                run/
                walk/
            labels.json (optional)

    Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DMT0PG
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        train_ratio: float = 0.8,
        split_by: str = "random",  # "random" or "actor" or "view"
        seed: int = 42,
        **kwargs,
    ):
        """
        Args:
            root: Dataset root directory
            split: "train" or "test"
            train_ratio: Ratio of data for training (when split_by="random")
            split_by: How to split data - "random", "actor", or "view"
            seed: Random seed for reproducible splits
        """
        self.train_ratio = train_ratio
        self.split_by = split_by
        self.seed = seed
        super().__init__(root, split, **kwargs)

    def _load_annotations(self):
        video_dir = self.root / "videos"

        # Main 5 action classes (excluding 'still' and 'plw' which are extras)
        self.classes = ["drink", "eat", "jump", "run", "walk"]
        self.label_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        all_samples = []
        for class_name in self.classes:
            class_dir = video_dir / class_name
            if not class_dir.exists():
                continue
            # Prefer tar files over avi; deduplicate by stem
            tar_stems = set()
            all_videos = []
            for video_path in class_dir.glob("*.tar"):
                tar_stems.add(video_path.stem)
                all_videos.append(video_path)
            for video_path in class_dir.glob("*.avi"):
                if video_path.stem not in tar_stems:
                    all_videos.append(video_path)
            for video_path in all_videos:
                # Parse filename: action_actor_background_view_variant_frame.avi/.tar
                # Example: drink_andrea_baker_0_2_0.avi
                parts = video_path.stem.split("_")
                if len(parts) >= 4:
                    actor = parts[1] if len(parts) > 1 else "unknown"
                    view = parts[3] if len(parts) > 3 else "0"
                else:
                    actor = "unknown"
                    view = "0"

                all_samples.append(
                    {
                        "path": video_path,
                        "label": self.label_to_idx[class_name],
                        "actor": actor,
                        "view": view,
                    }
                )

        # Split data
        import random

        rng = random.Random(self.seed)

        if self.split_by == "actor":
            # Split by actor for cross-actor generalization.
            # sorted(), NOT list(set(...)): set iteration order for str depends on
            # PYTHONHASHSEED (randomized per process), so the seeded shuffle below
            # was applied to a different ordering every run -> a different held-out
            # actor each extraction, making IARD incomparable across runs.
            actors = sorted(set(s["actor"] for s in all_samples))
            rng.shuffle(actors)
            split_idx = int(len(actors) * self.train_ratio)
            train_actors = set(actors[:split_idx])

            if self.split == "train":
                self.samples = [
                    (s["path"], s["label"])
                    for s in all_samples
                    if s["actor"] in train_actors
                ]
            else:
                self.samples = [
                    (s["path"], s["label"])
                    for s in all_samples
                    if s["actor"] not in train_actors
                ]

        elif self.split_by == "view":
            # Split by view for cross-view generalization
            views = sorted(set(s["view"] for s in all_samples))  # see note above
            rng.shuffle(views)
            split_idx = int(len(views) * self.train_ratio)
            train_views = set(views[:split_idx])

            if self.split == "train":
                self.samples = [
                    (s["path"], s["label"])
                    for s in all_samples
                    if s["view"] in train_views
                ]
            else:
                self.samples = [
                    (s["path"], s["label"])
                    for s in all_samples
                    if s["view"] not in train_views
                ]

        else:
            # Random split
            rng.shuffle(all_samples)
            split_idx = int(len(all_samples) * self.train_ratio)
            if self.split == "train":
                self.samples = [
                    (s["path"], s["label"]) for s in all_samples[:split_idx]
                ]
            else:
                self.samples = [
                    (s["path"], s["label"]) for s in all_samples[split_idx:]
                ]


class ARIDDataset(BaseVideoDataset):
    """
    ARID (Action Recognition in the Dark) dataset.

    First dataset for action recognition in dark/low-light videos.
    11-12 action classes with videos captured in challenging lighting conditions.

    Expected structure:
        root/
            clips_v1.5/
                Drink/
                    Drink_10_10.mp4
                    ...
                Jump/
                    ...
            list_cvt/
                split_0/
                    split0_train.txt
                    split0_test.txt
                split_1/
                    ...

    Split file format: index  label_id  relative_path
    Example: 0  10  Wave/Wave_10_1.mp4

    Source: https://xuyu0010.github.io/arid.html
    Paper: https://arxiv.org/abs/2006.03876
    """

    # Class names in order (index 0-10)
    CLASS_NAMES = [
        "Drink",
        "Jump",
        "Pick",
        "Pour",
        "Push",
        "Run",
        "Sit",
        "Stand",
        "Turn",
        "Walk",
        "Wave",
    ]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_id: int = 0,
        **kwargs,
    ):
        """
        Args:
            root: Dataset root directory
            split: "train" or "test"
            split_id: Which split to use (0 or 1)
        """
        self.split_id = split_id
        super().__init__(root, split, **kwargs)

    def _load_annotations(self):
        self.classes = self.CLASS_NAMES
        self.label_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Find video directory
        video_dir = self.root / "clips_v1.5"
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        # Load split file
        split_name = "train" if self.split == "train" else "test"
        split_file = (
            self.root
            / "list_cvt"
            / f"split_{self.split_id}"
            / f"split{self.split_id}_{split_name}.txt"
        )

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    # Format: index  label_id  path
                    label_id = int(parts[1])
                    rel_path = parts[2]
                    video_path = video_dir / rel_path
                    tar_path = video_path.with_suffix(".tar")

                    if label_id < len(self.classes):
                        if tar_path.exists():
                            self.samples.append((tar_path, label_id))
                        elif video_path.exists():
                            self.samples.append((video_path, label_id))


class Diving48Dataset(BaseVideoDataset):
    """
    Diving48 dataset — 48 competitive diving action classes.

    Uses V2 annotations (Diving48_V2_train.json / Diving48_V2_test.json).
    Videos organized by class in tar-JPEG format:
        <root>/
            <class_id>/
                video_name.tar

    Or with raw videos + annotations:
        <root>/
            videos/
                video_name.mp4
            annotations/
                Diving48_V2_train.json
                Diving48_V2_test.json
                Diving48_vocab.json

    Source: http://www.svcl.ucsd.edu/projects/resound/dataset.html
    """

    def _load_annotations(self):
        # Try tar-JPEG format first (class folders)
        class_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

        if class_dirs:
            # Tar-JPEG format: <root>/<class_id>/*.tar
            self.classes = [d.name for d in class_dirs]
            self.label_to_idx = {name: int(name) for name in self.classes}

            for class_dir in class_dirs:
                label = int(class_dir.name)
                for video_path in sorted(class_dir.glob("*.tar")):
                    self.samples.append((video_path, label))
                for video_path in sorted(class_dir.glob("*.mp4")):
                    if not (class_dir / f"{video_path.stem}.tar").exists():
                        self.samples.append((video_path, label))

            # Split by annotation files if available
            ann_dir = self.root.parent / (self.root.name.replace("-tar", "")) / "annotations"
            if not ann_dir.exists():
                ann_dir = self.root / "annotations"
            self._filter_by_split(ann_dir)
        else:
            # Raw video format with annotations
            ann_dir = self.root / "annotations"
            self._load_from_annotations(ann_dir)

    def _load_from_annotations(self, ann_dir: Path):
        """Load from JSON annotation files."""
        split_file = "Diving48_V2_train.json" if self.split == "train" else "Diving48_V2_test.json"
        ann_path = ann_dir / split_file

        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path) as f:
            annotations = json.load(f)

        # Build class list from vocab or annotations
        vocab_path = ann_dir / "Diving48_vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                vocab = json.load(f)
            self.classes = [str(i) for i in range(len(vocab))]
        else:
            labels = set(item["label"] for item in annotations)
            self.classes = [str(i) for i in sorted(labels)]
        self.label_to_idx = {name: int(name) for name in self.classes}

        video_dir = self.root / "videos"
        for item in annotations:
            vid_name = item["vid_name"]
            label = item["label"]
            # Check tar first, then video
            tar_path = video_dir / f"{vid_name}.tar"
            video_path = video_dir / f"{vid_name}.mp4"
            if tar_path.exists():
                self.samples.append((tar_path, label))
            elif video_path.exists():
                self.samples.append((video_path, label))

    def _filter_by_split(self, ann_dir: Path):
        """Filter samples by train/test split from annotation files."""
        split_file = "Diving48_V2_train.json" if self.split == "train" else "Diving48_V2_test.json"
        ann_path = ann_dir / split_file
        if not ann_path.exists():
            return  # No filter — use all samples

        with open(ann_path) as f:
            annotations = json.load(f)
        split_names = {item["vid_name"] for item in annotations}

        self.samples = [
            (path, label) for path, label in self.samples
            if path.stem in split_names
        ]
