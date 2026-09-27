"""End-to-end smoke test of encoder pretraining on synthetic video.

Builds the model exactly as ``scripts/train/pretrain_encoder.py`` does for a given
config (architecture, objective, decoder, frame VAE) and runs a few optimisation
steps through Lightning on random clips. No dataset is needed; the frame VAE and
(for TT-VidT / DisMo) the DINOv3 weights are.

    python scripts/tools/smoke_test.py configs/pretrain/ttvidt_tt3d_diffcomp.py [--steps 2] [--gpu]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ttvidt.config import load_config

REPO = Path(__file__).resolve().parents[2]


class RandomClips(Dataset):
    def __init__(self, n, frames, dual):
        self.n, self.frames, self.dual = n, frames, dual

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        clip = torch.rand(self.frames, 3, 256, 256) * 2 - 1
        return (clip, clip.clone()) if self.dual else clip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("pretrain_encoder", REPO / "scripts/train/pretrain_encoder.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    cfg = load_config(a.config)
    for k, v in cfg.globals_dict.items():
        setattr(script, k, v)
    script.DECODER_PRETRAINED = None  # weights irrelevant for a smoke test; avoids a download

    sched = {"lr": {"mode": "cosine", "end": a.steps + 1, "min_value": 0.01, "warmup": 1}}
    model = script.TTVidTrainer(**script.model_kwargs(sched))
    frames = script.FRAME_COUNT + script.BUFFER_FRAMES
    loader = DataLoader(RandomClips(a.steps, frames, script.DUAL_AUG), batch_size=1)
    trainer = pl.Trainer(max_steps=a.steps, accelerator="gpu" if a.gpu else "cpu", devices=1,
                         precision="16-mixed" if a.gpu else "32", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         gradient_clip_val=script.GRAD_CLIP_VAL)
    trainer.fit(model, loader)
    loss = trainer.callback_metrics
    print(f"OK {Path(a.config).name}: arch={script.BACKBONE_ARCH} mode={script.TRAIN_MODE} "
          f"steps={trainer.global_step} metrics={ {k: round(float(v), 4) for k, v in loss.items()} }")
    if trainer.global_step != a.steps:
        sys.exit("did not complete the requested steps")


if __name__ == "__main__":
    main()
