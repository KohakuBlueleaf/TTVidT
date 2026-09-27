# Data

All video data, for pretraining and evaluation, is stored as **tar-of-JPEG**: one
`.tar` per clip holding `meta.json` (`{"frame_count": N, "fps": F, ...}`) and frames
`000000.jpg`, `000001.jpg`, ... Reading needs no video decoder and is fast from any
file system.

```
data/                         pretraining (encoder + video decoder)
  openvid384-tar/             OpenVid-1M at 384 px, ~1.0M clips
  moments_in_time-tar/        Moments-in-Time v2 training set, ~0.76M clips
  imagenet-1k-tar/            ImageNet-1k shards (ImageNet decoder pretraining only)
eval-dataset/                 benchmarks
  hmdb51-tar/  arid-tar/  iard-tar/  jester-tar/  sthsthv2-tar/
  diving48-tar/  ek100-tar/  ek100_anticipation-tar/
```

The roots default to `data/` and `eval-dataset/`; set `TTVIDT_DATA` to move the
pretraining root, or override `DATASET_FOLDERS` / `DATA_ROOT` in a config.

## Pretraining corpora

1. **OpenVid-1M, 384 px**: download the 384 px release
   (`huggingface-cli download --repo-type dataset KBlueLeaf/OpenVid384px --local-dir data/openvid384`,
   then extract the archives into mp4 files).
2. **Moments-in-Time v2**: request access at the official site and extract the
   training videos to `data/moments_in_time/`.
3. Convert both to tar-of-JPEG and build the index files that make dataset start-up fast:

```bash
python scripts/data/convert_videos_to_tar.py data/openvid384 data/openvid384-tar --quality 85 --workers 64
python scripts/data/convert_videos_to_tar.py data/moments_in_time data/moments_in_time-tar --quality 85 --workers 64
python scripts/data/build_tar_index.py data/openvid384-tar data/moments_in_time-tar
```

Needs `ffmpeg` on `PATH` and `pip install -e ".[data]"`.

**ImageNet-1k** (only for ImageNet decoder pretraining): download the parquet
release from the Hugging Face Hub, then write 64-image shards:

```bash
python scripts/data/convert_imagenet_to_tar.py data/imagenet-1k data/imagenet-1k-tar
```

The shard size must equal `BATCH_SIZE` in `configs/_base/decoder.py` (64).

## Frame VAE

Reconstruction targets live in the latent space of a frozen frame VAE. The one used
in the paper is [`KBlueLeaf/latentmaid-vae`](https://huggingface.co/KBlueLeaf/latentmaid-vae)
(diffusers format, f8, 4 latent channels, latent statistics in its `config.json`). It
is the default (`DECODER_AE_PATH` / `AE_MODEL_PATH`, or the `TTVIDT_FRAME_VAE`
variable) and is downloaded automatically.

Any VAE with the diffusers `AutoencoderKL` API can replace it: a local diffusers
folder, a Hub id such as `KBlueLeaf/EQ-SDXL-VAE`, or `"<repo>:<subfolder>"` for a
VAE inside a pipeline repo (see `ttvidt/model/frame_vae.py`). The decoder's
`image_dim` must equal the VAE's latent channels (checked at start-up), and
`LATENT_MEAN` / `LATENT_STD` default to the `latents_mean` / `latents_std` stored in
the VAE config; set them in the config if the VAE has none.

The frame VAE is only needed for pretraining, not for evaluation.

## Benchmarks

Each benchmark has a setup script in `scripts/data/benchmarks/` that downloads (or
explains how to obtain) the data and organises it. Then convert to tar-of-JPEG with
`scripts/data/benchmarks/convert_to_tar.py` where noted.

| Benchmark | Classes | Train / test clips | Setup |
|---|---:|---:|---|
| HMDB51 (split 1) | 51 | 3,518 / 1,511 | `setup_hmdb51.sh` + convert |
| ARID (split 0) | 11 | 3,350 / 2,011 | `setup_arid.sh` + convert |
| IARD (held-out actors) | 5 | 3,640 / 910 | `setup_iard.sh` + convert |
| Jester | 27 | 118,562 / 14,787 | `setup_jester.sh` + convert |
| Something-Something v2 | 174 | 168,913 / 24,777 | `setup_sthsthv2.sh` |
| Diving48 (V2) | 48 | 15,027 / 1,970 | `setup_diving48.sh` (includes conversion) |
| EPIC-Kitchens-100 verb (trimmed) | 97 | | `setup_ek100_verb.sh` (includes conversion) |
| EPIC-Kitchens-100 verb anticipation | 97 | | `setup_ek100_anticipation.sh` (includes conversion) |

HMDB51 counts exclude the 71 clips that fail to decode.

**The loaders skip clips that are missing on disk without an error.** An incomplete
download or conversion therefore silently shrinks the probe's training set and
lowers every model's accuracy. Jester (148,092 clips from a 23 GB multi-part
archive) is the easiest to truncate. After extracting features, always run

```bash
python scripts/eval/check_features.py <model_name>
```

which compares the clip counts with the table above (`run_frozen_eval.sh` does this
automatically).

### Layouts expected by the loaders

| Benchmark | Clips | Split / label files |
|---|---|---|
| HMDB51 | `hmdb51-tar/<class>/<clip>.tar` | `hmdb51-tar/splits/<class>_test_split1.txt` (official) |
| ARID | `arid-tar/clips_v1.5/<class>/<clip>.tar` | `arid-tar/list_cvt/split_0/split0_{train,test}.txt` (official) |
| IARD | `iard-tar/videos/<action>/<clip>.tar` | none; the actor split is made at load time |
| Jester | `jester-tar/videos/<id>.tar` | `jester-tar/jester-v1-{labels,train,validation}.csv` |
| SSv2 | `sthsthv2-tar/videos/<id>.tar` | `sthsthv2-tar/labels/{labels,train,validation}.json` |
| Diving48 | `diving48-tar/<class>/<clip>.tar` | `Diving48_V2_{train,test}.json`, `Diving48_vocab.json` in `diving48-tar/annotations/` or `diving48/annotations/` |
| EK-100 | `ek100-tar/{train,val}/<clip>.tar` | `ek100-tar/annotations/EPIC_100_{train,validation}.csv` |

When the clips sit directly in the dataset folder, add a self-link instead of
moving them, e.g. `ln -s . eval-dataset/jester-tar/videos` or
`ln -s . eval-dataset/arid-tar/clips_v1.5`.

Converting a folder of videos while keeping the class sub-folders:

```bash
python scripts/data/benchmarks/convert_to_tar.py --input eval-dataset/hmdb51/videos \
    --output eval-dataset/hmdb51-tar --preserve-structure
```
