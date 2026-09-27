"""TT-VidT (proposed): TT3D + Diff Compression, decoder S video-pretrained.

The model reported in the final comparison (Table 4).
"""

from _base.pretrain import *  # noqa: F401,F403

RUN_NAME = "ttvidt_tt3d_diffcomp"
DECODER_PRETRAINED = f"{HF_DECODERS}/pretrain_video_S_qknorm"


def config_gen():
    print("run id:", RUN_ID)
    return Config.from_globals()
