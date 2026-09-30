"""Hugging Face ``transformers`` wrapper around the TT-VidT encoder.

``scripts/tools/export_hf.py`` copies this file (as ``modeling_ttvidt.py``) and the
encoder sources next to an exported model, so the same directory / Hub repo loads with

    from transformers import AutoModel
    model = AutoModel.from_pretrained("<repo>", trust_remote_code=True)
    motion = model(video).motion_output      # video: [B, T, 3, 256, 256] in [-1, 1]

and with ``ttvidt.hub.load_model("<repo>")``: ``config.json`` holds the release format
(``format``, ``model`` = TTVidTrainer arguments) plus the ``transformers`` fields, and
``model.safetensors`` uses the TTVidTrainer key names (``encoder.*``).
"""

from typing import ClassVar

from transformers import PretrainedConfig, PreTrainedModel

from ttvidt.model.dinov3_vit import DINOv3VidTConfig, DINOv3VidTModel


class TTVidTConfig(PretrainedConfig):
    """``model``: the TTVidTrainer keyword arguments of the release (``backbone_config``
    holds the full DINOv3VidTConfig, so no DINOv3 download is needed)."""

    model_type = "ttvidt"

    def __init__(
        self,
        format: str | None = None,
        model: dict | None = None,
        encoder_only: bool = False,
        **kwargs,
    ):
        self.format = format
        self.model = model or {}
        self.encoder_only = encoder_only
        super().__init__(**kwargs)

    def encoder_config(self) -> DINOv3VidTConfig:
        if self.model.get("backbone_arch", "ttvidt") != "ttvidt":
            raise ValueError(
                f"only TT-VidT encoders are supported, got {self.model.get('backbone_arch')!r}"
            )
        cfg = dict(self.model["backbone_config"])
        for k, v in (self.model.get("motion_encoder_config") or {}).items():
            cfg.setdefault(k, v)  # same merge as TTVidTrainer
        return DINOv3VidTConfig(**cfg)


class TTVidTModel(PreTrainedModel):
    """The TT-VidT video encoder (``DINOv3VidTModel``); ``forward`` returns its output
    (``motion_output``: [B, T, K, D] motion tokens per frame)."""

    config_class = TTVidTConfig
    base_model_prefix = "ttvidt"
    main_input_name = "pixel_values"
    # the decoder and the frame-VAE latent statistics of full releases are not used
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^decoder\.",
        r"^latent_(mean|std)$",
    ]

    def __init__(self, config: TTVidTConfig):
        super().__init__(config)
        self.encoder = DINOv3VidTModel(config.encoder_config())
        self.post_init()

    def _init_weights(self, module):
        pass  # DINOv3VidTModel initialises itself; pretrained weights come from the file

    def forward(self, pixel_values, **kwargs):
        """pixel_values: [B, T, 3, H, W] in [-1, 1]."""
        return self.encoder(pixel_values, **kwargs)
