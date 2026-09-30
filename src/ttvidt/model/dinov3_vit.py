from dataclasses import dataclass
from typing import Optional, Unpack

import torch
import torch.nn as nn
from transformers import DINOv3ViTModel, DINOv3ViTConfig
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import ModelOutput
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTEmbeddings
from optimfactory import mup_init, mup_init_output

from ttvidt.modules.layers import AttentivePooling, RMSNorm
from ttvidt.modules.patch import TemporalPatch
from ttvidt.modules.tt import TemporalTransfer
from ttvidt.modules.tt3d import TemporalTransfer3D


@dataclass
class VideoModelOutputWithMotionTokens(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    frames_output: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    motion_output: Optional[torch.FloatTensor] = None
    cls_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class DINOv3VidTConfig(DINOv3ViTConfig):
    model_type = "dinov3_vit"

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        temporal_patch_overlap: int = 0,
        hidden_size: int = 384,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 12,
        motion_layers_period: int = 0,
        num_attention_heads: int = 6,
        hidden_act: str = "gelu",
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 100.0,
        image_size: int = 224,
        num_channels: int = 3,
        query_bias: bool = True,
        key_bias: bool = False,
        value_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        layerscale_value: float = 1.0,
        drop_path_rate: float = 0.0,
        use_gated_mlp: bool = False,
        num_register_tokens: int = 0,
        num_motion_tokens: int = 0,
        motion_ffn_type: str = "swiglu",  # "swiglu" or "gelu" for TemporalTransfer
        tt_mode: str = "1d",  # "1d" (motion-only) or "3d" (motion + downsampled spatial)
        tt_downsample: int = 4,  # spatial downsample factor for tt_mode="3d"
        # train augs
        pos_embed_shift: Optional[float] = None,
        pos_embed_jitter: Optional[float] = None,
        pos_embed_rescale: Optional[float] = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_patch_overlap = temporal_patch_overlap
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.motion_layers_period = motion_layers_period
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_gated_mlp = use_gated_mlp
        self.rope_theta = rope_theta
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.proj_bias = proj_bias
        self.mlp_bias = mlp_bias
        self.num_register_tokens = num_register_tokens
        self.num_motion_tokens = num_motion_tokens
        self.motion_ffn_type = motion_ffn_type
        self.tt_mode = tt_mode
        self.tt_downsample = tt_downsample

        # train augs
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale


class DINOv3VidTModel(DINOv3ViTModel):
    config: DINOv3VidTConfig

    def __init__(self, config: DINOv3VidTConfig):
        super().__init__(config)
        self.num_motion_tokens = config.num_motion_tokens
        if self.num_motion_tokens > 0:
            self.motion_tokens = nn.Parameter(
                torch.randn(config.num_motion_tokens, config.hidden_size)
                / (config.hidden_size * config.num_motion_tokens) ** 0.5
            )
        else:
            self.motion_tokens = None
        if config.temporal_patch_size > 1:
            self.temporal_embeddings = TemporalPatch(
                config.hidden_size,
                config.hidden_size,
                config.temporal_patch_size,
                config.temporal_patch_overlap,
            )
        else:
            self.temporal_embeddings = nn.Identity()
        # Attentive pooling on motion tokens -> CLS-like conditioning token
        if self.num_motion_tokens > 0:
            self.motion_pooler = AttentivePooling(
                config.hidden_size, num_heads=config.num_attention_heads
            )
            # Pool all TT tokens → 1 motion token for decoder output
            self.motion_output_pooler = AttentivePooling(
                config.hidden_size, num_heads=config.num_attention_heads
            )
            # Normalize cls_output (AttentivePooling has no output norm; without this
            # cls_output can reach absmax~50 and destabilize AdaRMSNorm conditioning)
            self.cls_norm = RMSNorm(config.hidden_size)
        else:
            self.motion_pooler = None
            self.motion_output_pooler = None
            self.cls_norm = None

        self.motion_layers = nn.ModuleList([])
        self.tt_mode = config.tt_mode

        layer_added = 0
        for idx, layer in enumerate(self.model.layer):
            if (
                config.motion_layers_period > 0
                and idx % config.motion_layers_period == 0
            ):
                layer_added += 1
                if config.tt_mode == "3d":
                    self.motion_layers.append(
                        TemporalTransfer3D(
                            config.hidden_size,
                            config.intermediate_size,
                            config.num_attention_heads,
                            downsample_factor=config.tt_downsample,
                            ffn_type=config.motion_ffn_type,
                        )
                    )
                else:
                    self.motion_layers.append(
                        TemporalTransfer(
                            config.hidden_size,
                            config.intermediate_size,
                            config.num_attention_heads,
                            ffn_type=config.motion_ffn_type,
                        )
                    )
            else:
                self.motion_layers.append(nn.Identity())

        mup_init(self.motion_layers.parameters())
        # Restore special inits clobbered by mup_init
        for layer in self.motion_layers:
            if isinstance(layer, nn.Identity):
                continue
            # Residual output projections: near-zero init
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'out_proj'):
                mup_init_output(layer.attn.out_proj.weight)
            if hasattr(layer, 'out_proj'):  # TT3D has out_proj on the module
                mup_init_output(layer.out_proj.weight)
            if hasattr(layer.mlp, 'fc2'):
                mup_init_output(layer.mlp.fc2.weight)
            elif hasattr(layer.mlp, 'down'):
                mup_init_output(layer.mlp.down.weight)
            # TT3D-specific: zero-init spatial_out_proj, identity resample channel mix, qk_scale
            if hasattr(layer, 'spatial_out_proj'):
                nn.init.zeros_(layer.spatial_out_proj.weight)
                layer.spatial_down.reset_parameters()
                layer.spatial_up.reset_parameters()
            for module in layer.modules():
                if hasattr(module, 'qk_scale'):
                    module.qk_scale.data.fill_(10.0)
        print(f"Motion layers added: {layer_added} / {config.num_hidden_layers} (mode={config.tt_mode})")

    @torch.no_grad()
    def _init_weights(self, module) -> None:
        super()._init_weights(module)
        # transformers' from_pretrained builds on the meta device and leaves
        # non-persistent buffers empty: recompute those of the TT modules
        if hasattr(module, "reset_buffers"):
            module.reset_buffers()

    def freeze_pretrained(self, requires_grad: bool = False):
        """Freeze or unfreeze the pretrained backbone components.

        Pretrained components: embeddings, rope_embeddings, transformer layers, norm.
        Leaves trainable-from-scratch parts (motion_tokens, motion_layers,
        temporal_embeddings) untouched.
        """
        self.embeddings.requires_grad_(requires_grad)
        self.rope_embeddings.requires_grad_(requires_grad)
        self.model.layer.requires_grad_(requires_grad)
        self.norm.requires_grad_(requires_grad)

    def setup_patch(self, num_channels, patch_size):
        self.config.num_channels = num_channels
        self.config.patch_size = patch_size
        self.embeddings = DINOv3ViTEmbeddings(self.config)
        mup_init(self.embeddings.parameters())

    @classmethod
    def from_dino_v3(
        cls,
        dino_v3: DINOv3ViTModel,
        temporal_patch_size: int = 0,
        temporal_patch_overlap: int = 0,
        motion_layers_period: int = 0,
        num_motion_tokens: int = 0,
        motion_ffn_type: str = "swiglu",
        tt_mode: str = "1d",
        tt_downsample: int = 4,
    ):
        dino_v3_config = dino_v3.config
        ttvidt_config = DINOv3VidTConfig(
            **dino_v3_config.to_dict(),
            temporal_patch_size=temporal_patch_size,
            temporal_patch_overlap=temporal_patch_overlap,
            motion_layers_period=motion_layers_period,
            num_motion_tokens=num_motion_tokens,
            motion_ffn_type=motion_ffn_type,
            tt_mode=tt_mode,
            tt_downsample=tt_downsample,
        )
        new_model = cls(ttvidt_config)
        new_model.load_state_dict(dino_v3.state_dict(), strict=False)
        return new_model

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VideoModelOutputWithMotionTokens:
        r"""
        pixel_values: [B, T, C, H, W]
        spatial_mask: optional bool tensor [L], True = visible/keep (for MAE tube masking)
        """
        b, t = pixel_values.shape[:2]

        _, _, _, H_in, W_in = pixel_values.shape
        ps = self.config.patch_size
        spatial_h = H_in // ps
        spatial_w = W_in // ps

        pixel_values = pixel_values.to(
            self.embeddings.patch_embeddings.weight.dtype
        ).flatten(0, 1)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)
        pixel_length = position_embeddings[0].size(0)

        hidden_states = self.temporal_embeddings(hidden_states.unflatten(0, (b, -1)))

        # MAE tube masking: keep only visible spatial tokens
        if spatial_mask is not None:
            # hidden_states may have prefix tokens (CLS + registers) before spatial tokens
            # spatial_mask: bool [L] where L = num spatial patches
            n_prefix = hidden_states.shape[2] - spatial_mask.shape[0]
            if n_prefix > 0:
                prefix = hidden_states[:, :, :n_prefix]
                spatial = hidden_states[:, :, n_prefix:]
                hidden_states = torch.cat([prefix, spatial[:, :, spatial_mask]], dim=2)
            else:
                hidden_states = hidden_states[:, :, spatial_mask]
            # Subsample RoPE position embeddings: (cos[L, head_dim], sin[L, head_dim])
            cos, sin = position_embeddings
            position_embeddings = (cos[spatial_mask], sin[spatial_mask])
            pixel_length = spatial_mask.sum().item()
        b, ot, seq_len, dim = hidden_states.shape
        if self.motion_tokens is not None:
            hidden_states = torch.concat(
                [self.motion_tokens[None, None].repeat(b, ot, 1, 1), hidden_states],
                dim=2,
            )

        hidden_states = hidden_states.flatten(0, 1)
        for i, (layer_module, motion_module) in enumerate(
            zip(self.model.layer, self.motion_layers)
        ):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(
                hidden_states,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
            )
            if not isinstance(motion_module, nn.Identity):
                hidden_states = hidden_states.unflatten(0, (b, ot))
                motion_tokens = hidden_states[:, :, : self.num_motion_tokens]
                vit_state = hidden_states[:, :, self.num_motion_tokens :]

                if isinstance(motion_module, TemporalTransfer3D):
                    # TT3D: pass spatial tokens (last pixel_length of vit_state)
                    # vit_state layout: [CLS, REG_0..REG_3, S_0..S_L]
                    spatial_tokens = vit_state[:, :, -pixel_length:]

                    if spatial_mask is not None:
                        # MAE mode: pad visible tokens back to full spatial grid
                        # for pixel unshuffle/shuffle in TT3D, then extract visible
                        L_full = spatial_h * spatial_w
                        full_spatial = spatial_tokens.new_zeros(
                            b, ot, L_full, dim
                        )
                        full_spatial[:, :, spatial_mask] = spatial_tokens
                        motion_tokens, full_spatial_out = motion_module(
                            motion_tokens, full_spatial, spatial_h, spatial_w
                        )
                        spatial_out = full_spatial_out[:, :, spatial_mask]
                    else:
                        motion_tokens, spatial_out = motion_module(
                            motion_tokens, spatial_tokens, spatial_h, spatial_w
                        )

                    # Write spatial back (only last pixel_length of vit_state)
                    vit_state = torch.cat(
                        [vit_state[:, :, :-pixel_length], spatial_out], dim=2
                    )
                else:
                    # TT1D: motion tokens only
                    motion_tokens = motion_module(motion_tokens)

                hidden_states = torch.concat([motion_tokens, vit_state], dim=2)
                hidden_states = hidden_states.flatten(0, 1)

        sequence_output = self.norm(hidden_states).unflatten(0, (b, ot))
        frames_output = sequence_output[:, :, -pixel_length:, :]
        pooled_output = sequence_output[:, :, self.num_motion_tokens, :]
        if self.motion_tokens is None:
            motion_output = None
            cls_output = None
        else:
            all_tt_tokens = sequence_output[
                :, :, : self.num_motion_tokens, :
            ]  # [B, T, num_motion_tokens, dim]
            cls_output = self.cls_norm(self.motion_pooler(all_tt_tokens))  # [B, T, dim]
            # Pool all TT tokens → 1 motion token for decoder
            motion_output = self.motion_output_pooler(all_tt_tokens).unsqueeze(2)  # [B, T, 1, dim]

        return VideoModelOutputWithMotionTokens(
            last_hidden_state=sequence_output,
            frames_output=frames_output,
            pooler_output=pooled_output,
            motion_output=motion_output,
            cls_output=cls_output,
        )


if __name__ == "__main__":
    dino = DINOv3ViTModel.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m"
    ).cuda()
    dino_vidt = DINOv3VidTModel.from_dino_v3(
        dino,
        temporal_patch_size=4,
        temporal_patch_overlap=1,
        motion_layers_period=1,
        num_motion_tokens=16,
    ).cuda()

    print(dino)
    print(dino_vidt)

    test_x = torch.randn(1, 61, 3, 256, 256).cuda()
    with torch.autocast("cuda"):
        result = dino_vidt(test_x)

    print(result.frames_output.shape)
    print(result.pooler_output.shape)
    print(result.motion_output.shape)
