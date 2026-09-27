"""
VideoMAE-style 3D ViT backbone for TT-VidT.

Full spatio-temporal attention: all tokens (across all frames) attend to all others.
Inspired by VideoMAE (MCG-NJU) which uses a joint space-time ViT encoder with
tubelet embedding and sincos 3D position embeddings.

Key differences from the DINOv3VidT baseline:
  - No pretrained 2D ViT; the encoder is built from scratch as a 3D ViT.
  - Patch embedding via Linear projection on latent-space input [B, T, C, H, W].
  - TemporalPatch for temporal grouping before transformer blocks.
  - Tokens from ALL temporal frames are flattened into a single sequence for
    full 3D self-attention (every token attends to every other token).
  - Motion tokens are prepended per temporal frame before flattening.
  - TemporalTransfer modules applied at periodic intervals (unflatten, apply,
    reflatten) to inject causal temporal structure into motion tokens.
  - 3D sincos position embedding (temporal + spatial).
  - Standard transformer blocks: RMSNorm + Attention + SwiGLU.

Architecture at ViT-B scale (~100-200M params):
  hidden_size=768, intermediate_size=3072, num_heads=12, depth=12
"""

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from optimfactory import mup_init, mup_init_output

from ttvidt.modules.layers import (
    RMSNorm,
    SwiGLU,
    Attention,
    AttentivePooling,
    TransformerBlock,
    LatentPatchEmbed,
)
from ttvidt.modules.patch import TemporalPatch
from ttvidt.modules.tt import TemporalTransfer
from ttvidt.modules.pos_embed import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from ttvidt.utils import compile_wrapper
from ttvidt.model.dinov3_vit import VideoModelOutputWithMotionTokens

# ---------------------------------------------------------------------------
# 3D Sinusoidal Position Embedding
# ---------------------------------------------------------------------------


def get_3d_sincos_pos_embed(embed_dim: int, t: int, h: int, w: int) -> torch.Tensor:
    """
    Generate 3D sinusoidal positional embeddings (temporal + spatial).

    The embedding dimension is split: 1/4 for temporal, 3/4 for spatial (split
    equally between h and w within the spatial portion, matching the 2D sincos
    convention used elsewhere in the project).

    Args:
        embed_dim: Total embedding dimension.
        t: Number of temporal positions.
        h: Spatial height (in patches).
        w: Spatial width (in patches).

    Returns:
        [T*H*W, embed_dim] position embeddings.
    """
    # Split: temporal gets 1/4, spatial h gets 3/8, spatial w gets 3/8
    # Use a simpler split: temporal = embed_dim // 4, spatial = embed_dim - temporal_dim
    temporal_dim = embed_dim // 4
    spatial_dim = embed_dim - temporal_dim

    # Temporal embeddings: [T, temporal_dim]
    temporal_pos = torch.arange(t, dtype=torch.float32)
    temporal_emb = get_1d_sincos_pos_embed(
        temporal_dim, temporal_pos
    )  # [T, temporal_dim]

    # Spatial embeddings: [H*W, spatial_dim]
    spatial_emb = get_2d_sincos_pos_embed(spatial_dim, h, w)  # [H*W, spatial_dim]

    # Combine: broadcast [T, 1, temporal_dim] + [1, H*W, spatial_dim] -> [T, H*W, embed_dim]
    temporal_emb = temporal_emb.unsqueeze(1).expand(
        -1, h * w, -1
    )  # [T, H*W, temporal_dim]
    spatial_emb = spatial_emb.unsqueeze(0).expand(t, -1, -1)  # [T, H*W, spatial_dim]

    pos_embed = torch.cat([temporal_emb, spatial_emb], dim=-1)  # [T, H*W, embed_dim]
    return pos_embed.reshape(t * h * w, embed_dim)


# ---------------------------------------------------------------------------
# VideoMAE3DViTModel
# ---------------------------------------------------------------------------


@dataclass
class VideoMAE3DConfig:
    """Configuration for VideoMAE3DViTModel."""

    in_channels: int = 3
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    spatial_patch_size: int = 1
    temporal_patch_size: int = 4
    temporal_patch_overlap: int = 1
    num_motion_tokens: int = 16
    motion_layers_period: int = 3
    ffn_type: str = "swiglu"  # "swiglu" or "gelu"
    drop_path_rate: float = 0.0

    @property
    def patch_size(self) -> int:
        """Alias for spatial_patch_size, for compatibility with the trainer."""
        return self.spatial_patch_size


class VideoMAE3DViTModel(nn.Module):
    """
    VideoMAE-style 3D Vision Transformer for video understanding.

    All spatial and temporal tokens are flattened into a single sequence and
    processed with full self-attention.  Motion tokens are prepended per
    temporal frame to capture temporal dynamics, and TemporalTransfer modules
    are inserted at periodic intervals for causal temporal reasoning.
    """

    def __init__(self, config: VideoMAE3DConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_motion_tokens = config.num_motion_tokens

        # -- Patch embedding --------------------------------------------------
        self.patch_embed = LatentPatchEmbed(
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            spatial_patch_size=config.spatial_patch_size,
        )

        # -- Temporal patch grouping ------------------------------------------
        if config.temporal_patch_size > 1:
            self.temporal_patch = TemporalPatch(
                config.hidden_size,
                config.hidden_size,
                config.temporal_patch_size,
                config.temporal_patch_overlap,
            )
        else:
            self.temporal_patch = nn.Identity()

        # -- Motion tokens ----------------------------------------------------
        if config.num_motion_tokens > 0:
            self.motion_tokens = nn.Parameter(
                torch.randn(config.num_motion_tokens, config.hidden_size)
                / (config.hidden_size * config.num_motion_tokens) ** 0.5
            )
        else:
            self.motion_tokens = None

        # -- CLS conditioning from motion tokens --------------------------------
        if config.num_motion_tokens > 1:
            self.motion_pooler = AttentivePooling(
                config.hidden_size, num_heads=config.num_attention_heads
            )
        elif config.num_motion_tokens == 1:
            self.motion_pooler = None  # single token: just squeeze, no pooling needed
        else:
            self.motion_pooler = None
        if config.num_motion_tokens > 0:
            self.cls_norm = RMSNorm(config.hidden_size)
        else:
            self.cls_norm = None

        # -- Transformer blocks (+ optional TemporalTransfer) ------------------
        self.blocks = nn.ModuleList()
        self.motion_layers = nn.ModuleList()
        motion_layer_count = 0

        for idx in range(config.num_hidden_layers):
            self.blocks.append(
                TransformerBlock(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    ffn_type=config.ffn_type,
                )
            )
            if (
                config.motion_layers_period > 0
                and idx % config.motion_layers_period == 0
            ):
                motion_layer_count += 1
                self.motion_layers.append(
                    TemporalTransfer(
                        config.hidden_size,
                        config.intermediate_size,
                        config.num_attention_heads,
                        ffn_type=config.ffn_type,
                    )
                )
            else:
                self.motion_layers.append(nn.Identity())

        # -- Final norm --------------------------------------------------------
        self.norm = RMSNorm(config.hidden_size)

        # -- Initialization ----------------------------------------------------
        mup_init(self.blocks.parameters())
        mup_init(self.motion_layers.parameters())
        # Zero-like init for residual output projections
        for block in list(self.blocks) + list(self.motion_layers):
            if isinstance(block, nn.Identity):
                continue
            mup_init_output(block.attn.out_proj.weight)
            if hasattr(block.mlp, 'fc2'):
                mup_init_output(block.mlp.fc2.weight)
            elif hasattr(block.mlp, 'down'):
                mup_init_output(block.mlp.down.weight)

        self.gradient_checkpointing = False

        print(
            f"[VideoMAE3DViTModel] layers={config.num_hidden_layers}, "
            f"motion_layers={motion_layer_count}/{config.num_hidden_layers}, "
            f"ffn={config.ffn_type}, "
            f"hidden={config.hidden_size}, heads={config.num_attention_heads}"
        )

    def freeze_pretrained(self, requires_grad: bool = False):
        """Freeze or unfreeze the backbone components.

        Backbone: patch_embed, transformer blocks, norm.
        Leaves trainable-from-scratch parts (motion_tokens, motion_layers,
        temporal_patch) untouched.
        """
        self.patch_embed.requires_grad_(requires_grad)
        self.blocks.requires_grad_(requires_grad)
        self.norm.requires_grad_(requires_grad)

    # ------------------------------------------------------------------
    # Position embedding (computed on the fly for arbitrary T, H, W)
    # ------------------------------------------------------------------

    def _get_pos_embed(
        self, t: int, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return 3D sincos position embeddings [1, T*(M+L), D]."""
        L = h * w
        M = self.num_motion_tokens

        # Spatial-temporal sincos for visual tokens: [T*L, D]
        pos = get_3d_sincos_pos_embed(self.hidden_size, t, h, w).to(
            device=device, dtype=dtype
        )
        # Reshape to [T, L, D]
        pos = pos.reshape(t, L, self.hidden_size)

        if M > 0:
            # Motion tokens get temporal sincos but zero spatial.
            # Build temporal-only embedding: [T, temporal_dim] with zero spatial
            temporal_dim = self.hidden_size // 4
            temporal_pos = torch.arange(t, dtype=torch.float32)
            temporal_emb = get_1d_sincos_pos_embed(temporal_dim, temporal_pos).to(
                device=device, dtype=dtype
            )  # [T, temporal_dim]
            # Pad with zeros for the spatial portion
            spatial_pad = torch.zeros(t, self.hidden_size - temporal_dim, device=device, dtype=dtype)
            motion_pos = torch.cat([temporal_emb, spatial_pad], dim=-1)  # [T, D]
            motion_pos = motion_pos.unsqueeze(1).expand(-1, M, -1)  # [T, M, D]
            pos = torch.cat([motion_pos, pos], dim=1)  # [T, M+L, D]

        return pos.reshape(1, t * (M + L), self.hidden_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> VideoModelOutputWithMotionTokens:
        """
        Args:
            pixel_values: [B, T, C, H, W] input video (can be raw pixels or
                          latent representations).
            spatial_mask: optional bool tensor [L], True = visible/keep
                          (for MAE tube masking).  The same mask is applied to
                          every temporal frame so that the same spatial tubes
                          are kept across time.

        Returns:
            VideoModelOutputWithMotionTokens with:
              - frames_output:  [B, 1+T', L, D]  (or L_vis when spatial_mask)
              - motion_output:  [B, 1+T', M, D]  (or None if M == 0)
              - last_hidden_state: [B, 1+T', M+L, D]
              - pooler_output: [B, 1+T', D]  (first visual token per frame)
        """
        b, t_in, c, h_in, w_in = pixel_values.shape

        # 1. Patch embedding: [B, T, C, H, W] -> [B, T, L, D]
        x = self.patch_embed(pixel_values.to(self.norm.weight.dtype))

        # Compute spatial dims after patch embedding
        sp = self.config.spatial_patch_size
        h = h_in // sp
        w = w_in // sp
        L_full = h * w

        # 2. Temporal patch grouping: [B, T, L, D] -> [B, 1+T', L, D]
        x = self.temporal_patch(x)
        b, t_out, _, D = x.shape

        # 3. MAE tube masking: keep only visible spatial tokens
        if spatial_mask is not None:
            # spatial_mask: bool [L], True = keep
            # x: [B, 1+T', L, D] -> [B, 1+T', L_vis, D]
            x = x[:, :, spatial_mask]
        L = x.shape[2]

        M = self.num_motion_tokens

        # 4. Prepend motion tokens per temporal frame
        if self.motion_tokens is not None:
            mt = self.motion_tokens[None, None].expand(
                b, t_out, -1, -1
            )  # [B, 1+T', M, D]
            x = torch.cat([mt, x], dim=2)  # [B, 1+T', M+L, D]

        # 5. Flatten to single sequence for full 3D attention
        #    The Attention module expects >= 4D input [B, ..., seq, D] and
        #    calls flatten(1, 2).  We use [B, 1, (1+T')*(M+L), D] so that
        #    flatten(1,2) yields [B, (1+T')*(M+L), D] for the linear projections.
        total_per_frame = M + L
        x = x.reshape(b, 1, t_out * total_per_frame, D)  # [B, 1, (1+T')*(M+L), D]

        # 6. Add 3D position embedding (subsampled when spatial_mask is active)
        if spatial_mask is not None:
            pos_embed = self._get_pos_embed(t_out, h, w, x.device, x.dtype)
            # pos_embed: [1, (1+T')*(M+L_full), D] — reshape to [1, 1+T', M+L_full, D]
            pos_embed = pos_embed.reshape(1, t_out, M + L_full, D)
            if M > 0:
                # Keep all motion positions, subsample only visual positions
                motion_pos = pos_embed[:, :, :M, :]
                visual_pos = pos_embed[:, :, M:, :][:, :, spatial_mask, :]
                pos_embed = torch.cat([motion_pos, visual_pos], dim=2)
            else:
                pos_embed = pos_embed[:, :, spatial_mask, :]
            pos_embed = pos_embed.reshape(1, t_out * total_per_frame, D)
        else:
            pos_embed = self._get_pos_embed(t_out, h, w, x.device, x.dtype)
        x = x + pos_embed.unsqueeze(1)  # broadcast [1, 1, seq, D]

        # 7. Transformer blocks with periodic TemporalTransfer
        for i, (block, motion_module) in enumerate(
            zip(self.blocks, self.motion_layers)
        ):
            if self.gradient_checkpointing and self.training:
                x = ckpt.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

            # TemporalTransfer: unflatten -> apply to motion tokens -> reflatten
            if not isinstance(motion_module, nn.Identity) and M > 0:
                x = x.reshape(b, t_out, total_per_frame, D)
                motion_part = x[:, :, :M, :]  # [B, 1+T', M, D]
                visual_part = x[:, :, M:, :]  # [B, 1+T', L, D]

                if self.gradient_checkpointing and self.training:
                    motion_part = ckpt.checkpoint(
                        motion_module, motion_part, use_reentrant=False
                    )
                else:
                    motion_part = motion_module(motion_part)

                x = torch.cat([motion_part, visual_part], dim=2)
                x = x.reshape(b, 1, t_out * total_per_frame, D)

        # 8. Final norm — squeeze the dummy dim first
        x = x.reshape(b, t_out * total_per_frame, D)
        x = self.norm(x)

        # 9. Reshape back to [B, 1+T', M+L, D]
        x = x.reshape(b, t_out, total_per_frame, D)

        frames_output = x[:, :, M:, :]  # [B, 1+T', L, D]
        pooler_output = x[:, :, M, :]  # [B, 1+T', D] — first visual token

        if self.motion_tokens is not None:
            motion_output = x[:, :, :M, :]  # [B, 1+T', M, D]
            if self.motion_pooler is not None:
                cls_output = self.cls_norm(self.motion_pooler(motion_output))  # [B, 1+T', D]
            else:
                # M=1: squeeze instead of pooling
                cls_output = self.cls_norm(motion_output.squeeze(2))  # [B, 1+T', D]
        else:
            motion_output = None
            cls_output = None

        return VideoModelOutputWithMotionTokens(
            last_hidden_state=x,
            frames_output=frames_output,
            pooler_output=pooler_output,
            motion_output=motion_output,
            cls_output=cls_output,
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("VideoMAE3DViTModel smoke test")
    print("=" * 60)

    cfg = VideoMAE3DConfig(
        in_channels=3,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        spatial_patch_size=16,
        temporal_patch_size=4,
        temporal_patch_overlap=1,
        num_motion_tokens=8,
        motion_layers_period=3,
    )
    model = VideoMAE3DViTModel(cfg)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.1f}M")

    # Test input: 1 video, 17 frames, 3 channels, 224x224
    test_x = torch.randn(1, 17, 3, 224, 224)
    with torch.no_grad():
        out = model(test_x)

    print(f"frames_output:  {out.frames_output.shape}")
    print(f"pooler_output:  {out.pooler_output.shape}")
    if out.motion_output is not None:
        print(f"motion_output:  {out.motion_output.shape}")
    else:
        print("motion_output:  None")
    print(f"last_hidden_state: {out.last_hidden_state.shape}")
    print("PASSED")
