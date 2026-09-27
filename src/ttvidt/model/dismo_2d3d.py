"""
DisMo-inspired 2D+3D ViT backbone for TT-VidT.

Two-phase architecture that disentangles spatial and temporal processing:

Phase 1 (Spatial / 2D):
  - Per-frame 2D self-attention over spatial tokens.
  - NO motion tokens in this phase — purely spatial feature extraction.
  - Uses 2D sincos positional embeddings.
  - Can be initialized from a pretrained DINOv3 ViT (via from_dino_v3).

Phase 2 (Temporal / 3D):
  - Motion tokens are prepended per temporal frame.
  - All tokens are flattened for full spatio-temporal attention.
  - TemporalTransfer at periodic intervals for causal motion reasoning.
  - Learnable temporal position embeddings added on top of the spatial
    embeddings carried over from Phase 1.

Inspired by the DisMo framework which decouples motion capture from spatial
encoding, allowing efficient spatial pre-training + temporal fine-tuning.

Architecture at ViT-B scale (~100-200M params):
  Phase 1: hidden_size=768, depth=8, heads=12  (spatial)
  Phase 2: hidden_size=768, depth=4, heads=12  (temporal)
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
    GELUMLP,
)


class DINOv3LayerWrapper(nn.Module):
    """Wraps a DINOv3ViTLayer to match our [B, T, L, D] tensor API.

    Stores the rope_embeddings module to compute position embeddings
    for the DINOv3 layer's attention (which requires (cos, sin) tuple).
    """

    def __init__(self, layer, rope_embeddings):
        super().__init__()
        self.layer = layer
        self.rope_embeddings = rope_embeddings
        self._pos_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pos_cache_key: tuple[int, int, str] = (0, 0, "")

    def _get_position_embeddings(self, h: int, w: int, device: torch.device):
        key = (h, w, str(device))
        if self._pos_cache_key != key:
            # rope_embeddings expects [B, C, H, W] pixel_values to infer grid
            dummy = torch.zeros(1, 1, h, w, device=device)
            cos, sin = self.rope_embeddings(dummy)
            self._pos_cache = (cos, sin)
            self._pos_cache_key = key
        return self._pos_cache

    def forward(self, x: torch.Tensor, h: int = 0, w: int = 0) -> torch.Tensor:
        squeeze = False
        if x.ndim == 4:
            b, t, l, d = x.shape
            x = x.reshape(b * t, l, d)
            squeeze = True
        pos_emb = self._get_position_embeddings(h, w, x.device)
        out = self.layer(x, position_embeddings=pos_emb)
        if isinstance(out, tuple):
            out = out[0]
        if squeeze:
            out = out.reshape(b, t, l, d)
        return out
from ttvidt.modules.patch import TemporalPatch
from ttvidt.modules.pos_embed import get_2d_sincos_pos_embed
from ttvidt.modules.tt3d import RoPE3D
from ttvidt.utils import compile_wrapper
from ttvidt.model.dinov3_vit import VideoModelOutputWithMotionTokens

# ---------------------------------------------------------------------------
# DisMo2DPlus3DConfig
# ---------------------------------------------------------------------------


@dataclass
class DisMo2DPlus3DConfig:
    """Configuration for DisMo2DPlus3DModel."""

    in_channels: int = 3
    hidden_size: int = 768
    intermediate_size: int = 3072  # Phase 1 (DINOv3: 4× = 3072)
    num_attention_heads: int = 12
    spatial_patch_size: int = 1

    # Phase 1 (spatial 2D)
    num_spatial_layers: int = 8

    # Phase 2 (temporal 3D)
    num_temporal_layers: int = 4
    temporal_intermediate_size: int | None = None  # Phase 2 FFN; None = same as intermediate_size

    # Temporal patch
    temporal_patch_size: int = 4
    temporal_patch_overlap: int = 1

    # Motion tokens (only in Phase 2)
    num_motion_tokens: int = 16
    temporal_ffn_type: str = "gelu"  # "gelu" or "swiglu" for Phase 2 blocks

    drop_path_rate: float = 0.0

    @property
    def patch_size(self) -> int:
        """Alias for spatial_patch_size, for compatibility with the trainer."""
        return self.spatial_patch_size


# ---------------------------------------------------------------------------
# DisMo2DPlus3DModel
# ---------------------------------------------------------------------------


class DisMo2DPlus3DModel(nn.Module):
    """
    2D ViT (spatial) + 3D temporal layers for video understanding.

    Phase 1: Per-frame 2D attention (no motion tokens).
    Phase 2: Full spatio-temporal attention with motion tokens and
             periodic TemporalTransfer.

    Can initialise Phase 1 from a pretrained DINOv3 ViT via the
    ``from_dino_v3`` class method.
    """

    def __init__(self, config: DisMo2DPlus3DConfig):
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

        # -- Temporal patch grouping (applied between Phase 1 and Phase 2) ----
        if config.temporal_patch_size > 1:
            self.temporal_patch = TemporalPatch(
                config.hidden_size,
                config.hidden_size,
                config.temporal_patch_size,
                config.temporal_patch_overlap,
            )
        else:
            self.temporal_patch = nn.Identity()

        # =====================================================================
        # Phase 1: Spatial 2D blocks
        # =====================================================================
        self.spatial_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                )
                for _ in range(config.num_spatial_layers)
            ]
        )
        self.spatial_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # =====================================================================
        # Phase 2: Temporal 3D blocks + TemporalTransfer
        # =====================================================================
        # Motion tokens (only used in Phase 2)
        if config.num_motion_tokens > 0:
            self.motion_tokens = nn.Parameter(
                torch.randn(config.num_motion_tokens, config.hidden_size)
                / (config.hidden_size * config.num_motion_tokens) ** 0.5
            )
        else:
            self.motion_tokens = None

        # CLS conditioning from motion tokens
        if config.num_motion_tokens > 1:
            self.motion_pooler = AttentivePooling(
                config.hidden_size, num_heads=config.num_attention_heads
            )
        elif config.num_motion_tokens == 1:
            self.motion_pooler = None  # single token: just squeeze
        else:
            self.motion_pooler = None
        if config.num_motion_tokens > 0:
            self.cls_norm = RMSNorm(config.hidden_size)
        else:
            self.cls_norm = None

        temporal_inter = config.temporal_intermediate_size or config.intermediate_size
        self.temporal_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_size,
                    temporal_inter,
                    config.num_attention_heads,
                    ffn_type=config.temporal_ffn_type,
                )
                for _ in range(config.num_temporal_layers)
            ]
        )

        self.temporal_norm = RMSNorm(config.hidden_size)

        # 3D RoPE for Phase 2 spatio-temporal attention
        head_dim = config.hidden_size // config.num_attention_heads
        self.temporal_rope = RoPE3D(head_dim)
        self._phase2_pos_cache: dict[tuple, torch.Tensor] = {}

        # -- Initialization ----------------------------------------------------
        mup_init(self.temporal_blocks.parameters())
        # Zero-like init for temporal residual output projections
        for block in self.temporal_blocks:
            mup_init_output(block.attn.out_proj.weight)
            if hasattr(block.mlp, 'fc2'):
                mup_init_output(block.mlp.fc2.weight)
            elif hasattr(block.mlp, 'down'):
                mup_init_output(block.mlp.down.weight)

        self.gradient_checkpointing = False

        print(
            f"[DisMo2DPlus3DModel] spatial_layers={config.num_spatial_layers}, "
            f"temporal_layers={config.num_temporal_layers}, "
            f"ffn_type={config.temporal_ffn_type}, "
            f"hidden={config.hidden_size}, heads={config.num_attention_heads}"
        )

    def freeze_pretrained(self, requires_grad: bool = False):
        """Freeze or unfreeze the pretrained backbone components.

        Pretrained components: patch_embed, spatial_blocks, spatial_norm.
        These can be loaded from a DINOv3 checkpoint via ``from_dino_v3``.
        Leaves trainable-from-scratch parts (motion_tokens,
        temporal_blocks, temporal_norm, temporal_patch, temporal_rope)
        untouched.
        """
        self.patch_embed.requires_grad_(requires_grad)
        self.spatial_blocks.requires_grad_(requires_grad)
        self.spatial_norm.requires_grad_(requires_grad)

    # ------------------------------------------------------------------
    # Position embeddings
    # ------------------------------------------------------------------

    def _get_spatial_pos_embed(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """2D sincos spatial position embedding [1, H*W, D]."""
        pos = get_2d_sincos_pos_embed(self.hidden_size, h, w).to(
            device=device, dtype=dtype
        )
        return pos.unsqueeze(0)  # [1, L, D]

    def _get_phase2_3d_positions(
        self, t: int, M: int, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        """
        Build 3D positions for Phase 2 full spatio-temporal attention.

        Per-frame layout: [MT_0..MT_{M-1}, S_(0,0)..S_(w-1,h-1)]
        Motion: (0, 0, t)    Spatial: (x, y, t)

        Returns: [T*(M+h*w), 3]
        """
        key = (t, M, h, w, str(device))
        if key in self._phase2_pos_cache:
            return self._phase2_pos_cache[key]

        L = h * w
        tokens_per_frame = M + L
        positions = torch.zeros(t, tokens_per_frame, 3, device=device)

        for frame_idx in range(t):
            # Motion tokens: (0, 0, t)
            positions[frame_idx, :M, 2] = frame_idx
            # Spatial tokens: (x, y, t)
            idx = M
            for row in range(h):
                for col in range(w):
                    positions[frame_idx, idx, 0] = col
                    positions[frame_idx, idx, 1] = row
                    positions[frame_idx, idx, 2] = frame_idx
                    idx += 1

        positions = positions.reshape(t * tokens_per_frame, 3)
        self._phase2_pos_cache[key] = positions
        return positions

    # ------------------------------------------------------------------
    # from_dino_v3: initialise Phase 1 from a pretrained DINOv3 model
    # ------------------------------------------------------------------

    @classmethod
    def from_dino_v3(
        cls,
        dino_v3_model,
        num_spatial_layers: int = 8,
        num_temporal_layers: int = 12,
        temporal_patch_size: int = 4,
        temporal_patch_overlap: int = 1,
        num_motion_tokens: int = 16,
        temporal_ffn_type: str = "gelu",
        temporal_intermediate_size: int | None = None,
    ):
        """
        Create a DisMo2DPlus3DModel and load Phase 1 spatial block weights
        from a pretrained DINOv3 ViT.

        Args:
            dino_v3_model: A DINOv3ViTModel (from HuggingFace transformers).
            num_spatial_layers: How many DINOv3 layers to use for Phase 1.
            num_temporal_layers: Depth of Phase 2.
            temporal_patch_size: Temporal grouping factor.
            temporal_patch_overlap: Temporal patch overlap.
            num_motion_tokens: Number of motion tokens per frame.
            temporal_ffn_type: FFN type for Phase 2 blocks ("gelu" or "swiglu").
            temporal_intermediate_size: Phase 2 FFN intermediate size (None = same as DINOv3).

        Returns:
            DisMo2DPlus3DModel with Phase 1 weights loaded.
        """
        dino_config = dino_v3_model.config

        config = DisMo2DPlus3DConfig(
            in_channels=dino_config.num_channels,
            hidden_size=dino_config.hidden_size,
            intermediate_size=dino_config.intermediate_size,
            num_attention_heads=dino_config.num_attention_heads,
            spatial_patch_size=dino_config.patch_size,
            num_spatial_layers=num_spatial_layers,
            num_temporal_layers=num_temporal_layers,
            temporal_patch_size=temporal_patch_size,
            temporal_patch_overlap=temporal_patch_overlap,
            num_motion_tokens=num_motion_tokens,
            temporal_ffn_type=temporal_ffn_type,
            temporal_intermediate_size=temporal_intermediate_size,
        )

        model = cls(config)

        # --- Replace Phase 1 spatial blocks with actual DINOv3 layers ---
        # This gives 100% weight compatibility (LayerNorm, LayerScale, GELU MLP).
        dino_layers = list(dino_v3_model.model.layer.children())
        rope_emb = dino_v3_model.rope_embeddings
        n_load = min(num_spatial_layers, len(dino_layers))
        model.spatial_blocks = nn.ModuleList(
            [DINOv3LayerWrapper(dino_layers[i], rope_emb) for i in range(n_load)]
        )
        # Use DINOv3's LayerNorm for spatial_norm
        model.spatial_norm = dino_v3_model.norm

        print(
            f"[DisMo2DPlus3DModel.from_dino_v3] Loaded {n_load}/{num_spatial_layers} "
            f"spatial blocks directly from DINOv3 (full weight transfer)."
        )
        return model

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
            pixel_values: [B, T, C, H, W]
            spatial_mask: optional bool tensor [L], True = visible/keep
                          (for MAE tube masking).  Applied after Phase 1
                          spatial processing and temporal patching, before
                          Phase 2 temporal attention.

        Returns:
            VideoModelOutputWithMotionTokens with:
              - frames_output:  [B, 1+T', L, D]  (or L_vis when spatial_mask)
              - motion_output:  [B, 1+T', M, D]  (or None if M == 0)
              - last_hidden_state: [B, 1+T', M+L, D]
              - pooler_output: [B, 1+T', D]
        """
        b, t_in, c, h_in, w_in = pixel_values.shape

        # 1. Patch embedding: [B, T, C, H, W] -> [B, T, L, D]
        x = self.patch_embed(pixel_values.to(self.spatial_norm.weight.dtype))

        sp = self.config.spatial_patch_size
        h = h_in // sp
        w = w_in // sp
        L_full = h * w

        # =====================================================================
        # Phase 1: Per-frame spatial 2D attention
        # =====================================================================
        # DINOv3LayerWrapper: expects [B*T, 1, L, D], passes h/w for RoPE.
        # Non-DINOv3 (TransformerBlock): uses sincos pos embed, same 4D format.
        _is_dino = isinstance(self.spatial_blocks[0], DINOv3LayerWrapper)

        if _is_dino:
            # DINOv3 layers use RoPE internally — no additive pos embed
            x = x.reshape(b * t_in, 1, L_full, self.hidden_size)
            # Pass patch-grid dims (h, w) scaled by patch_size for RoPE
            # rope_embeddings expects pixel-space [B, C, H, W] to infer grid
            # but grid size = H/patch × W/patch, so pass h_patch*patch, w_patch*patch
            for block in self.spatial_blocks:
                if self.gradient_checkpointing and self.training:
                    x = ckpt.checkpoint(block, x, h_in, w_in, use_reentrant=False)
                else:
                    x = block(x, h_in, w_in)
        else:
            # Our TransformerBlock: add sincos pos embed, use 4D format
            x = x.reshape(b * t_in, 1, L_full, self.hidden_size)
            spatial_pos = self._get_spatial_pos_embed(h, w, x.device, x.dtype)
            x = x + spatial_pos.unsqueeze(1)
            for block in self.spatial_blocks:
                if self.gradient_checkpointing and self.training:
                    x = ckpt.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        x = x.squeeze(1)  # [B*T, L, D]
        x = self.spatial_norm(x)

        # Reshape back: [B*T, L, D] -> [B, T, L, D]
        x = x.reshape(b, t_in, L_full, self.hidden_size)

        # =====================================================================
        # Temporal patch grouping (between Phase 1 and Phase 2)
        # =====================================================================
        x = self.temporal_patch(x)  # [B, 1+T', L, D]
        b, t_out, _, D = x.shape

        # =====================================================================
        # MAE tube masking: keep only visible spatial tokens (after temporal
        # patching so that the same tubes are masked across all frames)
        # =====================================================================
        if spatial_mask is not None:
            # spatial_mask: bool [L], True = keep
            # x: [B, 1+T', L, D] -> [B, 1+T', L_vis, D]
            x = x[:, :, spatial_mask]
        L = x.shape[2]

        M = self.num_motion_tokens

        # =====================================================================
        # Phase 2: Full spatio-temporal attention with motion tokens + 3D RoPE
        # =====================================================================

        # Prepend motion tokens
        if self.motion_tokens is not None:
            mt = self.motion_tokens[None, None].expand(
                b, t_out, -1, -1
            )  # [B, 1+T', M, D]
            x = torch.cat([mt, x], dim=2)  # [B, 1+T', M+L, D]

        total_per_frame = M + L

        # Build 3D positions: motion at (0,0,t), spatial at (x,y,t)
        if spatial_mask is not None:
            # MAE mode: build full positions, then subsample spatial
            full_positions = self._get_phase2_3d_positions(t_out, M, h, w, x.device)
            # full_positions: [T*(M+L_full), 3] — reshape to [T, M+L_full, 3]
            L_full = h * w
            full_positions = full_positions.reshape(t_out, M + L_full, 3)
            # Keep all motion positions, subsample spatial
            motion_pos = full_positions[:, :M, :]
            spatial_pos = full_positions[:, M:, :][:, spatial_mask, :]
            positions = torch.cat([motion_pos, spatial_pos], dim=1)
            positions = positions.reshape(t_out * total_per_frame, 3)
        else:
            positions = self._get_phase2_3d_positions(t_out, M, h, w, x.device)

        # Use [B, 1, (1+T')*(M+L), D] for Attention module compatibility
        x = x.reshape(b, 1, t_out * total_per_frame, D)

        for block in self.temporal_blocks:
            if self.gradient_checkpointing and self.training:
                x = ckpt.checkpoint(
                    block, x, self.temporal_rope, positions, use_reentrant=False
                )
            else:
                x = block(x, rope=self.temporal_rope, positions=positions)

        # Final norm — squeeze dummy dim
        x = x.reshape(b, t_out * total_per_frame, D)
        x = self.temporal_norm(x)

        # =====================================================================
        # Reshape outputs
        # =====================================================================
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
# Helper: map DINOv3 layer state dict to TransformerBlock state dict
# ---------------------------------------------------------------------------


def _map_dino_to_block(src_state: dict, dst_state: dict) -> dict:
    """
    Best-effort mapping of DINOv3 ViT layer weights to TransformerBlock.

    DINOv3 layer typically has:
      attention.{q,k,v,o}_proj.{weight,bias}
      layer_norm1.{weight,bias}
      mlp.fc1.{weight,bias}, mlp.fc2.{weight,bias}
      layer_norm2.{weight,bias}

    Our TransformerBlock has:
      norm1.weight (RMSNorm — no bias)
      attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}
      norm2.weight (RMSNorm — no bias)
      mlp.up.{weight,bias}, mlp.down.{weight,bias}

    We load what matches and skip what doesn't (strict=False upstream).
    """
    key_map = {
        # Attention
        "attention.q_proj.weight": "attn.q_proj.weight",
        "attention.q_proj.bias": "attn.q_proj.bias",
        "attention.k_proj.weight": "attn.k_proj.weight",
        "attention.k_proj.bias": "attn.k_proj.bias",
        "attention.v_proj.weight": "attn.v_proj.weight",
        "attention.v_proj.bias": "attn.v_proj.bias",
        "attention.o_proj.weight": "attn.out_proj.weight",
        "attention.o_proj.bias": "attn.out_proj.bias",
        # Norms (DINOv3 uses LayerNorm, we use RMSNorm — load weight, skip bias)
        "layer_norm1.weight": "norm1.weight",
        "layer_norm2.weight": "norm2.weight",
    }
    # MLP: DINOv3 uses standard 2-layer MLP (fc1, fc2).
    # Our SwiGLU has up (hidden->2*inter) and down (inter->hidden).
    # Shapes differ, so we skip MLP weights — they will be randomly initialised.

    mapped = {}
    for src_key, dst_key in key_map.items():
        if src_key in src_state and dst_key in dst_state:
            src_val = src_state[src_key]
            dst_val = dst_state[dst_key]
            if src_val.shape == dst_val.shape:
                mapped[dst_key] = src_val

    return mapped if mapped else {}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DisMo2DPlus3DModel smoke test")
    print("=" * 60)

    cfg = DisMo2DPlus3DConfig(
        in_channels=3,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        spatial_patch_size=16,
        num_spatial_layers=8,
        num_temporal_layers=4,
        temporal_patch_size=4,
        temporal_patch_overlap=1,
        num_motion_tokens=8,
        motion_layers_period=2,
    )
    model = DisMo2DPlus3DModel(cfg)

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
