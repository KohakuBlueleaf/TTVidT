import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from optimfactory import mup_init, mup_init_output

from ttvidt.modules.layers import Attention, SwiGLU, RMSNorm, AdaRMSNorm, DiTTransformerBlock
from ttvidt.modules.pos_embed import (
    TemporalDistanceEmbedding,
    AdditiveRoPE2D,
    RoPE2DAttention,
)

# ============================================================================
# Timestep Embedding (for diffusion)
# ============================================================================


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding following additive RoPE idea.

    Uses sincos positional encoding with t * 1000 as position:
        output = proj(learnable_token * sincos(t * 1000))
    """

    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period

        # Learnable token
        self.learnable_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)

        # Projection after rope modulation
        self.proj = nn.Linear(hidden_size, hidden_size)

        # Precompute frequency bands for sincos encoding
        half_dim = hidden_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timesteps (normalized to [0, 1] or integer steps)

        Returns:
            [B, D] timestep embeddings
        """
        # Scale timestep
        t_scaled = t.float() * 1000  # [B]

        # Sincos encoding: [B, D/2] -> [B, D]
        args = t_scaled.unsqueeze(-1) * self.freqs.to(t.device)  # [B, D/2]
        sincos = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, D]

        # Modulate learnable token and project
        out = self.learnable_token * sincos  # [B, D]
        out = self.proj(out)

        return out


# ============================================================================
# Transformer Blocks
# ============================================================================


class BasicTransformer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads):
        super(BasicTransformer, self).__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RoPETransformer(nn.Module):
    """Transformer block with 2D RoPE attention (adaptive positions)."""

    def __init__(self, hidden_size, intermediate_size, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPE2DAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x))
        return x


class MLPTower(nn.Module):
    def __init__(self, hidden_size, num_layers, intermediate_size):
        super(MLPTower, self).__init__()
        self.mlps = nn.ModuleList(
            [SwiGLU(hidden_size, intermediate_size) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(num_layers)])

    def forward(self, x, grad_ckpt=False):
        for i in range(len(self.mlps)):
            if grad_ckpt:
                x = x + ckpt.checkpoint(
                    self.mlps[i], self.norms[i](x), use_reentrant=False
                )
            else:
                x = x + self.mlps[i](self.norms[i](x))
        return x


# ============================================================================
# Decode Heads
# ============================================================================


class SmoothBlock(nn.Module):
    """
    Lightweight smoothing block with depthwise separable convolutions.

    Architecture: x = x + mlp(spatial_dw(temporal_dw(norm(x))))

    Much cheaper than full Conv3d:
    - Temporal DW Conv: (3,1,1) with groups=channels
    - Spatial DW Conv: (1,3,3) with groups=channels
    - Small MLP: channel mixing with expansion factor (uses Linear to avoid DDP stride warnings)
    """

    def __init__(self, channels: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.channels = channels
        mlp_hidden = int(channels * mlp_ratio)

        # LayerNorm (group=1)
        self.norm = nn.GroupNorm(1, channels)

        # Temporal depthwise conv: (3,1,1)
        self.temporal_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
        )

        # Spatial depthwise conv: (1,3,3)
        self.spatial_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
        )

        # Small MLP for channel mixing (Linear avoids Conv3d stride issues with DDP)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.Mish(),
            nn.Linear(mlp_hidden, channels),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize MLP output to zero for residual
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W]"""
        # x = x + mlp(spatial_dw(temporal_dw(norm(x))))
        # Conv3d expects [B, C, T, H, W], Linear expects [..., C]
        h = self.spatial_dw(self.temporal_dw(self.norm(x)))
        h = h.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        h = self.mlp(h)
        h = h.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        return x + h


class LightweightDecodeHead(nn.Module):
    """
    Lightweight decode head for frame reconstruction with staged upsampling.

    Each stage: SmoothBlocks -> Upsample (spatial & temporal)

    Stage config: (spatial_up, temporal_up, num_blocks, dim_factor)
    - spatial_up: spatial upsample factor for this stage
    - temporal_up: temporal upsample factor for this stage
    - num_blocks: number of SmoothBlocks before upsampling
    - dim_factor: channel dimension as factor of hidden_size

    Total spatial upsample = product of all spatial_up = patch_size
    Total temporal upsample = product of all temporal_up = temporal_patch_size
    """

    def __init__(
        self,
        hidden_size: int,
        temporal_patch_size: int,
        patch_size: int,
        image_dim: int = 3,
        stages: list[tuple[int, int, int, float]] | None = None,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.image_dim = image_dim

        # Default stages if not provided
        # For patch_size=14, temporal_patch_size=6:
        # Stage 0: 7x spatial, 3x temporal, 2 blocks, 1.0x dim
        # Stage 1: 2x spatial, 2x temporal, 1 block, 0.25x dim
        if stages is None:
            # Try to factorize patch_size and temporal_patch_size
            stages = self._default_stages(patch_size, temporal_patch_size)

        self.stages = stages

        # Validate total upsample matches patch sizes
        total_spatial = 1
        total_temporal = 1
        for s_up, t_up, _, _ in stages:
            total_spatial *= s_up
            total_temporal *= t_up
        assert (
            total_spatial == patch_size
        ), f"Spatial upsample {total_spatial} != patch_size {patch_size}"
        assert (
            total_temporal == temporal_patch_size
        ), f"Temporal upsample {total_temporal} != temporal_patch_size {temporal_patch_size}"

        # Build stages
        self.stage_blocks = nn.ModuleList()
        self.stage_upsample = nn.ModuleList()

        prev_dim = hidden_size
        for i, (s_up, t_up, num_blocks, dim_factor) in enumerate(stages):
            curr_dim = int(hidden_size * dim_factor)

            # Blocks for this stage
            blocks = nn.ModuleList(
                [SmoothBlock(prev_dim, mlp_ratio) for _ in range(num_blocks)]
            )
            self.stage_blocks.append(blocks)

            # Upsample: Linear to next_dim * s_up * s_up * t_up
            # Then reshape to upsample spatial and temporal
            # Using nn.Linear to avoid DDP stride warnings from Conv3d(kernel_size=1)
            next_dim = (
                int(hidden_size * stages[i + 1][3]) if i + 1 < len(stages) else curr_dim
            )
            upsample_out = next_dim * s_up * s_up * t_up
            self.stage_upsample.append(nn.Linear(prev_dim, upsample_out))

            prev_dim = next_dim

        # Final projection to RGB (using Linear to avoid DDP stride warnings)
        final_dim = int(hidden_size * stages[-1][3])
        self.to_rgb = nn.Linear(final_dim, image_dim)

        self._init_weights()

    def _default_stages(self, patch_size: int, temporal_patch_size: int):
        """Generate default stages based on patch sizes."""
        # Simple factorization
        # Example: patch_size=14 -> 7*2, temporal_patch_size=6 -> 3*2
        # Example: patch_size=16 -> 4*4, temporal_patch_size=4 -> 2*2

        def factorize(n):
            """Simple factorization into ~2 factors."""
            for i in range(int(n**0.5), 0, -1):
                if n % i == 0:
                    return (n // i, i)
            return (n, 1)

        s1, s2 = factorize(patch_size)
        t1, t2 = factorize(temporal_patch_size)

        return [
            (s1, t1, 2, 1.0),  # First stage: more blocks, full dim
            (s2, t2, 1, 0.25),  # Second stage: fewer blocks, smaller dim
        ]

    def _init_weights(self):
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: [B, T, H*W, D] decoder output (T = num temporal patches)
            h, w: spatial dimensions (in patches)

        Returns:
            frames: [B, T*tp, C, H*p, W*p]
        """
        B, T, L, D = x.shape

        # Reshape to [B, C, T, H, W] for 3D convolutions
        x = x.view(B, T, h, w, D)
        x = x.permute(0, 4, 1, 2, 3)  # [B, D, T, H, W]

        curr_t, curr_h, curr_w = T, h, w

        # Process each stage
        for i, (s_up, t_up, _, dim_factor) in enumerate(self.stages):
            # Apply blocks (SmoothBlock expects [B, C, T, H, W])
            for block in self.stage_blocks[i]:
                x = block(x)

            # Upsample using Linear
            B, C, curr_t, curr_h, curr_w = x.shape

            # Linear expects [..., C], so permute: [B, C, T, H, W] -> [B, T, H, W, C]
            x = x.permute(0, 2, 3, 4, 1)
            x = self.stage_upsample[i](
                x
            )  # [B, T, H, W, C'] where C' = next_dim * s_up * s_up * t_up

            # Reshape to upsample spatial and temporal
            # [B, T, H, W, next_dim * t_up * s_up * s_up]
            # -> [B, T, H, W, next_dim, t_up, s_up, s_up]
            # -> [B, next_dim, T * t_up, H * s_up, W * s_up]
            next_dim = x.shape[-1] // (s_up * s_up * t_up)
            x = x.view(B, curr_t, curr_h, curr_w, next_dim, t_up, s_up, s_up)
            x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [B, C, T, t_up, H, s_up, W, s_up]
            x = x.reshape(B, next_dim, curr_t * t_up, curr_h * s_up, curr_w * s_up)

            curr_t *= t_up
            curr_h *= s_up
            curr_w *= s_up

        # Final projection to RGB using Linear
        # [B, C, T, H, W] -> [B, T, H, W, C] -> Linear -> [B, T, H, W, 3]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.to_rgb(x)
        x = x.permute(0, 1, 4, 2, 3)  # [B, T*tp, 3, H*p, W*p]

        return x


# Legacy alias
SmoothDecodeHead = LightweightDecodeHead


# ============================================================================
# Video Patchify (for diffusion)
# ============================================================================


def patchify_video(
    x: torch.Tensor, patch_size: int, temporal_patch_size: int
) -> torch.Tensor:
    """
    Patchify video tensor for diffusion decoder input.

    Args:
        x: [B, T*tp, C, H, W] video tensor
        patch_size: spatial patch size (p)
        temporal_patch_size: temporal patch size (tp)

    Returns:
        [B, T, L, C*tp*p*p] patchified tensor where L = (H/p) * (W/p)
    """
    B, T_total, C, H, W = x.shape
    T = T_total // temporal_patch_size
    tp = temporal_patch_size
    p = patch_size
    h, w = H // p, W // p

    # [B, T*tp, C, H, W] -> [B, T, tp, C, H, W]
    x = x.reshape(B, T, tp, C, H, W)

    # [B, T, tp, C, H, W] -> [B*T*tp, C, H, W] for pixel_unshuffle
    x = x.reshape(B * T * tp, C, H, W)

    # pixel_unshuffle: [B*T*tp, C, H, W] -> [B*T*tp, C*p*p, h, w]
    x = F.pixel_unshuffle(x, p)

    # [B*T*tp, C*p*p, h, w] -> [B, T, tp, C*p*p, h, w]
    x = x.reshape(B, T, tp, C * p * p, h, w)

    # [B, T, tp, C*p*p, h, w] -> [B, T, h, w, tp, C*p*p]
    x = x.permute(0, 1, 4, 5, 2, 3)

    # [B, T, h, w, tp, C*p*p] -> [B, T, h*w, tp*C*p*p]
    x = x.reshape(B, T, h * w, tp * C * p * p)

    return x


# ============================================================================
# Motion Decoder
# ============================================================================


class MotionDecoder(nn.Module):
    """
    Motion decoder for frame reconstruction.

    Supports two decoder styles:
    - "concat" (default): original concat-based approach with BasicTransformer layers
    - "dit": DiT-style with AdaRMSNorm conditioning and optional cross-attention

    input:
        - img_emb: [batch_size, (H*W/patch_size**2), encoder_hidden_size]
        - motion_emb: [batch_size, frame_num, num_motion_token, encoder_hidden_size]
    output:
        - frame_recon: [batch_size, frame_num * temporal_patch_size, 3, H, W]
    """

    def __init__(
        self,
        image_dim: int,
        patch_size: int,
        temporal_patch_size: int,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        # New features
        encoder_hidden_size: int | None = None,
        # Positional encoding options
        use_temporal_distance_token: bool = False,
        use_spatial_rope: bool = False,
        rope_type: str = "additive",
        # MLP tower config
        mlp_tower_depth: int | None = None,
        # Decode head options
        to_image: bool = True,
        use_smooth_head: bool = False,
        decode_stages: (
            list[tuple[int, int, int, float]] | None
        ) = None,
        decode_mlp_ratio: float = 2.0,
        # Decode mode
        decode_mode: str = "regression",
        # DiT-style decoder
        decoder_style: str = "concat",  # "concat" or "dit"
        cond_size: int | None = None,  # conditioning dim for DiT (defaults to hidden_size)
        qk_norm: bool = False,  # QK-norm (cosine-sim attention) for decoder self-attention
        attn_bias: bool = True,  # bias in attention Q/K/V/out projections
        use_final_norm: bool = True,  # AdaRMSNorm before unpatch (False = direct gradient flow)
        # Frozen source encoder for cross-attention context (DisMo-style)
        source_encoder_name: str | None = None,  # e.g. "facebook/dinov3-vitb16-pretrain-lvd1689m"
    ):
        super(MotionDecoder, self).__init__()
        self.image_dim = image_dim
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size or hidden_size
        self.use_smooth_head = use_smooth_head
        self.use_temporal_distance_token = use_temporal_distance_token
        self.use_spatial_rope = use_spatial_rope
        self.rope_type = rope_type
        self.to_image = to_image
        self.decode_stages = decode_stages
        self.decode_mlp_ratio = decode_mlp_ratio
        self.decode_mode = decode_mode
        self.decoder_style = decoder_style

        if decoder_style == "dit":
            self._init_dit(
                image_dim, patch_size, temporal_patch_size, num_layers,
                hidden_size, intermediate_size, num_heads,
                cond_size=cond_size, mlp_tower_depth=mlp_tower_depth,
                use_smooth_head=use_smooth_head, decode_stages=decode_stages,
                decode_mlp_ratio=decode_mlp_ratio,
                source_encoder_name=source_encoder_name,
                qk_norm=qk_norm, attn_bias=attn_bias,
                use_final_norm=use_final_norm,
            )
        else:
            self._init_concat(
                image_dim, patch_size, temporal_patch_size, num_layers,
                hidden_size, intermediate_size, num_heads,
                use_spatial_rope=use_spatial_rope, rope_type=rope_type,
                mlp_tower_depth=mlp_tower_depth, use_smooth_head=use_smooth_head,
                decode_stages=decode_stages, decode_mlp_ratio=decode_mlp_ratio,
            )

        self.gradient_checkpointing = False
        self.init_weight()

        # Log decoder configuration
        proj_info = ""
        if self.encoder_hidden_size != hidden_size:
            proj_info = f", encoder_hidden_size={self.encoder_hidden_size} (with projection)"
        print(
            f"[MotionDecoder] style={decoder_style}, hidden={hidden_size}, inter={intermediate_size}, "
            f"heads={num_heads}, layers={num_layers}{proj_info}"
        )

    # ================================================================
    # Concat-style init (original)
    # ================================================================

    def _init_concat(
        self, image_dim, patch_size, temporal_patch_size, num_layers,
        hidden_size, intermediate_size, num_heads, *,
        use_spatial_rope, rope_type, mlp_tower_depth, use_smooth_head,
        decode_stages, decode_mlp_ratio,
    ):
        if self.decode_mode == "regression":
            img_hidden_size = self.encoder_hidden_size
        elif self.decode_mode == "diffusion":
            img_hidden_size = (
                self.encoder_hidden_size
                + image_dim * patch_size * patch_size * temporal_patch_size
            )
        else:
            raise ValueError(f"Unknown decode mode: {self.decode_mode}")

        if img_hidden_size != hidden_size:
            self.img_proj = nn.Linear(img_hidden_size, hidden_size)
        else:
            self.img_proj = nn.Identity()

        if self.encoder_hidden_size != hidden_size:
            self.motion_proj = nn.Linear(self.encoder_hidden_size, hidden_size)
        else:
            self.motion_proj = nn.Identity()

        if self.use_temporal_distance_token:
            print("Deprecated: temporal distance token is no longer used")

        if use_spatial_rope and rope_type == "additive":
            self.spatial_rope = AdditiveRoPE2D(hidden_size)
        else:
            self.spatial_rope = None

        if self.decode_mode == "diffusion":
            self.timestep_emb = TimestepEmbedding(hidden_size)
        else:
            self.timestep_emb = None

        if use_spatial_rope and rope_type == "rotary":
            self.layers = nn.ModuleList(
                [RoPETransformer(hidden_size, intermediate_size, num_heads)
                 for _ in range(num_layers)]
            )
            self.use_rope_attention = True
        else:
            self.layers = nn.ModuleList(
                [BasicTransformer(hidden_size, intermediate_size, num_heads)
                 for _ in range(num_layers)]
            )
            self.use_rope_attention = False

        if mlp_tower_depth is None:
            mlp_tower_depth = num_layers
        if mlp_tower_depth > 0:
            mlp_tower_depth = mlp_tower_depth or num_layers
            self.mlp_tower = MLPTower(hidden_size, num_layers, intermediate_size)
        else:
            self.mlp_tower = None

        self._init_output_head(
            hidden_size, temporal_patch_size, patch_size, image_dim,
            use_smooth_head, decode_stages, decode_mlp_ratio,
        )

    # ================================================================
    # DiT-style init (new)
    # ================================================================

    def _init_dit(
        self, image_dim, patch_size, temporal_patch_size, num_layers,
        hidden_size, intermediate_size, num_heads, *,
        cond_size, mlp_tower_depth, use_smooth_head,
        decode_stages, decode_mlp_ratio, source_encoder_name=None,
        qk_norm=False, attn_bias=True, use_final_norm=True,
    ):
        cond_size = cond_size or hidden_size
        self._dit_cond_size = cond_size
        patch_dim = image_dim * patch_size * patch_size * temporal_patch_size

        # Conditioning projection: encoder_hidden -> cond_size
        if self.encoder_hidden_size != cond_size:
            self.cond_proj = nn.Linear(self.encoder_hidden_size, cond_size)
        else:
            self.cond_proj = nn.Identity()

        # Timestep embedding for diffusion
        if self.decode_mode == "diffusion":
            self.timestep_emb = TimestepEmbedding(cond_size)
        else:
            self.timestep_emb = None

        # Input projections
        self.motion_proj = nn.Linear(self.encoder_hidden_size, hidden_size)
        self.img_proj = nn.Linear(self.encoder_hidden_size, hidden_size)

        # For diffusion: separate xt patch projection
        if self.decode_mode == "diffusion":
            self.xt_proj = nn.Linear(patch_dim, hidden_size)
            # Mask channel projection for mae-diffusion (1 binary channel per patch)
            mask_patch_dim = patch_size * patch_size * temporal_patch_size
            self.mask_proj = nn.Linear(mask_patch_dim, hidden_size)
            nn.init.zeros_(self.mask_proj.weight)
            nn.init.zeros_(self.mask_proj.bias)
        else:
            self.xt_proj = None
            self.mask_proj = None

        # Frozen source encoder for cross-attention context (DisMo-style)
        # When enabled, decoder receives raw source frames and encodes them
        # internally with a frozen pretrained DINOv3, instead of using
        # encoder features as cross-attention context.
        self.source_encoder_name = source_encoder_name
        if source_encoder_name is not None:
            from transformers import DINOv3ViTModel
            self.source_encoder = DINOv3ViTModel.from_pretrained(
                source_encoder_name, attn_implementation="sdpa"
            ).eval().requires_grad_(False)
            source_dim = self.source_encoder.config.hidden_size
            self.context_proj = nn.Linear(source_dim, hidden_size)
            print(f"[MotionDecoder] Frozen source encoder: {source_encoder_name} (dim={source_dim})")
        else:
            self.source_encoder = None
            # Context projection for cross-attention K/V (source frame features from our encoder)
            self.context_proj = nn.Linear(self.encoder_hidden_size, hidden_size)

        # DiT transformer blocks with 2D RoPE self-attention + cross-attn
        self.layers = nn.ModuleList(
            [DiTTransformerBlock(
                hidden_size, intermediate_size, num_heads, cond_size,
                use_cross_attn=True, context_size=hidden_size,
                use_rope=True, qk_norm=qk_norm, bias=attn_bias,
            ) for _ in range(num_layers)]
        )
        self.use_rope_attention = True

        # Final norm — unconditional RMSNorm (DisMo-style), not AdaRMSNorm
        # Conditional final norm creates gradient path from loss directly to cls_emb,
        # allowing unbounded hidden states (norm rescales before output).
        # Unconditional norm just normalizes magnitude without conditioning amplification.
        if use_final_norm:
            self.final_norm = RMSNorm(hidden_size)
        else:
            self.final_norm = None

        # MLP tower (optional, AIM-style — only when explicitly requested)
        if mlp_tower_depth is not None and mlp_tower_depth > 0:
            self.mlp_tower = MLPTower(hidden_size, mlp_tower_depth, intermediate_size)
        else:
            self.mlp_tower = None

        self.spatial_rope = None

        self._init_output_head(
            hidden_size, temporal_patch_size, patch_size, image_dim,
            use_smooth_head, decode_stages, decode_mlp_ratio,
        )

    # ================================================================
    # Shared output head init
    # ================================================================

    def _init_output_head(
        self, hidden_size, temporal_patch_size, patch_size, image_dim,
        use_smooth_head, decode_stages, decode_mlp_ratio,
    ):
        if use_smooth_head:
            print("Using smooth decode head")
            self.decode_head = LightweightDecodeHead(
                hidden_size, temporal_patch_size, patch_size, image_dim,
                stages=decode_stages, mlp_ratio=decode_mlp_ratio,
            )
            self.unpatch = None
        else:
            self.unpatch = nn.Linear(
                hidden_size, image_dim * temporal_patch_size * patch_size**2
            )
            self.decode_head = None

    def init_weight(self):
        mup_init(self.parameters())
        # Near-zero output at start
        if self.unpatch is not None:
            mup_init_output(self.unpatch.weight)
        else:
            mup_init_output(self.decode_head.to_rgb.weight)
        # Restore special inits clobbered by mup_init (which overwrites all ≥2D params)
        if hasattr(self, 'mask_proj') and self.mask_proj is not None:
            nn.init.zeros_(self.mask_proj.weight)
            nn.init.zeros_(self.mask_proj.bias)
        for name, module in self.named_modules():
            # QK-norm scale: init=10.0 (cosine-sim temperature)
            if hasattr(module, 'qk_scale'):
                module.qk_scale.data.fill_(10.0)
            # AdaRMSNorm: zero-init linear so modulation starts as identity (scale=1)
            if isinstance(module, AdaRMSNorm):
                nn.init.zeros_(module.linear.weight)
            # Near-zero residual output projections (attn.out_proj, mlp.down/fc2)
            if hasattr(module, 'out_proj') and hasattr(module, 'q_proj'):  # any attention module
                mup_init_output(module.out_proj.weight)
            if hasattr(module, 'down') and hasattr(module, 'up'):  # SwiGLU
                mup_init_output(module.down.weight)
            if hasattr(module, 'fc2') and hasattr(module, 'fc1'):  # GELUMLP
                mup_init_output(module.fc2.weight)

    # ================================================================
    # Forward dispatch
    # ================================================================

    def forward(
        self,
        frame_emb: torch.Tensor,
        motion_emb: torch.Tensor,
        h: int | None = None,
        w: int | None = None,
        xt: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        source_frame_emb: torch.Tensor | None = None,
        source_frames: torch.Tensor | None = None,
        xt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            frame_emb: [B, T, L, D_enc] reference frame patch embeddings
            motion_emb: [B, T, M, D_enc] motion tokens
            h, w: spatial dimensions (in patches), inferred if not provided
            xt: [B, T*tp, C, H, W] noisy video for diffusion mode (optional)
            t: [B] timesteps for diffusion mode (optional)
            cls_emb: [B, T, D_enc] CLS-like conditioning from encoder attentive pooling (dit only)
            source_frame_emb: [B, T, L, D_enc] source frame features for cross-attn (dit only)
            source_frames: [B, T, C, H, W] raw source frames for frozen source encoder (dit only)
            xt_mask: [B, T*tp, 1, H, W] binary mask for mae-diffusion (optional)
        Returns:
            frame_recon: [B, T*tp, C, H*p, W*p]
        """
        if self.decoder_style == "dit":
            return self._forward_dit(
                frame_emb, motion_emb, h, w, xt, t, cls_emb,
                source_frame_emb, source_frames, xt_mask,
            )
        else:
            return self._forward_concat(frame_emb, motion_emb, h, w, xt, t)

    # ================================================================
    # Concat forward (original behavior)
    # ================================================================

    def _forward_concat(
        self,
        frame_emb: torch.Tensor,
        motion_emb: torch.Tensor,
        h: int | None = None,
        w: int | None = None,
        xt: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t_frames, l, d = frame_emb.shape
        b, t_frames, m, _ = motion_emb.shape

        h, w = self._infer_hw(l, h, w)

        # Diffusion mode: concat xt patches with frame_emb
        if self.decode_mode == "diffusion":
            assert (
                xt is not None and t is not None
            ), "xt and t required for diffusion mode"
            xt_patches = patchify_video(xt, self.patch_size, self.temporal_patch_size)
            frame_input = torch.cat([frame_emb, xt_patches], dim=-1)
            frame_emb = self.img_proj(frame_input)
            t_token = self.timestep_emb(t).unsqueeze(1).unsqueeze(2)
            t_token = t_token.expand(b, t_frames, 1, -1)
        else:
            frame_emb = self.img_proj(frame_emb)
            t_token = None

        motion_emb = self.motion_proj(motion_emb)

        x = torch.cat([frame_emb, motion_emb], dim=2)
        if t_token is not None:
            x = torch.cat([x, t_token], dim=2)

        if self.spatial_rope is not None:
            frame_part = x[:, :, :l]
            frame_part = self.spatial_rope(frame_part, h, w)
            x = torch.cat([frame_part, x[:, :, l:]], dim=2)

        if self.use_rope_attention:
            for layer in self.layers:
                if self.gradient_checkpointing:
                    x = ckpt.checkpoint(layer, x, h, w, use_reentrant=False)
                else:
                    x = layer(x, h, w)
        else:
            for layer in self.layers:
                if self.gradient_checkpointing:
                    x = ckpt.checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)

        x = x[:, :, :l]
        if self.mlp_tower:
            x = self.mlp_tower(x, grad_ckpt=self.gradient_checkpointing)

        return self._decode_to_pixels(x, h, w)

    # ================================================================
    # DiT forward (new)
    # ================================================================

    def _forward_dit(
        self,
        frame_emb: torch.Tensor,
        motion_emb: torch.Tensor,
        h: int | None = None,
        w: int | None = None,
        xt: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        source_frame_emb: torch.Tensor | None = None,
        source_frames: torch.Tensor | None = None,
        xt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t_frames, l, d = frame_emb.shape
        h, w = self._infer_hw(l, h, w)

        # Build conditioning signal: c = cond_proj(cls_emb) [+ timestep_emb(t)]
        if cls_emb is not None:
            c = self.cond_proj(cls_emb)  # [B, T, cond_size]
        else:
            # Fallback: zero conditioning
            c = torch.zeros(
                b, t_frames, self._dit_cond_size,
                device=frame_emb.device, dtype=frame_emb.dtype,
            )

        if self.decode_mode == "diffusion":
            assert xt is not None and t is not None, "xt and t required for diffusion mode"
            t_emb = self.timestep_emb(t)  # [B, cond_size]
            c = c + t_emb.unsqueeze(1)  # [B, T, cond_size]

        # Expand c for broadcast: [B, T, 1, cond_size]
        c = c.unsqueeze(2)

        # Build main stream
        if self.decode_mode == "diffusion":
            xt_patches = patchify_video(xt, self.patch_size, self.temporal_patch_size)
            # Project xt patches and frame_emb separately, then add
            frame_tokens = self.img_proj(frame_emb) + self.xt_proj(xt_patches)  # [B, T, L, D]
            if xt_mask is not None and self.mask_proj is not None:
                mask_patches = patchify_video(xt_mask, self.patch_size, self.temporal_patch_size)
                frame_tokens = frame_tokens + self.mask_proj(mask_patches)
        else:
            frame_tokens = self.img_proj(frame_emb)  # [B, T, L, D]

        motion_tokens = self.motion_proj(motion_emb)  # [B, T, M, D]
        x = torch.cat([frame_tokens, motion_tokens], dim=2)  # [B, T, L+M, D]

        # Build cross-attention context
        if self.source_encoder is not None and source_frames is not None:
            # Frozen DINOv3 encodes raw source frames → cross-attention context
            # source_frames: [B, T, C, H, W] or [B, C, H, W]
            with torch.no_grad():
                if source_frames.ndim == 5:
                    sf_b, sf_t = source_frames.shape[:2]
                    sf_flat = source_frames.reshape(sf_b * sf_t, *source_frames.shape[2:])
                else:
                    sf_flat = source_frames
                    sf_t = 1
                src_feats = self.source_encoder(sf_flat).last_hidden_state  # [B*T, L_src, D_src]
                src_feats = src_feats.reshape(b, sf_t, src_feats.shape[1], src_feats.shape[2])
                if sf_t == 1 and t_frames > 1:
                    src_feats = src_feats.expand(-1, t_frames, -1, -1)
            context = self.context_proj(src_feats.to(frame_emb.dtype))  # [B, T, L_src, D]
        elif source_frame_emb is not None:
            context = self.context_proj(source_frame_emb)  # [B, T, L, D]
        else:
            context = None

        # DiT transformer blocks (with 2D RoPE spatial positions)
        for layer in self.layers:
            if self.gradient_checkpointing:
                x = ckpt.checkpoint(layer, x, c, context, h, w, use_reentrant=False)
            else:
                x = layer(x, c, context, h, w)

        # Extract frame patch outputs (first L tokens)
        x = x[:, :, :l]

        # MLP tower (regression only)
        if self.mlp_tower:
            x = self.mlp_tower(x, grad_ckpt=self.gradient_checkpointing)

        # Final RMSNorm (optional, unconditional)
        if self.final_norm is not None:
            x = self.final_norm(x)

        return self._decode_to_pixels(x, h, w)

    # ================================================================
    # Shared helpers
    # ================================================================

    @staticmethod
    def _infer_hw(l: int, h: int | None, w: int | None) -> tuple[int, int]:
        if h is None and w is None:
            assert l**0.5 % 1 == 0
            h = w = int(l**0.5)
        elif h is None:
            assert l % w == 0
            h = l // w
        elif w is None:
            assert l % h == 0
            w = l // h
        else:
            assert l == h * w
        return h, w

    def _decode_to_pixels(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if self.use_smooth_head and self.decode_head is not None:
            return self.decode_head(x, h, w)

        frame_recon = self.unpatch(x)
        if not self.to_image:
            return frame_recon
        frame_recon = frame_recon.unflatten(
            -1,
            (
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
                self.image_dim,
            ),
        )
        frame_recon = frame_recon.unflatten(2, (h, w))
        frame_recon = (
            frame_recon.permute(0, 1, 4, 7, 2, 5, 3, 6)
            .flatten(1, 2)
            .flatten(3, 4)
            .flatten(4, 5)
        )
        return frame_recon
