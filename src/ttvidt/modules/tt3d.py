"""
TemporalTransfer3D: Temporal attention with downsampled spatial context and 3D RoPE.

Each frame contributes M motion tokens + S downsampled spatial tokens.
Block-causal attention across time with 3D RoPE (x, y, t).
Both motion and spatial tokens receive attention residual.
Spatial tokens are downsampled via pixel unshuffle + linear,
and upsampled back via linear + pixel shuffle for residual add-back.

3D RoPE head_dim split: [x, y, t, unused] with 1/4 each.
Motion tokens use position (0, 0, t) — no spatial, only temporal.
Spatial tokens use position (x, y, t) — full 3D.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttvidt.utils import compile_wrapper
from ttvidt.modules.layers import SwiGLU, GELUMLP, RMSNorm


# =============================================================================
# Block-causal mask (cached)
# =============================================================================

_mask_cache: dict[tuple, torch.Tensor] = {}


def get_block_causal_mask(t: int, tokens_per_frame: int, device: torch.device) -> torch.Tensor:
    """Bool mask [T*N, T*N] where frame i attends to frames 0..i."""
    key = (t, tokens_per_frame, str(device))
    if key not in _mask_cache:
        causal = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool))
        n = tokens_per_frame
        block = causal[:, :, None, None].expand(-1, -1, n, n)
        mask = block.permute(0, 2, 1, 3).reshape(t * n, t * n)
        _mask_cache[key] = mask
    return _mask_cache[key]


# =============================================================================
# 3D RoPE
# =============================================================================

def _compute_freqs(dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Frequency bands for RoPE. Returns [dim//2]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    )
    return freqs


def _apply_rope_1d(x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to a slice of x along the last dim.

    Args:
        x: [..., dim] — the slice of q or k to rotate
        freqs: [dim//2] — precomputed frequency bands
        positions: [...] — positions for each token

    Returns:
        [..., dim] rotated tensor
    """
    half = x.shape[-1] // 2
    angles = positions.unsqueeze(-1).float() * freqs.to(x.device)
    cos = torch.cos(angles).to(x.dtype)
    sin = torch.sin(angles).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding for (x, y, t) positions.

    Head dim split into 4 equal parts: [x_rope, y_rope, t_rope, unused].
    The unused portion passes through unchanged (identity).
    """

    def __init__(self, head_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.dim_x = head_dim // 4
        self.dim_y = head_dim // 4
        self.dim_t = head_dim // 4
        self.dim_unused = head_dim - self.dim_x - self.dim_y - self.dim_t

        self.register_buffer("freqs_x", _compute_freqs(self.dim_x, max_period), persistent=False)
        self.register_buffer("freqs_y", _compute_freqs(self.dim_y, max_period), persistent=False)
        self.register_buffer("freqs_t", _compute_freqs(self.dim_t, max_period), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """
        Args:
            q, k: [B, H, S, head_dim]
            positions: [S, 3] — (x, y, t) per token
        Returns:
            q_rot, k_rot: same shape
        """
        pos_x = positions[:, 0]
        pos_y = positions[:, 1]
        pos_t = positions[:, 2]

        def apply(tensor):
            t_x = tensor[..., :self.dim_x]
            t_y = tensor[..., self.dim_x:self.dim_x + self.dim_y]
            t_t = tensor[..., self.dim_x + self.dim_y:self.dim_x + self.dim_y + self.dim_t]
            t_u = tensor[..., self.dim_x + self.dim_y + self.dim_t:]

            t_x = _apply_rope_1d(t_x, self.freqs_x, pos_x)
            t_y = _apply_rope_1d(t_y, self.freqs_y, pos_y)
            t_t = _apply_rope_1d(t_t, self.freqs_t, pos_t)

            return torch.cat([t_x, t_y, t_t, t_u], dim=-1)

        return apply(q), apply(k)


# =============================================================================
# Pixel unshuffle/shuffle spatial resampling
# =============================================================================

class SpatialDownsample(nn.Module):
    """
    Depth-separable downsample: spatial pool (f² → 1, channel-shared) then
    channel mix (D → D).

    Flow: [B, T, L, D] → pixel_unshuffle → [BT, D*f², H', W']
          → reshape to [BT, H'W', D, f²]
          → matmul with [f²] spatial-mix weight → [BT, H'W', D]
          → Linear(D, D) channel mix → [B, T, H'W', D]

    Params: f² (spatial pool) + D² (channel mix), instead of D²·f² for the
    full Linear(D·f² → D).
    """

    def __init__(self, hidden_size: int, factor: int, depthwise: bool = True):
        super().__init__()
        self.factor = factor
        self.depthwise = depthwise
        f2 = factor * factor
        if depthwise:
            self.spatial_mix = nn.Parameter(torch.full((f2,), 1.0 / f2))  # [f²]
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:  # full resample: pixel-unshuffle -> Linear(D*f^2 -> D)
            self.proj = nn.Linear(hidden_size * f2, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: [B, T, H*W, D]
            h, w: spatial grid dims
        Returns:
            [B, T, (H//f)*(W//f), D]
        """
        B, T, _, D = x.shape
        f = self.factor
        if not self.depthwise:  # full Linear(D*f^2, D) resample
            x = x.unflatten(2, (h, w))                     # [B, T, H, W, D]
            x = x.transpose(-1, -2).transpose(-2, -3)      # [B, T, D, H, W]
            x = x.flatten(0, 1)                            # [BT, D, H, W]
            x = F.pixel_unshuffle(x, f)                    # [BT, D*f*f, H', W']
            x = x.unflatten(0, (B, T))                     # [B, T, D*f*f, H', W']
            x = x.flatten(3, 4).transpose(-1, -2)          # [B, T, H'W', D*f*f]
            return self.proj(x)                            # [B, T, H'W', D]
        H_p, W_p = h // f, w // f
        x = x.unflatten(2, (h, w))                    # [B, T, H, W, D]
        x = x.permute(0, 1, 4, 2, 3)                  # [B, T, D, H, W]
        x = x.flatten(0, 1)                           # [BT, D, H, W]
        x = F.pixel_unshuffle(x, f)                   # [BT, D*f², H', W']
        x = x.reshape(B * T, D, f * f, H_p, W_p)      # [BT, D, f², H', W']
        x = x.permute(0, 3, 4, 1, 2)                  # [BT, H', W', D, f²]
        x = (x * self.spatial_mix).sum(dim=-1)        # [BT, H', W', D]
        x = x.flatten(1, 2)                           # [BT, H'·W', D]
        x = self.proj(x)                              # [BT, H'·W', D]
        return x.unflatten(0, (B, T))                 # [B, T, H'·W', D]


class SpatialUpsample(nn.Module):
    """
    Depth-separable upsample: channel mix (D → D) then spatial broadcast
    (1 → f²).

    Flow: [B, T, H'W', D] → Linear(D, D) channel mix
          → reshape to [BT, H', W', D, 1] · [f²] → [BT, H', W', D, f²]
          → reshape and pixel_shuffle → [BT, D, H, W]
          → [B, T, H*W, D]

    Params: D² (channel mix) + f² (spatial broadcast).
    """

    def __init__(self, hidden_size: int, factor: int, depthwise: bool = True):
        super().__init__()
        self.factor = factor
        self.depthwise = depthwise
        f2 = factor * factor
        if depthwise:
            # zero-init so residual add-back starts as identity
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.proj.weight)
            self.spatial_broadcast = nn.Parameter(torch.full((f2,), 1.0 / f2))  # [f²]
        else:  # plain: Linear(D -> D*f²) -> pixel-shuffle
            self.proj = nn.Linear(hidden_size, hidden_size * f2, bias=False)
            nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: [B, T, H'·W', D]    where H' = h // f, W' = w // f
            h, w: ORIGINAL spatial grid dims (output size)
        Returns:
            [B, T, H*W, D]
        """
        B, T, _, D = x.shape
        f = self.factor
        H_p, W_p = h // f, w // f
        if not self.depthwise:  # full Linear(D*f^2, D) resample
            x = self.proj(x)                                    # [B, T, H'W', D*f*f]
            x = x.transpose(-1, -2).unflatten(-1, (H_p, W_p))   # [B, T, D*f*f, H', W']
            x = x.flatten(0, 1)                                 # [BT, D*f*f, H', W']
            x = F.pixel_shuffle(x, f)                           # [BT, D, H, W]
            x = x.unflatten(0, (B, T))                          # [B, T, D, H, W]
            x = x.flatten(3, 4).transpose(-1, -2)               # [B, T, H*W, D]
            return x
        x = self.proj(x)                                              # [B, T, H'W', D]
        x = x.unflatten(2, (H_p, W_p))                                # [B, T, H', W', D]
        x = x.unsqueeze(-1) * self.spatial_broadcast                  # [B, T, H', W', D, f²]
        x = x.flatten(0, 1)                                           # [BT, H', W', D, f²]
        x = x.permute(0, 3, 4, 1, 2)                                  # [BT, D, f², H', W']
        x = x.reshape(B * T, D * f * f, H_p, W_p)                     # [BT, D*f², H', W']
        x = F.pixel_shuffle(x, f)                                     # [BT, D, H, W]
        x = x.unflatten(0, (B, T))                                    # [B, T, D, H, W]
        x = x.flatten(3, 4).transpose(-1, -2)                         # [B, T, H*W, D]
        return x


# =============================================================================
# TemporalTransfer3D
# =============================================================================

class TemporalTransfer3D(nn.Module):
    """
    Temporal attention with downsampled spatial context and 3D RoPE.

    Both motion and spatial tokens participate in block-causal attention.
    Spatial tokens are downsampled (pixel unshuffle + linear), attend temporally,
    then upsampled (linear + pixel shuffle) for residual add-back to the spatial stream.

    Args:
        hidden_size: model dimension
        intermediate_size: FFN hidden dim
        num_heads: attention heads
        downsample_factor: spatial downsample factor (e.g. 4 → 16x16 → 4x4)
        ffn_type: "swiglu" or "gelu"
        qk_norm: use cosine-similarity attention
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        downsample_factor: int = 4,
        ffn_type: str = "swiglu",
        qk_norm: bool = False,
        depthwise: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.downsample_factor = downsample_factor
        self.qk_norm = qk_norm

        # Spatial resampling: depthwise=True (default) is a patchify-style depth-separable
        # resample; depthwise=False uses a full Linear(D*f^2, D) instead
        self.spatial_down = SpatialDownsample(hidden_size, downsample_factor, depthwise=depthwise)
        self.spatial_up = SpatialUpsample(hidden_size, downsample_factor, depthwise=depthwise)
        # Zero-init output projection for spatial add-back (identity at init)
        self.spatial_out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.spatial_out_proj.weight)

        # Pre-norm
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        if qk_norm:
            self.qk_scale = nn.Parameter(torch.full([num_heads, 1, 1], 10.0))

        # 3D RoPE
        self.rope = RoPE3D(self.head_dim)

        # FFN
        if ffn_type == "gelu":
            self.mlp = GELUMLP(hidden_size, intermediate_size)
        else:
            self.mlp = SwiGLU(hidden_size, intermediate_size)

        # Position cache
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    def _build_positions(
        self, M: int, ds_h: int, ds_w: int, T: int, device: torch.device
    ) -> torch.Tensor:
        """
        Build 3D positions for all tokens across all frames.

        Per-frame layout: [MT_0..MT_{M-1}, DS_(0,0)..DS_(ds_w-1,ds_h-1)]
        Motion: (0, 0, t)    Spatial: (x, y, t)

        Returns: [T * (M + ds_h*ds_w), 3]
        """
        key = (M, ds_h, ds_w, T, str(device))
        if key in self._pos_cache:
            return self._pos_cache[key]

        tokens_per_frame = M + ds_h * ds_w
        positions = torch.zeros(T, tokens_per_frame, 3, device=device)

        for t in range(T):
            # Motion tokens: (0, 0, t)
            positions[t, :M, 2] = t
            # Spatial tokens: (x, y, t)
            idx = M
            for row in range(ds_h):
                for col in range(ds_w):
                    positions[t, idx, 0] = col
                    positions[t, idx, 1] = row
                    positions[t, idx, 2] = t
                    idx += 1

        positions = positions.reshape(T * tokens_per_frame, 3)
        self._pos_cache[key] = positions
        return positions

    @compile_wrapper
    def _attention(self, x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Block-causal attention with 3D RoPE.

        Args:
            x: [B, S, D] — normed, flattened (T * tokens_per_frame)
            positions: [S, 3] — (x, y, t) per token
            mask: [S, S] — block-causal bool mask
        """
        from ttvidt.modules.layers import _qk_norm

        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 3D RoPE
        q, k = self.rope(q, k, positions)

        # QK-norm (cosine-sim attention)
        if self.qk_norm:
            q, k = _qk_norm(q, k, self.qk_scale)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return self.out_proj(attn.transpose(1, 2).reshape(B, S, D))

    def forward(
        self,
        motion_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            motion_tokens:  [B, T, M, D] — motion tokens from DINOv3 stream
            spatial_tokens: [B, T, H*W, D] — spatial tokens from DINOv3 stream
            spatial_h, spatial_w: spatial grid dimensions (e.g. 16, 16)

        Returns:
            motion_tokens:  [B, T, M, D] — enriched motion tokens
            spatial_tokens: [B, T, H*W, D] — spatial tokens with temporal residual
        """
        B, T, M, D = motion_tokens.shape
        f = self.downsample_factor
        ds_h, ds_w = spatial_h // f, spatial_w // f
        tokens_per_frame = M + ds_h * ds_w

        # 1. Downsample spatial: pixel unshuffle + linear
        ds_spatial = self.spatial_down(spatial_tokens, spatial_h, spatial_w)
        # ds_spatial: [B, T, ds_h*ds_w, D]

        # 2. Concat motion + downsampled spatial per frame
        x = torch.cat([motion_tokens, ds_spatial], dim=2)  # [B, T, M+S_down, D]

        # 3. Pre-norm + flatten temporal dim
        x_normed = self.norm1(x).flatten(1, 2)  # [B, T*(M+S_down), D]

        # 4. Positions and mask
        positions = self._build_positions(M, ds_h, ds_w, T, x.device)
        mask = get_block_causal_mask(T, tokens_per_frame, x.device)

        # 5. Block-causal attention with 3D RoPE
        attn_out = self._attention(x_normed, positions, mask)
        attn_out = attn_out.unflatten(1, (T, tokens_per_frame))

        # 6. Attention residual on ALL tokens (motion + spatial)
        x = x + attn_out

        # 7. FFN on ALL tokens (motion + spatial)
        x = x + self.mlp(self.norm2(x))

        # 8. Split back into motion and spatial
        motion_tokens = x[:, :, :M]
        spatial_part = x[:, :, M:]

        # 9. Spatial add-back: zero-init proj → upsample → residual
        #    spatial_out_proj is zero-init, so initially this is a no-op
        spatial_residual = self.spatial_up(
            self.spatial_out_proj(spatial_part), spatial_h, spatial_w
        )
        spatial_tokens = spatial_tokens + spatial_residual

        return motion_tokens, spatial_tokens


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    B, T, M, D = 2, 8, 8, 768
    H, W = 16, 16

    layer = TemporalTransfer3D(
        hidden_size=D,
        intermediate_size=D * 4,
        num_heads=12,
        downsample_factor=4,
    )

    motion = torch.randn(B, T, M, D)
    spatial = torch.randn(B, T, H * W, D)

    motion_out, spatial_out = layer(motion, spatial, H, W)
    print(f"Input:  motion={motion.shape}, spatial={spatial.shape}")
    print(f"Output: motion={motion_out.shape}, spatial={spatial_out.shape}")
    assert motion_out.shape == (B, T, M, D)
    assert spatial_out.shape == (B, T, H * W, D)

    # Verify spatial add-back is zero-init (residual starts as no-op)
    diff = (spatial_out - spatial).abs().max().item()
    print(f"Spatial diff at init (should be ~0 from zero-init out proj): {diff:.6f}")

    # Check gradient flows through spatial
    motion.requires_grad_(True)
    spatial.requires_grad_(True)
    m_out, s_out = layer(motion, spatial, H, W)
    loss = m_out.sum() + s_out.sum()
    loss.backward()
    print(f"Gradient on spatial: {spatial.grad is not None}, norm={spatial.grad.norm():.4f}")
    print(f"Gradient on motion: {motion.grad is not None}, norm={motion.grad.norm():.4f}")

    # Different downsample factors
    for ds in [1, 2, 4, 8]:
        l = TemporalTransfer3D(D, D * 4, 12, downsample_factor=ds)
        m_o, s_o = l(motion.detach(), spatial.detach(), H, W)
        ds_tokens = (H // ds) * (W // ds)
        print(f"  ds={ds}: {ds_tokens} spatial tokens/frame, total={M + ds_tokens}/frame")

    print("\nSmoke test passed!")
