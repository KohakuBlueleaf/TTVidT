"""
Positional embedding modules for TT-VidT.

Includes:
- Sinusoidal 1D/2D position embeddings
- Adaptive RoPE (Rotary Position Embedding) with aspect-ratio-aware positions
- Temporal distance embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ttvidt.utils import compile_wrapper

# ============================================================================
# Sinusoidal Position Embeddings
# ============================================================================


def get_1d_sincos_pos_embed(
    embed_dim: int, pos: torch.Tensor, max_period: int = 10000
) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        pos: Position tensor [N]
        max_period: Maximum period for frequency bands

    Returns:
        Sinusoidal embeddings [N, embed_dim]
    """
    half_dim = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, dtype=torch.float32, device=pos.device)
        / half_dim
    )
    args = pos.unsqueeze(-1) * freqs  # [N, D/2]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [N, D]


def get_2d_sincos_pos_embed(
    embed_dim: int, h: int, w: int, max_period: int = 10000
) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        h, w: Grid height and width
        max_period: Maximum period for frequency bands

    Returns:
        Sinusoidal embeddings [H*W, embed_dim]
    """
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid = torch.stack(
        torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=-1
    )  # [H, W, 2]
    grid = grid.reshape(-1, 2)  # [H*W, 2]

    # Split embedding dim for h and w
    half_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed(half_dim, grid[:, 0], max_period)
    emb_w = get_1d_sincos_pos_embed(half_dim, grid[:, 1], max_period)
    return torch.cat([emb_h, emb_w], dim=-1)  # [H*W, D]


def get_adaptive_2d_pos(
    h: int, w: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate adaptive 2D positions with aspect-ratio-aware normalization.

    Position mapping is normalized such that:
    - max(H) * max(W) = 1 (positions are in normalized range)
    - max(H) / max(W) = h / w (preserves aspect ratio)

    This means: max_H = sqrt(h/w), max_W = sqrt(w/h)

    Args:
        h, w: Grid dimensions
        device: Torch device

    Returns:
        (grid_h, grid_w): Flattened position grids, each [H*W]
    """
    aspect = h / w
    max_H = math.sqrt(aspect)
    max_W = math.sqrt(1.0 / aspect)

    pos_h = torch.linspace(0, max_H, h, device=device, dtype=torch.float32)
    pos_w = torch.linspace(0, max_W, w, device=device, dtype=torch.float32)

    grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing="ij")
    return grid_h.reshape(-1), grid_w.reshape(-1)


# ============================================================================
# Temporal Distance Embedding
# ============================================================================


class TemporalDistanceEmbedding(nn.Module):
    """
    Temporal distance embedding as a concatenated token using sincos encoding.

    Instead of adding temporal pos encoding, we create a dedicated token
    that encodes the distance from source frame to target motion frame.
    This token is concatenated: [frame_emb, motion_emb, time_info]

    Uses sinusoidal positional encoding (like standard transformer timestep)
    which generalizes to arbitrary frame distances without a fixed max.
    """

    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period

        # Precompute frequency bands for sincos encoding
        half_dim = hidden_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        start_distance: int = 1,
    ) -> torch.Tensor:
        """
        Generate temporal distance tokens using sincos encoding.

        Args:
            batch_size: batch size
            num_frames: number of motion frames (T)
            device: torch device
            start_distance: distance of first motion frame from reference

        Returns:
            [B, T, 1, D] temporal distance tokens
        """
        distances = torch.arange(
            start_distance,
            start_distance + num_frames,
            device=device,
            dtype=torch.float32,
        )

        # Sincos encoding: [T, D/2] -> [T, D]
        args = distances.unsqueeze(-1) * self.freqs.to(device)  # [T, D/2]
        time_tokens = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [T, D]

        # Expand: [1, T, 1, D] -> [B, T, 1, D]
        return time_tokens.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1, -1)


# ============================================================================
# Additive RoPE
# ============================================================================


class AdditiveRoPE2D(nn.Module):
    """
    Additive 2D Rotary Position Embedding with adaptive aspect-ratio-aware positions.

    x = x + rope(learnable_token, pos)

    Position mapping is normalized such that:
    - max(H) * max(W) = 1 (positions are in [0, 1] range)
    - max(H) / max(W) = vid_h / vid_w (preserves aspect ratio)

    This removes the need for max_h/max_w parameters and generalizes to any resolution.
    """

    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period

        # Learnable token that gets modulated by position
        self.learnable_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # Precompute frequency bands for sincos encoding (half for h, half for w)
        quarter_dim = hidden_size // 4
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(quarter_dim, dtype=torch.float32)
            / quarter_dim
        )
        self.register_buffer("freqs", freqs)

    def _get_adaptive_pos_embed(
        self, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        """Generate position embeddings with adaptive aspect-ratio-aware mapping."""
        grid_h, grid_w = get_adaptive_2d_pos(h, w, device)

        # Sincos encoding for h and w positions
        args_h = grid_h.unsqueeze(-1) * self.freqs.to(device)  # [L, D/4]
        emb_h = torch.cat([torch.sin(args_h), torch.cos(args_h)], dim=-1)  # [L, D/2]

        args_w = grid_w.unsqueeze(-1) * self.freqs.to(device)  # [L, D/4]
        emb_w = torch.cat([torch.sin(args_w), torch.cos(args_w)], dim=-1)  # [L, D/2]

        return torch.cat([emb_h, emb_w], dim=-1)  # [L, D]

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: [B, T, L, D] or [B, L, D] input tensor
            h, w: spatial dimensions

        Returns:
            x with additive RoPE applied
        """
        pos = self._get_adaptive_pos_embed(h, w, x.device).unsqueeze(0)  # [1, L, D]
        rope_emb = self.learnable_token * pos  # [1, L, D]

        if x.ndim == 4:
            rope_emb = rope_emb.unsqueeze(1)  # [1, 1, L, D]

        return x + rope_emb


# ============================================================================
# Rotary RoPE Attention
# ============================================================================


class RoPE2DAttention(nn.Module):
    """
    Attention with 2D Rotary Position Embedding applied to Q and K.

    Uses standard RoPE formulation extended to 2D with adaptive aspect-ratio-aware positions.
    Position mapping is normalized such that max(H) * max(W) = 1 and preserves aspect ratio.
    """

    def __init__(self, hidden_size: int, num_heads: int, max_period: int = 10000, qk_norm: bool = False, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_period = max_period
        self.qk_norm = qk_norm
        if qk_norm:
            self.qk_scale = nn.Parameter(torch.full([num_heads, 1, 1], 10.0))

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Precompute frequency bands (quarter for h, quarter for w)
        quarter_head_dim = self.head_dim // 4
        freqs = 1.0 / (
            max_period
            ** (
                torch.arange(0, quarter_head_dim, dtype=torch.float32)
                / quarter_head_dim
            )
        )
        self.register_buffer("freqs", freqs)

    def _compute_adaptive_rope_freqs(self, h: int, w: int, device: torch.device):
        """Compute RoPE frequencies for 2D grid with adaptive aspect-ratio mapping."""
        grid_h, grid_w = get_adaptive_2d_pos(h, w, device)

        freqs_h = torch.outer(grid_h, self.freqs.to(device))  # [L, head_dim/4]
        freqs_w = torch.outer(grid_w, self.freqs.to(device))  # [L, head_dim/4]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # [L, head_dim/2]

        return torch.cos(freqs), torch.sin(freqs)

    def _apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary position embedding."""
        # x: [B, num_heads, L, head_dim]
        # cos, sin: [L, head_dim/2]
        x_half = x.shape[-1] // 2
        x1, x2 = x[..., :x_half], x[..., x_half:]

        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim/2]
        sin = sin.unsqueeze(0).unsqueeze(0)

        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    @compile_wrapper
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: [B, T, S, D] input (T=temporal, S=sequence length)
               S may be h*w (spatial only) or h*w + extra tokens (motion, timestep, etc.)
            h, w: spatial dimensions (h*w <= S)
        """
        B, T, S, D = x.shape
        L = h * w  # spatial token count
        total = T * S

        x_flat = x.flatten(1, 2)  # [B, T*S, D]

        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        q = q.view(B, total, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, total, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, total, self.num_heads, self.head_dim).transpose(1, 2)

        # Get adaptive RoPE frequencies for spatial positions
        cos_spatial, sin_spatial = self._compute_adaptive_rope_freqs(h, w, x.device)
        # cos_spatial, sin_spatial: [L, head_dim/2]

        # Build per-token RoPE: spatial tokens get position encoding,
        # extra tokens (motion, timestep) get zero rotation (cos=1, sin=0)
        half = cos_spatial.shape[-1]
        if S > L:
            extra = S - L
            ones = torch.ones(extra, half, device=x.device, dtype=cos_spatial.dtype)
            zeros = torch.zeros(extra, half, device=x.device, dtype=cos_spatial.dtype)
            cos_frame = torch.cat([cos_spatial, ones], dim=0)   # [S, half]
            sin_frame = torch.cat([sin_spatial, zeros], dim=0)  # [S, half]
        else:
            cos_frame = cos_spatial
            sin_frame = sin_spatial

        # Tile for all temporal frames
        cos = cos_frame.unsqueeze(0).expand(T, -1, -1).reshape(total, -1)
        sin = sin_frame.unsqueeze(0).expand(T, -1, -1).reshape(total, -1)

        # Apply RoPE to Q and K
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # QK-norm (after RoPE)
        if self.qk_norm:
            from ttvidt.modules.layers import _qk_norm
            q, k = _qk_norm(q, k, self.qk_scale)
            attn = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)

        out = attn.transpose(1, 2).flatten(2, 3)  # [B, T*S, D]
        out = self.out_proj(out)

        return out.view(B, T, S, D)
