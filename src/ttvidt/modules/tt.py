"""
TemporalTransfer (TT1D): Block-causal temporal attention on motion tokens with 1D temporal RoPE.

Each frame's M motion tokens attend to current and past frames' motion tokens.
1D RoPE encodes temporal position (frame index) so tokens know their temporal order.
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


def get_block_causal_mask(t, m, device):
    """Bool mask [T*M, T*M] where frame i attends to frames 0..i."""
    key = (t, m, str(device))
    if key not in _mask_cache:
        causal = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool))
        block = causal[:, :, None, None].expand(-1, -1, m, m)
        mask = block.permute(0, 2, 1, 3).reshape(t * m, t * m)
        _mask_cache[key] = mask
    return _mask_cache[key]


# =============================================================================
# 1D Temporal RoPE
# =============================================================================

class TemporalRoPE1D(nn.Module):
    """1D Rotary Position Embedding for temporal dimension.

    Applies RoPE to half the head_dim (temporal), leaves the other half unchanged.
    """

    def __init__(self, head_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        # Use half for temporal RoPE, half unchanged
        self.rope_dim = head_dim // 2
        self.unused_dim = head_dim - self.rope_dim
        self.max_period = max_period
        self.register_buffer("freqs", self._freqs(), persistent=False)

    def _freqs(self) -> torch.Tensor:
        half = self.rope_dim // 2
        return torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32) / half
        )

    def reset_buffers(self) -> None:
        """Recompute the non-persistent buffers (e.g. after transformers' meta-device loading)."""
        self.freqs.copy_(self._freqs())

    def forward(self, q: torch.Tensor, k: torch.Tensor, t: int, m: int):
        """
        Apply 1D temporal RoPE to q and k.

        Args:
            q, k: [B, H, T*M, head_dim]
            t: number of frames
            m: tokens per frame
        """
        # Build temporal positions: each token in frame i gets position i
        # [0,0,...,0, 1,1,...,1, 2,2,...,2, ...] repeated m times per frame
        positions = torch.arange(t, device=q.device).repeat_interleave(m)  # [T*M]

        angles = positions.unsqueeze(-1).float() * self.freqs.to(q.device)  # [T*M, half]
        cos = torch.cos(angles).to(q.dtype)
        sin = torch.sin(angles).to(q.dtype)

        def apply(x):
            x_rope = x[..., :self.rope_dim]
            x_pass = x[..., self.rope_dim:]
            half = self.rope_dim // 2
            x1, x2 = x_rope[..., :half], x_rope[..., half:]
            x_rope = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
            return torch.cat([x_rope, x_pass], dim=-1)

        return apply(q), apply(k)


# =============================================================================
# Block-causal temporal attention with 1D RoPE
# =============================================================================

class BlockCausalTemporalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qk_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qk_norm = qk_norm
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        if qk_norm:
            self.qk_scale = nn.Parameter(torch.full([num_heads, 1, 1], 10.0))

        # 1D temporal RoPE
        self.rope = TemporalRoPE1D(self.head_dim)

    @compile_wrapper
    def forward(self, x):
        from ttvidt.modules.layers import _qk_norm
        b, t, m, _ = x.shape
        x = x.flatten(1, 2)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply 1D temporal RoPE
        q, k = self.rope(q, k, t, m)

        mask = get_block_causal_mask(t, m, q.device)
        if self.qk_norm:
            q, k = _qk_norm(q, k, self.qk_scale)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        x = self.out_proj(attn.transpose(1, 2).flatten(2, 3))
        return x.unflatten(1, (t, m))


# =============================================================================
# TemporalTransfer (TT1D)
# =============================================================================

class TemporalTransfer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, ffn_type="swiglu", qk_norm=False):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = BlockCausalTemporalAttention(hidden_size, num_heads, qk_norm=qk_norm)
        self.norm2 = RMSNorm(hidden_size)
        if ffn_type == "gelu":
            self.mlp = GELUMLP(hidden_size, intermediate_size)
        else:
            self.mlp = SwiGLU(hidden_size, intermediate_size)

    @compile_wrapper
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    tt_layer = TemporalTransfer(768, 3072, 12)
    x = torch.randn(2, 8, 8, 768)
    y = tt_layer(x)
    print(f"TT1D: {x.shape} -> {y.shape}")
    assert y.shape == x.shape
    print("TT1D smoke test passed!")
