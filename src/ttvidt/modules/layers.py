import torch
import torch.nn as nn
import torch.nn.functional as F

from ttvidt.utils import compile_wrapper


class RMSNorm(nn.RMSNorm):
    @compile_wrapper
    def forward(self, x):
        return F.rms_norm(
            x.to(self.weight), self.normalized_shape, self.weight, self.eps
        )


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(SwiGLU, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up = nn.Linear(hidden_size, intermediate_size * 2)
        self.down = nn.Linear(intermediate_size, hidden_size)

    @compile_wrapper
    def forward(self, x):
        x, gate = self.up(x).chunk(2, dim=-1)
        x = F.silu(x) * gate
        x = self.down(x)
        return x


class GELUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    @compile_wrapper
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _qk_norm(q, k, scale, eps=1e-6):
    """Cosine-similarity attention with learnable temperature (DisMo-style).

    Logits = scale * cos(q, k).
    Computes in float32 for numerical stability, uses rsqrt(sum_sq + eps).
    Scale is clamped to min=eps to prevent NaN from sqrt of negative values.
    """
    dtype = q.dtype
    q, k, scale = q.float(), k.float(), scale.float()
    sum_sq_q = torch.sum(q ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.clamp(min=eps))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return (q * scale_q).to(dtype), (k * scale_k).to(dtype)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, qk_norm=False, bias=True):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qk_norm = qk_norm
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        if qk_norm:
            self.qk_scale = nn.Parameter(torch.full([num_heads, 1, 1], 10.0))

    @compile_wrapper
    def forward(self, x, rope=None, positions=None):
        """
        Args:
            x: [B, T, SL, D] or [B, SL, D]
            rope: optional RoPE3D module (from tt3d.py)
            positions: optional [SL, 3] positions for 3D RoPE
        """
        # [B, T, SL, D]
        *b, seq_len, dim = x.shape
        x = x.flatten(1, 2)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Optional 3D RoPE
        if rope is not None and positions is not None:
            q, k = rope(q, k, positions)

        if self.qk_norm:
            q, k = _qk_norm(q, k, self.qk_scale)
            attn = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)

        x = self.out_proj(attn.transpose(1, 2).flatten(2, 3))
        return x.reshape(*b, seq_len, dim)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block: RMSNorm + Attention + FFN (SwiGLU or GELU)."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 ffn_type: str = "swiglu", qk_norm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qk_norm=qk_norm)
        self.norm2 = RMSNorm(hidden_size)
        if ffn_type == "gelu":
            self.mlp = GELUMLP(hidden_size, intermediate_size)
        else:
            self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, rope=None, positions=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope=rope, positions=positions)
        x = x + self.mlp(self.norm2(x))
        return x


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm: scale-only modulation from conditioning signal (DisMo-style)."""

    def __init__(self, hidden_size: int, cond_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.linear = nn.Linear(cond_size, hidden_size, bias=False)
        nn.init.zeros_(self.linear.weight)

    @compile_wrapper
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., D] input tensor
            c: [..., C] conditioning tensor (broadcastable to x batch dims)
        Returns:
            [..., D] modulated tensor
        """
        return (self.linear(F.silu(c)) + 1) * self.norm(x.to(self.norm.weight))


class CrossAttention(nn.Module):
    """Cross-attention: Q from main stream, K/V from context."""

    def __init__(self, hidden_size: int, num_heads: int, context_size: int | None = None, qk_norm: bool = False, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qk_norm = qk_norm
        context_size = context_size or hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(context_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(context_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        if qk_norm:
            self.qk_scale = nn.Parameter(torch.full([num_heads, 1, 1], 10.0))

    @compile_wrapper
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, S, D] main stream (query)
            context: [B, T, Sc, Dc] context (key/value)
        Returns:
            [B, T, S, D]
        """
        bt = x.shape[0] * x.shape[1]
        seq_q = x.shape[2]
        seq_kv = context.shape[2]

        x_flat = x.reshape(bt, seq_q, self.hidden_size)
        ctx_flat = context.reshape(bt, seq_kv, -1)

        q = self.q_proj(x_flat).view(bt, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx_flat).view(bt, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx_flat).view(bt, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q, k = _qk_norm(q, k, self.qk_scale)
            attn = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)
        out = self.out_proj(attn.transpose(1, 2).flatten(2, 3))
        return out.view(x.shape)


class AttentivePooling(nn.Module):
    """Learnable query token attends to M tokens to produce a CLS-like token."""

    def __init__(self, hidden_size: int, num_heads: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Parameter(torch.randn(1, 1, 1, hidden_size) * 0.02)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, M, D] motion tokens
        Returns:
            [B, T, D] pooled CLS-like token
        """
        B, T, M, D = x.shape
        bt = B * T

        q = self.query.expand(B, T, -1, -1).reshape(bt, 1, self.num_heads, self.head_dim).transpose(1, 2)
        x_flat = x.reshape(bt, M, D)
        k = self.k_proj(x_flat).view(bt, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(bt, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)  # [bt, H, 1, head_dim]
        out = self.out_proj(attn.transpose(1, 2).flatten(2, 3))  # [bt, 1, D]
        return out.squeeze(2).view(B, T, D)


class DiTTransformerBlock(nn.Module):
    """DiT-style transformer block with AdaRMSNorm conditioning and optional cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        cond_size: int,
        use_cross_attn: bool = False,
        context_size: int | None = None,
        use_rope: bool = False,
        qk_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        from ttvidt.modules.pos_embed import RoPE2DAttention

        # Self-attention sub-layer (with optional 2D RoPE)
        self.ada_norm1 = AdaRMSNorm(hidden_size, cond_size)
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPE2DAttention(hidden_size, num_heads, qk_norm=qk_norm, bias=bias)
        else:
            self.attn = Attention(hidden_size, num_heads, qk_norm=qk_norm, bias=bias)

        # Optional cross-attention sub-layer
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.ada_norm_cross = AdaRMSNorm(hidden_size, cond_size)
            self.cross_attn = CrossAttention(hidden_size, num_heads, context_size, qk_norm=qk_norm, bias=bias)

        # FFN sub-layer
        self.ada_norm2 = AdaRMSNorm(hidden_size, cond_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor | None = None,
        h: int | None = None,
        w: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, S, D] main stream
            c: [B, T, 1, Dc] conditioning (broadcast over S)
            context: [B, T, Sc, Dc_ctx] optional cross-attn context
            h, w: spatial dims (required when use_rope=True)
        Returns:
            [B, T, S, D]
        """
        if self.use_rope:
            x = x + self.attn(self.ada_norm1(x, c), h, w)
        else:
            x = x + self.attn(self.ada_norm1(x, c))
        if self.use_cross_attn and context is not None:
            x = x + self.cross_attn(self.ada_norm_cross(x, c), context)
        x = x + self.mlp(self.ada_norm2(x, c))
        return x


class LatentPatchEmbed(nn.Module):
    """
    Patch embedding for latent-space video input.

    Expects input of shape [B, T, C_in, H, W] where the spatial dimensions are
    already at patch resolution (e.g. from a VAE encoder).  If the input channel
    dimension differs from hidden_size, a linear projection is applied per-patch.

    When spatial_patch_size > 1 the spatial grid is further down-sampled via a
    Conv2d with kernel_size = stride = spatial_patch_size.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        spatial_patch_size: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.spatial_patch_size = spatial_patch_size

        if spatial_patch_size > 1:
            self.proj = nn.Conv2d(
                in_channels,
                hidden_size,
                kernel_size=spatial_patch_size,
                stride=spatial_patch_size,
            )
        else:
            self.proj = nn.Linear(in_channels, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            [B, T, H'*W', D]  where H' = H // spatial_patch_size, etc.
        """
        b, t, c, h, w = x.shape
        if self.spatial_patch_size > 1:
            x = x.reshape(b * t, c, h, w)
            x = self.proj(x)  # [B*T, D, H', W']
            _, d, hp, wp = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B*T, H'*W', D]
            x = x.reshape(b, t, hp * wp, d)
        else:
            # x: [B, T, C, H, W] -> [B, T, H*W, C] -> Linear -> [B, T, H*W, D]
            x = x.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c)
            x = self.proj(x)
        return x
