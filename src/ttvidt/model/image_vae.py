import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Encoder (LatentMaid)
# ============================================================

class SwiGLU(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, mlp_dim * 2, kernel_size=1)
        self.fc2 = nn.Conv2d(mlp_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.fc1(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = F.silu(x) * gate
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim)
        self.dw_conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.swiglu = SwiGLU(in_dim, mlp_dim, out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dw_conv(x)
        x = self.swiglu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=4, **kwargs):
        super().__init__()
        block_configs = (
            (1, 64, 128, 2),
            (2, 128, 256, 2),
            (2, 256, 512, 2),
            (2, 512, 1024, 2),
        )

        scales = []
        stages = []
        stage_projs = []
        dim = input_dim
        for scale, hidden_dim, mlp_dim, blocks in block_configs:
            dim = dim * scale ** 2
            stages.append(nn.ModuleList([BasicBlock(hidden_dim, mlp_dim, hidden_dim) for _ in range(blocks)]))
            if hidden_dim != dim:
                stage_projs.append(nn.Conv2d(dim, hidden_dim, 1))
            else:
                stage_projs.append(nn.Identity())
            scales.append(scale)
            dim = hidden_dim

        self.stages = nn.ModuleList(stages)
        self.stage_projs = nn.ModuleList(stage_projs)
        self.scales = scales
        self.out_proj = nn.Conv2d(dim, latent_dim * 2, kernel_size=1)
        self.quant_conv = nn.Conv2d(latent_dim * 2, latent_dim * 2, 1)

    def forward(self, x):
        for scale, stage, proj in zip(self.scales, self.stages, self.stage_projs):
            x = F.pixel_unshuffle(x, scale)
            x = proj(x)
            for block in stage:
                x = block(x)
        x = self.out_proj(x)
        x = self.quant_conv(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar


# ============================================================
# Decoder (standard LDM)
# ============================================================

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, temb=None):
        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)
        h = nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, 1, c, h * w).transpose(-1, -2)
        k = k.reshape(b, 1, c, h * w).transpose(-1, -2)
        v = v.reshape(b, 1, c, h * w).transpose(-1, -2)
        attn_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        h_ = F.scaled_dot_product_attention(q.to(attn_dtype), k.to(attn_dtype), v.to(attn_dtype))
        h_ = h_.to(v.dtype).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, *, ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                 attn_resolutions=(), dropout=0.0, resamp_with_conv=True,
                 resolution=256, z_channels=4, tanh_out=False, **kwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)

    def forward(self, z):
        h = self.conv_in(self.post_quant_conv(z))
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = nonlinearity(self.norm_out(h))
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


# ============================================================
# AutoencoderKL wrapper + loader
# ============================================================

class AutoencoderKL(nn.Module):
    def __init__(self, input_dim=3, latent_dim=4,
                 ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                 attn_resolutions=(), dropout=0.0, resolution=256, z_channels=4):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult,
                               num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions,
                               dropout=dropout, resolution=resolution,
                               z_channels=z_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample=False):
        mean, logvar = self.encode(x)
        if sample:
            z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            z = mean
        return self.decode(z), mean, logvar


def load_autoencoder(path, device="cpu", dtype=None):
    """Load AutoencoderKL from a single .safetensors or .pt file.

    Returns a ready-to-use AutoencoderKL in eval mode.
    """
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(path), device=str(device))
    else:
        state_dict = torch.load(str(path), map_location=device, weights_only=True)

    model = AutoencoderKL()
    model.load_state_dict(state_dict)
    model.eval()
    if dtype is not None:
        model = model.to(dtype=dtype)
    if device != "cpu":
        model = model.to(device)
    return model
