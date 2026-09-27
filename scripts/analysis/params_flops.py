"""Analytical parameter counts and forward FLOPs for the encoders and decoders.

    python scripts/analysis/params_flops.py                     # TT3D with depth-separable resample
    python scripts/analysis/params_flops.py --tt3d-full-linear   # TT3D with full Linear resample

Conventions:
  - Linear(in -> out) on ``seq`` tokens = 2 * seq * in * out FLOPs.
  - flops_attention(seq, D)              = 8·seq·D² + 4·seq²·D
  - flops_cross_attn(seq_q, seq_kv, D)   = 4·seq_q·D² + 4·seq_kv·D² + 4·seq_q·seq_kv·D
  - flops_swiglu(seq, D, I)              = 6·seq·D·I
  - flops_gelu_mlp(seq, D, I)            = 4·seq·D·I

Forward-only FLOPs reported.
Params reported as a sum of Linear weight numels (no bias when configured off).

Operating point (everywhere): T=8 frames, 256x256 inputs, patch=16,
M=8 motion tokens; the frame VAE maps a 256 px frame to a 32x32x4 latent, which
the decoder patchifies 2x2 into 16x16 tokens.
"""
from __future__ import annotations

import argparse


# ---------------------------------------------------------------------------
# FLOP helpers
# ---------------------------------------------------------------------------
def f_attn(seq, D):           return 8 * seq * D * D + 4 * seq * seq * D
def f_xattn(seq_q, seq_kv, D): return 4 * seq_q * D * D + 4 * seq_kv * D * D + 4 * seq_q * seq_kv * D
def f_swiglu(seq, D, I):      return 6 * seq * D * I
def f_gelu(seq, D, I):        return 4 * seq * D * I
def f_linear(seq, in_d, out_d): return 2 * seq * in_d * out_d


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------
T          = 8
RES        = 256
PATCH      = 16
M          = 8                   # motion tokens / frame (TT-VidT only)
N_REG      = 5                   # CLS + 4 registers (DINOv3)
H_p = W_p  = RES // PATCH        # 16 patches per side
N_SP       = H_p * W_p           # 256 spatial tokens / frame
N_DINO     = N_REG + N_SP        # 261 — DINOv3 baseline sequence length
N_TTVIDT   = M + N_REG + N_SP    # 269 — DINOv3 sequence inside TT-VidT (motion tokens injected)
DEC_PATCH    = 2
LATENT_DIM   = 32                # f8 frame VAE: 256 px / 8 = 32 latent per side
DEC_H = DEC_W = LATENT_DIM // DEC_PATCH    # 32 / 2 = 16
DEC_LAT_C    = 4                 # latent channels
DEC_N        = DEC_H * DEC_W     # 16*16 = 256 latent tokens / frame
TT3D_DEPTHWISE = True            # False with --tt3d-full-linear

TT_F       = 4                   # tt_downsample
TT_S_DOWN  = (H_p // TT_F) * (W_p // TT_F)   # 4*4 = 16


def fmt_p(p):
    return f"{p/1e6:.1f}M" if p < 1e9 else f"{p/1e9:.2f}B"

def fmt_g(g):
    return f"{g:.1f}"


# ===========================================================================
# Encoders
# ===========================================================================

# ---- DINOv3 ViT-B (12L, D=768, GELU MLP I=3072), per-frame ---------------
DINO_L, DINO_D, DINO_I = 12, 768, 3072

def dinov3_block_params(D, I):
    return 4 * D * D + 2 * D * I    # self-attn + GELU MLP

def dinov3_block_flops(D, I, seq):
    return f_attn(seq, D) + f_gelu(seq, D, I)


# ---- TT-1D layer (seq = T*M, GELU MLP) -------------------------------------
def tt1d_layer_params(D, I):
    return 4 * D * D + 2 * D * I

def tt1d_layer_flops(D, I, T_, M_):
    return f_attn(T_ * M_, D) + f_gelu(T_ * M_, D, I)


# ---- TT-3D layer -----------------------------------------------------------
# Full-linear resample (tt_spatial_depthwise=False): pixel-unshuffle f x f patches
# then Linear(D*f^2 -> D); up: Linear(D -> D*f^2) then pixel-shuffle.
def tt3d_layer_params_plain(D, I, f):
    return 2 * D * f * f * D + D * D + 4 * D * D + 2 * D * I   # down + up + out_proj + attn + mlp

def tt3d_layer_flops_plain(D, I, f, T_, M_, N_sp_full, N_sp_down):
    down = f_linear(T_ * N_sp_down, D * f * f, D)
    up = f_linear(T_ * N_sp_down, D, D * f * f)
    out_proj = f_linear(T_ * N_sp_full, D, D)
    seq = T_ * (M_ + N_sp_down)
    return down + up + out_proj + f_attn(seq, D) + f_gelu(seq, D, I)


# Depth-separable resample (optional, tt_spatial_depthwise=True).
# Per-layer, per-frame: spatial_down acts on N_SP positions, spatial_up acts
# on TT_S_DOWN positions, spatial_out_proj acts on N_SP positions.
def tt3d_layer_params(D, I, f):
    sp_mix     = f * f                # spatial f²→1 weights
    sp_proj    = D * D                # depth-separable channel mix (down)
    up_mix     = f * f                # spatial 1→f² weights
    up_proj    = D * D                # depth-separable channel mix (up)
    out_proj   = D * D                # spatial_out_proj
    attn       = 4 * D * D
    mlp        = 2 * D * I            # GELU MLP per current configs
    return sp_mix + sp_proj + up_mix + up_proj + out_proj + attn + mlp

def tt3d_layer_flops(D, I, f, T_, M_, N_sp_full, N_sp_down):
    # spatial_down: per-frame mix over N_sp_down (post-unshuffle), channel-mix Linear D→D over N_sp_down
    spatial_pool_down = T_ * N_sp_down * D * (f * f)        # 1 mul per (channel, position, f²) ≈ 2·N_sp_down·D·f²
    proj_down         = f_linear(T_ * N_sp_down, D, D)
    spatial_pool_up   = T_ * N_sp_down * D * (f * f)        # broadcast 1→f²
    proj_up           = f_linear(T_ * N_sp_down, D, D)
    out_proj          = f_linear(T_ * N_sp_full, D, D)
    seq = T_ * (M_ + N_sp_down)
    attn = f_attn(seq, D)
    mlp  = f_gelu(seq, D, I)
    return spatial_pool_down + proj_down + spatial_pool_up + proj_up + out_proj + attn + mlp


# ---- TT-VidT TT-1D / TT-3D total -------------------------------------------
def attentive_pool_params(D):
    # Q (D, learnable per-pool), K/V (D→D), out (D→D)
    return 4 * D * D

def attentive_pool_flops(M_, D):
    # 1 query attending to M tokens: KV (2·M·D²), Q (2·D²), QK + AV (4·M·D), out (2·D²)
    return 2 * M_ * D * D + 2 * D * D + 4 * M_ * D + 2 * D * D


def ttvidt_tt1d():
    backbone_p = DINO_L * dinov3_block_params(DINO_D, DINO_I)
    tt_p       = DINO_L * tt1d_layer_params(DINO_D, DINO_I)
    pool_p     = 2 * attentive_pool_params(DINO_D)
    patch_emb  = 3 * PATCH * PATCH * DINO_D                       # conv2d(3, D, k=patch, s=patch)
    motion_emb = M * DINO_D                                       # learnable motion tokens
    p = backbone_p + tt_p + pool_p + patch_emb + motion_emb

    # FLOPs (DINOv3 stream now sees motion tokens too: N=N_TTVIDT=269)
    backbone_f = T * DINO_L * dinov3_block_flops(DINO_D, DINO_I, N_TTVIDT)
    tt_f       = DINO_L * tt1d_layer_flops(DINO_D, DINO_I, T, M)
    pool_f     = T * 2 * attentive_pool_flops(M, DINO_D)
    patch_f    = T * f_linear(N_SP, 3 * PATCH * PATCH, DINO_D)
    return p, backbone_f + tt_f + pool_f + patch_f


def ttvidt_tt3d():
    backbone_p = DINO_L * dinov3_block_params(DINO_D, DINO_I)
    layer_p, layer_f = ((tt3d_layer_params, tt3d_layer_flops) if TT3D_DEPTHWISE
                        else (tt3d_layer_params_plain, tt3d_layer_flops_plain))
    tt_p       = DINO_L * layer_p(DINO_D, DINO_I, TT_F)
    pool_p     = 2 * attentive_pool_params(DINO_D)
    patch_emb  = 3 * PATCH * PATCH * DINO_D
    motion_emb = M * DINO_D
    p = backbone_p + tt_p + pool_p + patch_emb + motion_emb

    backbone_f = T * DINO_L * dinov3_block_flops(DINO_D, DINO_I, N_TTVIDT)
    tt_f       = DINO_L * layer_f(DINO_D, DINO_I, TT_F, T, M, N_SP, TT_S_DOWN)
    pool_f     = T * 2 * attentive_pool_flops(M, DINO_D)
    patch_f    = T * f_linear(N_SP, 3 * PATCH * PATCH, DINO_D)
    return p, backbone_f + tt_f + pool_f + patch_f


# ---- VideoMAE-3D (24L full-3D, GELU) --------------------------------------
VMAE_L, VMAE_D, VMAE_I = 24, 768, 3072
def vmae3d():
    p_block = 4 * VMAE_D * VMAE_D + 2 * VMAE_D * VMAE_I        # GELU
    p = VMAE_L * p_block + 3 * PATCH * PATCH * VMAE_D
    seq = T * (1 + N_SP)
    f_block = f_attn(seq, VMAE_D) + f_gelu(seq, VMAE_D, VMAE_I)
    fl = VMAE_L * f_block + T * f_linear(N_SP, 3 * PATCH * PATCH, VMAE_D)
    return p, fl


# ---- DisMo-2D-3D (12 spatial per-frame + 12 temporal-3D, GELU) ------------
DISMO_L_S, DISMO_L_T, DISMO_D, DISMO_I = 12, 12, 768, 3072
def dismo():
    p_block = 4 * DISMO_D * DISMO_D + 2 * DISMO_D * DISMO_I
    p = (DISMO_L_S + DISMO_L_T) * p_block + 3 * PATCH * PATCH * DISMO_D + 1 * DISMO_D
    # spatial: per-frame seq = N_SP
    f_spatial = T * DISMO_L_S * (f_attn(N_SP, DISMO_D) + f_gelu(N_SP, DISMO_D, DISMO_I))
    # temporal: full 3D seq = T*(1 + N_SP)
    seq_3d = T * (1 + N_SP)
    f_temporal = DISMO_L_T * (f_attn(seq_3d, DISMO_D) + f_gelu(seq_3d, DISMO_D, DISMO_I))
    return p, f_spatial + f_temporal + T * f_linear(N_SP, 3 * PATCH * PATCH, DISMO_D)


# ---- DINOv3 ViT-B (frozen baseline; per-frame cost) ----------------------
def dinov3_per_frame():
    """Returns (params, FLOPs for ONE frame). Multiply by T for video."""
    p = DINO_L * dinov3_block_params(DINO_D, DINO_I) + 3 * PATCH * PATCH * DINO_D
    fl = DINO_L * dinov3_block_flops(DINO_D, DINO_I, N_DINO) \
       + f_linear(N_SP, 3 * PATCH * PATCH, DINO_D)
    return p, fl


# ---- V-JEPA2 (ours impl., 24L × 768d, joint space-time, tubelet=2) --------
# Evaluated at T_in = 16 input frames (paper setting) → 8 latent frames.
VJEPA_L, VJEPA_D, VJEPA_I = 24, 768, 3072
VJEPA_TUBE = 2
VJEPA_T_in  = 16
VJEPA_T_eff = VJEPA_T_in // VJEPA_TUBE       # 8 latent frames
VJEPA_SEQ   = VJEPA_T_eff * N_SP             # 8 * 256 = 2048
def vjepa2_ours():
    p_block = 4 * VJEPA_D * VJEPA_D + 2 * VJEPA_D * VJEPA_I
    # Tubelet conv3d(3, D, k=(tube, patch, patch), s=(tube, patch, patch))
    p = VJEPA_L * p_block + 3 * VJEPA_TUBE * PATCH * PATCH * VJEPA_D
    f_block = f_attn(VJEPA_SEQ, VJEPA_D) + f_gelu(VJEPA_SEQ, VJEPA_D, VJEPA_I)
    fl = VJEPA_L * f_block + f_linear(VJEPA_SEQ, 3 * VJEPA_TUBE * PATCH * PATCH, VJEPA_D)
    return p, fl


# ===========================================================================
# Decoders (DiT, S/B/L; with vs. without cross-attn)
# ===========================================================================

DEC_VARIANTS = {
    "S": dict(L=12, D=768,  I=3072),
    "B": dict(L=16, D=1024, I=4096),
    "L": dict(L=28, D=1152, I=3456),
}

def dit_block_params(D, I, with_xattn):
    self_attn  = 4 * D * D                    # q,k,v,out (no bias under our config)
    cross_attn = 4 * D * D if with_xattn else 0
    mlp        = 3 * D * I                    # SwiGLU: up=2DI + down=DI = 3DI
    n_ada      = 3 if with_xattn else 2
    ada        = n_ada * D * D
    return self_attn + cross_attn + mlp + ada

def dit_block_flops(D, I, seq, ctx_seq, with_xattn):
    sa = f_attn(seq, D)
    xa = f_xattn(seq, ctx_seq, D) if with_xattn else 0
    mlp = f_swiglu(seq, D, I)
    n_ada = 3 if with_xattn else 2
    ada = n_ada * f_linear(seq, D, D)
    return sa + xa + mlp + ada


def decoder(name, with_xattn):
    cfg = DEC_VARIANTS[name]
    L, D, I = cfg["L"], cfg["D"], cfg["I"]
    seq = DEC_N + 1                              # latent tokens + timestep
    ctx_seq = M                                  # cross-attn context = motion tokens / frame
    # patch_in: image_dim (4) → D, patch_size² = 4
    patch_in_p = (DEC_LAT_C * DEC_PATCH * DEC_PATCH) * D
    unpatch_p  = D * (DEC_LAT_C * DEC_PATCH * DEC_PATCH)
    proj_p     = 4 * D * D                       # img/motion/context/timestep proj (small misc)
    p = L * dit_block_params(D, I, with_xattn) + patch_in_p + unpatch_p + proj_p
    # FLOPs over T frames
    f_block = dit_block_flops(D, I, seq, ctx_seq, with_xattn)
    patch_in_f = T * f_linear(DEC_N, DEC_LAT_C * DEC_PATCH * DEC_PATCH, D)
    unpatch_f  = T * f_linear(DEC_N, D, DEC_LAT_C * DEC_PATCH * DEC_PATCH)
    fl = T * L * f_block + patch_in_f + unpatch_f
    return p, fl


# ===========================================================================
# Render
# ===========================================================================

def render_md_encoders(ENCODER_ROWS):
    out = ["| Encoder | Params | FLOPs (GF) | Scope |",
           "|---------|-------:|-----------:|-------|"]
    for r in ENCODER_ROWS:
        out.append(f"| {r['name']} | {fmt_p(r['params'])} | {fmt_g(r['flops']/1e9)} | {r['scope']} |")
    return "\n".join(out)


def render_md_decoders(DECODER_ROWS):
    out = ["| Variant | L | D | I | xattn | Params | FLOPs (GF, T=8) |",
           "|---------|---|---|---|-------|-------:|----------------:|"]
    for r in DECODER_ROWS:
        out.append(f"| {r['variant']} | {r['L']} | {r['D']} | {r['I']} | "
                   f"{'yes' if r['xattn'] else 'no'} | {fmt_p(r['params'])} | "
                   f"{fmt_g(r['flops']/1e9)} |")
    return "\n".join(out)



def main():
    global TT3D_DEPTHWISE
    ap = argparse.ArgumentParser(description="Analytical params / FLOPs (see module docstring).")
    ap.add_argument("--tt3d-full-linear", action="store_true", help="count the full Linear(D*f^2, D) TT3D resample")
    TT3D_DEPTHWISE = not ap.parse_args().tt3d_full_linear
    ENCODER_ROWS = []
    def add(name, p, fl, scope=None):
        ENCODER_ROWS.append({"name": name, "params": p, "flops": fl,
                             "scope": scope or f"T={T}, {RES}^2"})

    p, fl_1f = dinov3_per_frame()
    add("DINOv3 ViT-B (1-frame)", p, fl_1f, scope=f"1 frame, {RES}^2")
    add("DINOv3 ViT-B (T=8 per-frame)", p, T * fl_1f, scope=f"T={T}, {RES}^2")
    p, fl = vmae3d()
    add("VideoMAE-3D (ours impl., 24L)", p, fl)
    p, fl = dismo()
    add("DisMo-2D-3D (ours impl., 12s+12t)", p, fl)
    p, fl = ttvidt_tt1d()
    add("TT-VidT TT-1D (ours)", p, fl)
    p, fl = ttvidt_tt3d()
    add("TT-VidT TT-3D (ours)" + (" [full-linear resample]" if not TT3D_DEPTHWISE else ""), p, fl)
    p, fl = vjepa2_ours()
    add("V-JEPA2 (ours impl., 24L x 768d, tube=2)", p, fl, scope=f"T={VJEPA_T_in} (8 latent), {RES}^2")


    DECODER_ROWS = []
    for k in ("S", "B", "L"):
        for x in (False, True):
            p, fl = decoder(k, x)
            DECODER_ROWS.append({"variant": f"dec{k}", "L": DEC_VARIANTS[k]["L"],
                                 "D": DEC_VARIANTS[k]["D"], "I": DEC_VARIANTS[k]["I"],
                                 "xattn": x, "params": p, "flops": fl})
    print("## Encoders (T=8, 256x256)\n")
    print(render_md_encoders(ENCODER_ROWS))
    print("\n## Decoders (T=8)\n")
    print(render_md_decoders(DECODER_ROWS))


if __name__ == "__main__":
    main()
