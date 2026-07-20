"""
Transformer building blocks: RoPE, GQA, SwiGLU, TransformerBlock
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RoPE(nn.Module):
    def __init__(self, dim, max_seq=8192, rope_theta=10000.0, rope_scaling_factor=1.0):
        super().__init__()
        if rope_scaling_factor != 1.0:
            # NTK-aware scaling: theta' = theta * s^(d/(d-2))
            theta = rope_theta * (rope_scaling_factor ** (dim / (dim - 2)))
        else:
            theta = rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, offset=0):
        seq = x.shape[-2]
        device = x.device
        t = torch.arange(offset, offset + seq, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class GQA(nn.Module):
    def __init__(self, dim, n_heads=8, n_kv=4, rope_theta=10000.0, rope_scaling_factor=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, f'dim={dim} must be divisible by n_heads={n_heads}'
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.head_dim * n_kv, bias=False)
        self.wv = nn.Linear(dim, self.head_dim * n_kv, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_theta=rope_theta, rope_scaling_factor=rope_scaling_factor)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


class FlashGQA(nn.Module):
    """
    GQA con Flash Attention 2 vía torch.nn.functional.scaled_dot_product_attention.
    Auto-selecciona el kernel óptimo (FlashAttention / memory-efficient / math) según hardware y dtype.
    Reemplazo directo de GQA con la misma interfaz forward(x, mask).
    """
    def __init__(self, dim, n_heads=8, n_kv=4, rope_theta=10000.0, rope_scaling_factor=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, f'dim={dim} must be divisible by n_heads={n_heads}'
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.head_dim * n_kv, bias=False)
        self.wv = nn.Linear(dim, self.head_dim * n_kv, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim, rope_theta=rope_theta, rope_scaling_factor=rope_scaling_factor)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # Flash Attention 2 — is_causal=True evita crear mask explícita (más eficiente)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if mask is None else mask,
            is_causal=mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, ffn_dim=None):
        super().__init__()
        ffn_dim = ffn_dim or dim * 4
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, n_heads=8, n_kv=4, use_flash=False,
                 rope_theta=10000.0, rope_scaling_factor=1.0):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        if use_flash:
            self.attn = FlashGQA(dim, n_heads, n_kv, rope_theta=rope_theta, rope_scaling_factor=rope_scaling_factor)
        else:
            self.attn = GQA(dim, n_heads, n_kv, rope_theta=rope_theta, rope_scaling_factor=rope_scaling_factor)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
