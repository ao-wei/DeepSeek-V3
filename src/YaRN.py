import math
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    n_head: int
    head_dim: int
    origin_seq: int
    max_seq: int
    low: int
    high: int
    base: int
    factor: float

def get_index(r, dim, base, train_seq_len):
    # 根据比值（圈数）来反推 dim 的 index
    return dim * math.log(train_seq_len / 2 * r * math.pi) / ( 2 * math.log(base))

def get_correction_range(
    beta_fast,
    beta_slow,
    dim,
    base,
    train_seq_len,
):
    high_freq_idx = math.floor(get_index(beta_fast, dim, base, train_seq_len))

    low_freq_idx = math.ceil(get_index(beta_slow, dim, base, train_seq_len))

    return max(high_freq_idx, 0), min(low_freq_idx, dim // 2 - 1)

def ramp_func(low, high, dim):
    linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)

    return torch.clamp(linear_func, 0, 1)

def precompute_freqs_cis(args):
    freqs_range = torch.arange(0, args.head_dim, 2, dtype=torch.float32)

    freqs = 1.0 / (args.base ** (freqs_range / args.head_dim))

    if args.max_seq > args.origin_seq:
        low, high = get_correction_range(args.high, args.low, args.head_dim, args.base, args.origin_seq)

        smooth = 1 - ramp_func(low, high, args.head_dim // 2)

        freqs = (1 - smooth) * freqs / args.factor + smooth * freqs

    max_seq_arange = torch.arange(args.max_seq)

    freqs = torch.outer(max_seq_arange, freqs)

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    B, T, H, D = x.shape

    x = x.view(B, T, H, -1, 2)

    x = torch.view_as_complex(x)

    freqs_cis = freqs_cis.view(1, T, 1, x.size(-1))

    rotate_x = torch.view_as_real(x * freqs_cis).flatten(3)

    return rotate_x
