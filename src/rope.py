import torch
import torch.nn as nn

class ROPE(nn.Module):
    def __init__(self, dim, max_pos=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_pos = max_pos
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        t = torch.arange(
            self.max_seq_len_cached, dtype=self.inv_freq.dtype
        )

        angles = torch.outer(t, self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1)

        self.register_buffer("cos_cached", angles.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", angles.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dytpe),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin, position_ids, unsqueeze_dim=1):
    # cos, sin: [max_seq_len, d]
    # position_ids: [B, S]
    cos = cos[position_ids] # [B, S, d]
    sin = sin[position_ids] # [B, S, d]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = x.shape
    x = x.view(b, h, s, d // 2, 2).transpose(3, 4).reshape(b, h, s, d)

    x_embed = (x * cos) + (rotate_half(x)*sin)

    return x_embed


