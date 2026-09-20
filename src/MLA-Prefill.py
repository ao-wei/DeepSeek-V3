import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import math

from rope import ROPE, apply_rotary_pos_emb

torch.random.manual_seed(123)

@dataclass
class Config:
    hidden_size: int
    n_heads: int
    max_pos: int
    rope_theta: float
    rope_head_dim: int
    q_down_dim: int
    kv_down_dim: int
    head_dim: int
    dropout: float

class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_pos = config.max_pos
        self.rope_theta = config.rope_theta

        self.q_down_dim = config.q_down_dim

        self.rope_head_dim = config.rope_head_dim

        self.kv_dowm_dim = config.kv_down_dim

        self.head_dim = config.head_dim

        self.q_head_dim = config.head_dim + config.rope_head_dim

        self.q_down_proj = nn.Linear(self.hidden_size, self.q_down_dim, bias=False)

        self.q_up_proj = nn.Linear(self.q_down_dim, self.n_heads * self.q_head_dim, bias=False)

        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_dowm_dim + self.rope_head, bias=False)

        self.kv_up_proj = nn.Linear(self.kv_dowm_dim, self.n_heads*(self.head_dim + self.head_dim), bias=False)

        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = ROPE(
            self.rope_head_dim, 
            self.max_pos,
            self.rope_theta
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        B, T, D = hidden_states.size()

        # 1. Q
        q = self.q_up_proj(self.q_down_dim(hidden_states)) # [B, T, H*(head_dim+rope_dim)]
        q = q.view(B, T, self.n_heads, self.q_head_dim) # [B, T, H, head_dim+rope_dim]
        q = q.transpose(1, 2) # [B, H, T, head_dim+rope_dim]

        q_nope, q_rope = torch.split(
            q,
            [self.head_dim, self.rope_head_dim],
            dim=-1
        ) # [B, H, T, head_dim] [B, H, T, rope_dim]

        # 2. KV
        c_kv = self.kv_down_proj(hidden_states) # [B, T, c_dim + rope_dim]
        c_kv, k_pe = torch.split(
            c_kv,
            [self.kv_dowm_dim, self.rope_head_dim],
            dim=-1,
        ) # [B, T, c_dim] [B, T, rope_dim]

        k_pe = k_pe.view(B, T, 1, self.rope_head_dim).transpose(1, 2) # [B, 1, T, rope_dim]

        kv = self.kv_up_proj(c_kv) # [B, T, H*(head_dim + head_dim)]

        kv = kv.view(B, T, self.n_heads, self.head_dim + self.head_dim).transpose(1, 2) # [B, H, T, head_dim+head_dim]

        k_nope, v = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)

        cos, sin = self.rotary_emb(v, seq_len=T)

        q_pe = apply_rotary_pos_emb(q_rope, cos, sin, position_ids) # [B, H, T, rope_dim]
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, position_ids) # [B, H, T, rope_dim]

        q = torch.cat([q_nope, q_pe], dim=-1) # [B, H, T, head_dim + rope_dim]

        k = [k_nope, k_pe] # [B, H, T, head_dim+rope_dim]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.q_head_dim) # [B, H, T, T]

        if attention_mask is not None:
            attn_scores = torch.masked_fill(attn_scores, attention_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_weights = F.dropout(attn_weights, p=self.dropout) # [B, H, T, T]

        attn_output = attn_weights @ v # [B, H, T, head_dim]

        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)

        output = self.out_proj(attn_output) # [B, T, D]

        return output, c_kv, k_pe