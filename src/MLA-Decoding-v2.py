import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import math

from rope import ROPE, apply_rotary_pos_emb

@dataclass
class Config:
    hidden_size: int
    n_heads: int
    max_pos: int
    rope_theta: float
    rope_head_dim: int
    q_lora_rank: int
    kv_lora_rank: int
    head_dim: int
    dropout: float

class MLADecodingV2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_pos = config.max_pos
        self.rope_theta = config.rope_theta

        self.q_lora_rank = config.q_lora_rank

        self.rope_head_dim = config.rope_head_dim

        self.kv_lora_rank = config.kv_lora_rank

        self.head_dim = config.head_dim

        self.q_head_dim = config.head_dim + config.rope_head_dim

        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
        )

        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.n_heads * self.q_head_dim,
            bias=False,
        )

        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.rope_head_dim,
            bias=False,
        )

        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.head_dim + self.head_dim),
            bias=False
        )

        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        self.rotary_emb = ROPE(
            self.rope_head_dim,
            self.max_pos,
            self.rope_theta,
        )

    def forward(self, x, attention_mask = None, position_ids = None, compressed_kv_cache = None):
        # compressed_kv_cache: [B, L, kv_lora_rank]
        B, T, D = x.size()

        q = self.q_up_proj(self.q_down_proj(x)) # [B, T, H*(head_dim+rope_dim)]

        q = q.view(B, T, self.n_heads, self.q_head_dim).transpose(1, 2) # [B, H, T, head_dim+rope_dim]

        # [B, H, T, head_dim] [B, H, T, rope_dim]
        q_nope, q_pe = torch.split(q, [self.head_dim, self.rope_head_dim], dim=-1)

        compressed_kv_new = self.kv_down_proj(x) # [B, T, kv_lora_rank+rope_dim]

        compressed_kv = torch.cat([compressed_kv_cache, compressed_kv_new], dim=1) # [B, L+T, kv_lora_rank+rope_dim]

        kv_seq_len = compressed_kv.size(1)
        # [B, L+T, kv_lora_rank] [B, L+T, rope_dim]
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.rope_head_dim], dim=-1)

        k_pe = k_pe.view(B, kv_seq_len, 1, self.rope_head_dim).transpose(1, 2) # [B, H, L+T, rope_dim]

        kv_up_proj = self.kv_up_proj.weight.view(self.n_heads, -1, self.kv_lora_rank) # [H, head_dim+head_dim, kv_lora_rank]

        w_uk = kv_up_proj[:, :self.head_dim, :] # [H, head_dim, kv_lora_rank]
        w_uv = kv_up_proj[:, self.head_dim:, :] # [H, head_dim, kv_lora_rank]

        cos, sin = self.rotary_emb(q_pe)

        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, position_ids)

        q_nope = q_nope @ w_uk # [B, H, T, head_dim] @ [H, head_dim, kv_lora_rank] = [B, H, T, kv_lora_rank]
        
        q_merge = torch.concat([q_nope, q_pe], dim=-1)
        k_merge = torch.concat([compressed_kv.unsqueeze(1), k_pe], dim=-1)

        # [B, H, T, kv_lora_rank+rope_dime] @ [B, H, kv_lora_rank+rope_dim, L+T] = [B, H, T, L+T]
        attention_scores = q_merge @ k_merge.transpose(2, 3) / math.sqrt(self.q_head_dim) 

        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32)

        attn_output = attention_weights @ compressed_kv.unsqueeze(1) # [B, H, T, L+T] @ [B, H, L+T, kv_lora_rank] = [B, H, T, kv_lora_rank]

        attn_output = attn_output @ w_uv.transpose(1, 2) # [B, H, T, kv_lora_rank] @ [H, kv_lora_rank, head_dim] = [B, H, T, head_dim]

        attn_output = self.o_proj(attn_output)

        return attn_output

