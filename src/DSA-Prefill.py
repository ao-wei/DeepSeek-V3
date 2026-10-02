import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

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

    index_heads: int
    index_head_dim: int
    index_topk: int

class Indexer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.index_heads = config.index_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.index_topk = config.index_topk

        self.w_dq = nn.Linear(self.q_lora_rank, self.index_heads * self.head_dim, bias=False)
        self.w_dk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_dh = nn.Linear(self.hidden_size, self.index_heads, bias=False)

        self.k_norm = nn.LayerNorm(self.head_dim)

        self.rotary_emb = ROPE(
            self.rope_head_dim,
            config.max_pos,
            config.rope_theta
        )

    def forward(self, hidden_states, qr, position_ids, attention_mask=None):
        # qr: [B, T, q_lora_rank]
        B, T, D = hidden_states.size()

        q = self.w_dq(qr) # [B, T, index_heads * head_dim]

        q = q.view(B, T, self.index_heads, self.head_dim).transpose(1, 2) # [B, index_heads, T, head_dim]

        # Partially apply RoPE
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        cos, sin = self.rotary_emb(q_pe)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, position_ids)

        q = torch.cat([q_pe, q_nope], dim=-1) # [B, index_heads, T, head_dim]

        k = self.w_dk(hidden_states) # [B, T, head_dim]

        k = self.k_norm(k).unsqueeze(1) # [B, 1, T, head_dim]

        # Partially apply RoPE
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        cos, sin = self.rotary_emb(k_pe)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, position_ids)
        k = torch.cat([k_pe, k_nope], dim=-1) # [B, 1, T, head_dim]

        weights = self.w_dh(hidden_states) # [B, T, index_heads]

        weights = weights.transpose(1, 2).unsqueeze(-1) # [B, index_heads, T, 1]

        k = k.repeat_interleave(self.index_heads, dim=1) # [B, index_heads, T, head_dim]

        scale = self.head_dim ** (-0.5)

        attn = torch.matmul(q, k.transpose(-1, -2)) * scale

        attn = F.relu(attn)

        index_score = torch.sum(weights * attn, dim=1) # [B, T, T]

        topk_indices = index_score.top(min(self.index_topk, T), dim=-1)[1] # [B, T, K]

        return index_score, topk_indices

class DSA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_pos = config.max_pos
        self.rope_theta = config.rope_theta

        # 对应于query压缩的向量，在deepseek v3中，hidden_size 7168
        # 压缩后的kv d_c=512, 压缩比例 1/14
        # q的压缩为1536，压缩比例为 1/4.7
        # rope的维度是 64

        # 对应 query 的压缩向量
        self.q_lora_rank = config.q_lora_rank

        # 对应query 和 key进行rope的维度
        self.rope_head_dim = config.rope_head_dim

        # 对应 key, value的压缩向量
        self.kv_lora_rank = config.kv_lora_rank

        # 对应每一个head的维度
        self.head_dim = config.head_dim

        # 对应Q， K做attention的维度，nope_dim + rope_dim
        self.q_head_dim = config.head_dim + config.rope_head_dim

        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)

        self.q_up_proj = nn.Linear(self.q_lora_rank, self.n_heads * self.q_head_dim, bias=False)

        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, self.rope_head_dim, bias=False)

        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.n_heads * (self.head_dim + self.head_dim), bias=False)

        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = ROPE(
            self.rope_head_dim,
            self.max_pos,
            self.rope_theta
        )

        self.indexer = Indexer(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        B, T, D = hidden_states.size()

        qr = self.q_down_proj(hidden_states)

        q = self.q_up_proj(qr)
        q = q.view(B, T, self.n_heads, self.q_head_dim).transpose(1, 2)

        q_nope, q_pe = torch.split(q, [self.head_dim, self.rope_head_dim], dim=-1)

        c_kv = self.kv_down_proj(hidden_states)
        c_kv, k_pe = torch.split(c_kv, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        k_pe = k_pe.view(B, T, 1, self.rope_head_dim).transpose(1, 2)

        kv = self.kv_up_proj(c_kv)
        kv = kv.view(B, T, self.n_heads, -1).transpose(1, 2)

        k_nope, v = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)

        cos, sin = self.rotary_emb(v, seq_len=T)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, position_ids)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, position_ids)

        q = torch.cat([q_nope, q_pe], dim=-1)

        k = torch.cat([k_nope, k_pe], dim=-1)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.q_head_dim)

        index_score, topk_indices = self.indexer(hidden_states, qr, position_ids, attention_mask)

        index_mask = torch.full((B, T, T), True).scatter_(-1, topk_indices, False)

        if attention_mask is not None:
            mask = index_mask + attention_mask
            mask = mask.unsqueeze(1)
            print("mask: ", mask)
            attn_weights = torch.masked_fill(
                attn_weights,
                mask,
                float("-inf")
            )

        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 7. 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, c_kv, k_pe, index_score
