import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class ModelArgs:
    n_dim: int
    n_expert: int
    top_k: int
    dropout: float
    batch: int
    seq_len: int
    add_noise: bool

# 经典的 sparse MOE 的 Top-K Router
class TopkRouter(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.top_k = args.top_k
        self.add_noise = args.add_noise

        self.gate_linear = nn.Linear(args.n_dim, args.n_expert)

        if args.add_noise:
            self.noise_linear = nn.Linear(args.n_dim, args.n_expert)

    def forward(self, mha_output):
        logits = self.gate_linear(mha_output)

        if self.add_noise:
            noise_logits = self.noise_linear(mha_output)

            noise = torch.randn_like(logits) * nn.functional.softplus(noise_logits)
            logits = logits + noise

        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

        infs = torch.full_like(logits, float("-inf"))
        sparse_logits = infs.scatter(dim=-1, index=top_k_indices, src=top_k_logits)

        gating_output = nn.functional.softmax(sparse_logits, dim=-1)

        return gating_output, top_k_indices

"""
但 DeepSeek-V3 的 Router 逻辑不是这样，而是更接近：
x -> Linear -> Sigmoid affinity -> 加入 correction bias，仅用于选择 -> group-limited Top-K -> 取原 affinity -> 对选中的 K 个归一化
"""

class Expert(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(args.n_dim, 4 * args.n_dim),
            nn.ReLU(),
            nn.Linear(4 * args.n_dim, args.n_dim),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.ffn(x)

class SparseMOE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.router = TopkRouter(args)
        self.experts = nn.ModuleList([Expert(args) for _ in range(args.n_experts)])
        self.top_k = args.top_k

    def forward(self, x):
        gating_output, indices = self.router(x) # [B, T, N], [B, T, K]
        print("selected experts: ", indices)

        final_output = torch.zeros_like(x) # [B, T, D]

        # MoE 的路由单位是 token。一个 expert 可能同时接收来自不同 batch（样本）、不同位置的 token，所以先把 batch 和序列长度两个维度合并
        flat_x = x.view(-1, x.size(-1)) # [B, T, D] -> [B*T, D]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1)) # [B, T, N] -> [B*T, N]

        # 逐个调用专家；每个专家批量处理分配给自己的 token
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1) # [B, T]
            flat_mask = expert_mask.view(-1) # [B*T]

            if flat_mask.any():
                expert_input = flat_x[flat_mask] # [selected_token, D]

                expert_output = expert(expert_input) # [selected_token, D]

                gating_scores = flat_gating_output[flat_mask, i] # [selected_token]

                gating_scores = gating_scores.unsqueeze(1) # [selected_token, 1]

                weighted_output = expert_output * gating_scores # [selected_token, D]

                final_output[expert_mask] += weighted_output

        return final_output 
