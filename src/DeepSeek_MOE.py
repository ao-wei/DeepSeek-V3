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

        self.gate_linear = nn.Linear(args.n_dim, args.n_experts)

        if args.add_noise:
            self.noise_linear = nn.Linear(args.n_dim, args.n_experts)

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

# 类 DeepSeek MoE（简化的路由机制和负载均衡）
@dataclass
class ModelArgs:
    n_dim: int
    n_experts: int
    n_shared_experts: int
    top_k: int
    dropout: float
    batch: int
    seq_len: int
    add_noise: bool

args = ModelArgs(n_dim=32, n_experts=4, n_shared_experts=1, top_k=2, dropout=0.1, batch=1, seq_len=5, add_noise=True)

class DeepSeekExpert(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.gate = nn.Linear(args.n_dim, 4 * args.n_dim, bias=False)
        self.value = nn.Linear(args.n_dim, 4 * args.n_dim, bias=False)
        self.out = nn.Linear(4 * args.n_dim, args.n_dim, bias=False)

    def forward(self, x):
        return self.out(nn.functional.silu(self.gate(x)) * self.value(x))

class DeepSeekMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.router = TopkRouter(args)
        self.routed_experts = nn.ModuleList([DeepSeekExpert(args) for _ in range(args.n_experts)])
        self.shared_experts = nn.ModuleList([DeepSeekExpert(args) for _ in range(args.n_shared_experts)])
        self.top_k = args.top_k

    def forward(self, x):
        gating_output, indices = self.router(x) # [B, T, N] [B, T, K]
        
        final_output = torch.zeros_like(x) # [B, T, D]

        for i, expert in enumerate(self.routed_experts):
            expert_mask = (indices == i).any(dim=-1) # [B, T]

            if expert_mask.any():
                expert_input = x[expert_mask] # [selected_token, D]

                expert_output = expert(expert_input) # [selected_token, D]

                gating_scores = gating_output[..., i] # [B, T]
                gating_scores = gating_scores[expert_mask] # [selected_token]
                gating_scores = gating_scores.unsqueeze(1) # [selected_token, 1]

                # [B, T, N] -> [selected_token, N] -> [selected_token] -> [selected_token, 1]
                # 或者是：gating_scores = gating_output[expert_mask][:, i].unsqueeze(1)

                weighted_output = gating_scores * expert_output

                final_output[expert_mask] += weighted_output

        for i, expert in enumerate(self.shared_experts):
            final_output += expert(x)

        return final_output, indices

torch.manual_seed(666)
import numpy as np

deepseek_moe = DeepSeekMoE(args)

count = 100
experts_utilization = np.zeros(args.n_experts, dtype=int)
for _ in range(count):
    mha_output = torch.randn(args.batch, args.seq_len, args.n_dim)
    _, indices = deepseek_moe(mha_output)
    indices = indices.detach().cpu().numpy()
    indices = indices.flatten()
    for idx in indices:
        experts_utilization[idx] += 1

print("专家负载：", experts_utilization)
print("激活的总次数：", experts_utilization.sum())