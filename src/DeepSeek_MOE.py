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