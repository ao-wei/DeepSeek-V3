import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class ModelArgs:
    n_dim: int
    n_experts: int
    n_shared_experts: int
    top_k: int
    dropout: float
    batch: int
    seq_len: int

    bias_lr: float
    enable_bias: bool = False
    enable_seq_aux: bool = False # Auxiliary Loss
    alpha: float = 0.1 

class TopkRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.top_k

        self.topk_linear = nn.Linear(args.n_dim, args.n_experts)

    def forward(self, x, bias):
        logits = self.topk_linear(x) # [T, N]

        logits = logits + bias # [T, N]  动态调整的 bias 是 Auxiliary-loss-free 的关键

        top_k_logits, indices = logits.topk(self.top_k, dim=-1) # [T, k]

        infs = torch.full_like(logits, float("-inf")) # [T, N]

        sparse_logits = infs.scatter_(-1, indices, top_k_logits) # [T, N]

        router_output = nn.functional.softmax(sparse_logits, dim=-1) # [T, N]

        return router_output, indices

class Expert(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.gate = nn.Linear(args.n_dim, 4 * args.n_dim, bias=False)
        self.value = nn.Linear(args.n_dim, 4 * args.n_dim, bias=False)
        self.out = nn.Linear(4 * args.n_dim, args.n_dim, bias=False)

    def forward(self, x):
        return self.out(nn.functional.silu(self.gate(x)) * self.value(x))

class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.router = TopkRouter(args)
        self.routed_experts = nn.ModuleList([Expert(args) for _ in range(args.n_experts)])
        self.shared_experts = nn.ModuleList([Expert(args) for _ in range(args.n_shared_experts)])
        self.top_k = args.top_k
        self.n_experts = args.n_experts

        # 1. Auxiliary-loss seq load balancing
        self.enable_seq_aux = args.enable_seq_aux
        self.alpha = args.alpha

        # 2. Auxiliary-loss-free load balancing
        self.enable_bias = args.enable_bias
        if args.enable_bias:
            self.bias_lr = args.bias_lr
            self.register_buffer("expert_bias", torch.zeros(self.n_experts))
        
    def forward(self, x):
        B, T, D = x.shape

        x = x.view(-1, D) # [T, D]

        gating_output, indices = self.router(x, self.expert_bias) # [T, D] [T, K]

        final_output = torch.zeros_like(x)
        expert_usage = torch.zeros(self.n_experts)

        # routed experts
        for i, expert in enumerate(self.routed_experts):
            expert_mask = (indices == i).any(dim=-1) # [T]

            expert_usage[i] = expert_mask.sum().float()

            if expert_mask.any():
                expert_input = x[expert_mask] # [M, D]
                expert_output = expert(expert_input) # [M, D]

                gating_scores = gating_output[expert_mask, i].unsqueeze(1) # [M, 1]
                weighted_output = expert_output * gating_scores # [M, D]

                final_output[expert_mask] += weighted_output # [T, D]

        # shared_experts
        for i, expert in enumerate(self.shared_experts):
            final_output += expert(x) # [T, D]

        # Auxiliary-loss-free load balancing
        if self.enable_bias:
            for i in range(self.n_experts):
                self.expert_bias += (expert_usage.mean() - expert_usage[i])

        # Auxiliary-loss sequence-level load balancing
        aux_loss = None
        if self.enable_seq_aux:
            topk_idx_for_aux_loss = indices.view(B, -1) # [B, S*K] 以每条 sequence 为单位统计专家负载

            scores_for_seq_aux = gating_output.view(B, T, -1) # [B, S, N]

            fi = torch.zeros(B, self.n_experts) # [B, N] 每个 sequence 中各专家的负载

            fi = torch.scatter_add(fi, 1, topk_idx_for_aux_loss, torch.ones_like(topk_idx_for_aux_loss)) # [B, N]

            fi = torch.div(fi, T * self.top_k / self.n_experts) # [B, N]

            pi = scores_for_seq_aux.mean(dim=1) # [B, N]

            # fi 表示各专家的实际流量，pi 表示 Router 的偏好

            aux_loss = (fi * pi).sum(dim=1).mean() * self.alpha

        return final_output, expert_usage, aux_loss