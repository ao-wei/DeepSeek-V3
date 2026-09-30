import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        x = (x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + self.eps)) * self.scale

class MTPModule(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        self.combine_proj = nn.Linear(2 * d_model, d_model, bias=False)

        self.prev_norm = RMSNorm(d_model)
        self.cur_norm = RMSNorm(d_model)

        self.block = nn.TransformerEncoderLayer(d_model, n_head, dropout=dropout)

    def forward(self, prev_hidden, cur_token_embed):
        # prev_hidden: [B, T, D]  cur_token_embed: [B, T, D]
        prev_norm = self.prev_norm(prev_hidden)

        cur_norm = self.cur_norm(cur_token_embed)

        combined = torch.cat([prev_norm, cur_norm], dim=-1)
        hidden = self.combine_proj(combined)

        hidden = self.block(hidden)

        return hidden

class MTP(nn.Module):
    def __init__(self, d_model, vocab_size, n_mtp_module=3, n_head=2, dropout=0.1, num_layers=3):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_mtp_module = n_mtp_module

        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight

        block = nn.TransformerEncoderLayer(d_model, n_head, dropout=dropout)
        self.blocks = nn.TransformerEncoder(block, num_layers=num_layers)

        self.mtp_modules = nn.ModuleList([MTPModule(d_model, n_head, dropout) for _ in range(n_mtp_module)])

    def forward(self, token_ids, targets, mtp_loss_weight=0.3):
        # token_ids, targets: [B, T]
        B, T = token_ids.shape

        embeds = self.embed(token_ids)

        h_0 = self.blocks(embeds)

        main_logits = self.lm_head(h_0)
        main_loss = F.cross_entropy(
            main_logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=-100,
        )

        mtp_losses = []
        current_hidden = h_0
        for depth, mtp_module in enumerate(self.mtp_modules, 1):
            """
            main head input ids: 1,2,3,4,5
            target input ids: 2,3,4,5,-100
            mtp head input ids: 2,3,4,5|3,4,5|4,5
            """

            future_embeds = embeds[:, depth:, :]
            pad_size = depth

            padding = torch.zeros(B, pad_size, self.d_model)

            future_embeds = torch.cat([future_embeds, padding], dim=1)

            current_hidden = mtp_module(current_hidden, future_embeds)

            mtp_logits = self.lm_head(current_hidden)

            shift_logits = mtp_logits[:, :-depth, :].coutiguous()

            shift_targets = targets[:, depth:].contiguous()

            mtp_module_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
            )

            mtp_losses.append(mtp_module_loss)

        mtp_loss = torch.stack(mtp_losses).mean()

        total_loss = main_loss + mtp_loss_weight * mtp_loss

        return total_loss