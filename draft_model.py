import torch
import torch.nn as nn
import torch.nn.functional as F

from draft_head import DraftHead 
# from draft_head2 import DraftHead [comment out to use]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=128,
        num_heads=4,
        num_drafts=4,
        ff_dim=256,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_size)

        # DraftHead as a topping
        self.draft_head = DraftHead(
            hidden_size=hidden_size,
            num_drafts=num_drafts,
        )

        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_size),
        )

        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x):

        attn_out, _ = self.attn(
            x,
            x,
            x,
        )

        x = self.norm1(x + attn_out)

        draft_out, confidence = self.draft_head(x)

        x = self.norm2(x + draft_out)

        ff_out = self.ffn(x)

        x = self.norm3(x + ff_out)

        return x, confidence


class DraftHeadModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        num_drafts=4,
        max_length=256,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.position_embedding = nn.Embedding(
            max_length,
            hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_drafts=num_drafts,
                )
                for _ in range(num_layers)
            ]
        )

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        confidences = []

        for layer in self.layers:
            x, conf = layer(x)
            confidences.append(conf)

        logits = self.lm_head(x)

        return logits, confidences
