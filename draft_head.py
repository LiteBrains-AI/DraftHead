# draft_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DraftHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_drafts: int = 4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_drafts = num_drafts

        # Draft generators
        self.drafts = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size)
                for _ in range(num_drafts)
            ]
        )

        # Confidence predictors
        self.confidence_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, 1)
                for _ in range(num_drafts)
            ]
        )

        # Final projection
        self.output_proj = nn.Linear(
            hidden_size,
            hidden_size,
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq, hidden]

        Returns:
            output: [batch, seq, hidden]
            confidences: [batch, seq, num_drafts]
        """

        draft_outputs = []
        confidence_scores = []

        for draft_layer, conf_layer in zip(
            self.drafts,
            self.confidence_heads,
        ):
            draft = draft_layer(x)

            confidence = conf_layer(draft)

            draft_outputs.append(draft)
            confidence_scores.append(confidence)

        drafts = torch.stack(
            draft_outputs,
            dim=2,
        )

        confidences = torch.cat(
            confidence_scores,
            dim=-1,
        )

        # Normalize confidence scores
        confidences = F.softmax(
            confidences,
            dim=-1,
        )

        weighted_drafts = (
            drafts
            * confidences.unsqueeze(-1)
        )

        fused = weighted_drafts.sum(dim=2)

        output = self.output_proj(fused)

        return output, confidences

# Demo

x = torch.randn(
    2,      # batch
    16,     # sequence length
    128,    # hidden size
)

layer = DraftHead(
    hidden_size=128,
    num_drafts=4,
)

output, confidence = layer(x)

print(output.shape)
print(confidence.shape)
