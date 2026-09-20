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

        # Add jitter to input x
        if self.training:
            jitter_std = 0.25  # Standard deviation of the noise
            noise = torch.randn_like(x) * jitter_std
            x = x + noise

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

        # [B, S, D, H]
        drafts = torch.stack(
            draft_outputs,
            dim=2,
        )

        # [B, S, D]
        confidences = torch.cat(
            confidence_scores,
            dim=-1,
        )

        # Normalize confidence
        confidences = F.softmax(
            confidences,
            dim=-1,
        )

        # Select top 2 drafts based on confidence
        top2_confidences, top2_indices = torch.topk(confidences, k=2, dim=-1)

        # Expand top2_indices for gather operation
        expanded_top2_indices = top2_indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)

        # Gather the top 2 drafts
        top2_drafts = torch.gather(drafts, dim=2, index=expanded_top2_indices)

        # Expand top2_confidences for weighted sum broadcasting
        expanded_top2_confidences = top2_confidences.unsqueeze(-1)

        # Perform weighted sum of the top 2 drafts
        combined_draft = (top2_drafts * expanded_top2_confidences).sum(dim=2)

        output = self.output_proj(
            combined_draft
        )

        return output, confidences


# Demo

x = torch.randn(
    2,
    16,
    128,
)

layer = DraftHead(
    hidden_size=128,
    num_drafts=4,
)

output, confidence = layer(x)

print(f"Output shape (combined top 2 drafts): {output.shape}")
print(f"Confidence shape: {confidence.shape}")
