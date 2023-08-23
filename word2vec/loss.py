from torch import nn
import torch


class NegativeSampling(nn.Module):
    def __init__(self, proba_input: bool = False):
        super().__init__()
        self._bce = nn.BCELoss(reduction='none') if proba_input else nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
        positive_labels = torch.ones_like(positive_logits, dtype=torch.float32).to(positive_logits)
        positive_loss = self._bce(positive_logits, positive_labels).sum(-1)

        negative_labels = torch.ones_like(positive_logits, dtype=torch.float32).to(positive_logits)
        negative_loss = self._bce(negative_logits, negative_labels).sum(-1)

        return torch.mean(positive_loss + negative_loss)
