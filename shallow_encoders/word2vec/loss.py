"""
Implementation of Word2vec losses.
"""
from typing import Dict

import torch
from torch import nn


class NegativeSamplingLoss(nn.Module):
    """
    Implementation of negative sampling loss.
    """
    def forward(self, positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        positive_loss = - torch.log(torch.clamp(torch.sigmoid(positive_logits), min=1e-6))
        negative_loss = - torch.log(torch.clamp(torch.sigmoid(-negative_logits), min=1e-6)).sum(-1)

        return {
            'loss': torch.mean(positive_loss + negative_loss),
            'positive-loss': torch.mean(positive_loss),
            'negative-loss': torch.mean(negative_loss)
        }
