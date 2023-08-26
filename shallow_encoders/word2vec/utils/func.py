"""
Custom torch functions.
"""
import torch


def pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculates pairwise cosine between two vectors.

    Args:
        x: First vector
        y: Second vector

    Returns:
        Pairwise similarity (similarity matrix)
    """
    x = x / torch.norm(x, dim=-1).unsqueeze(-1).expand_as(x)
    y = y / torch.norm(y, dim=-1).unsqueeze(-1).expand_as(y)
    return torch.mm(x, torch.transpose(y, 0, 1))
