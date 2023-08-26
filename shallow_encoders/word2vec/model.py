"""
Implementation of Word2vec models
"""
from typing import Optional

import torch
from torch import nn


class W2VBase(nn.Module):
    """
    Base W2V model. Contains embeddings for both input and context words.
    """
    def __init__(self, vocab_size: int, embedding_size: int, max_norm: Optional[float] = None):
        """
        Args:
            vocab_size: Dataset vocabulary size
            embedding_size: Embeddings dimensionality
            max_norm: Maximum norm
        """
        super().__init__()
        self._input_embedding = nn.Embedding(vocab_size, embedding_size, max_norm=max_norm)
        self._output_embedding = nn.Embedding(vocab_size, embedding_size, max_norm=max_norm)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self._input_embedding.weight)
        torch.nn.init.xavier_uniform_(self._output_embedding.weight)

    @property
    def input_embedding(self) -> torch.Tensor:
        """
        Fetch input embedding weights.

        Returns:
            Input embedding weights
        """
        return self._input_embedding.weight.to('cpu').data

    @property
    def output_embedding(self) -> torch.Tensor:
        """
        Fetch output embedding weights.

        Returns:
            Output embedding weights
        """
        return self._output_embedding.weight.to('cpu').data

    def embed_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Transforms input word indices (long) to their vector representation (embedding lookup)

        Args:
            inputs: Input word indices

        Returns:
            Inputs embeddings
        """
        return self._input_embedding(inputs)

    def embed_outs(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Transforms context word indices (long) to their vector representation (embedding lookup)


        Args:
            outputs: Output word indices

        Returns:
            Outputs (context) embeddings
        """
        return self._output_embedding(outputs)


class SkipGram(W2VBase):
    """
    Implementation of SkipGram model.
    """
    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, proba: bool = True) -> torch.Tensor:
        # B is batch size, E is embedding size
        # inputs shape: (B, 1)
        # outputs shape: (B, N)
        batch_size, _ = outputs.shape

        inputs_emb = self.embed_inputs(inputs).view(batch_size, -1, 1)  # shape: (B, E, 1)
        outputs_emb = self.embed_outs(outputs)  # shape: (B, N, E)

        scalars = torch.bmm(outputs_emb, inputs_emb).view(batch_size, -1)  # shape: (B, N)
        if proba:
            scalars = torch.sigmoid(scalars)
        return scalars


class CBOW(W2VBase):
    """
    Implementation of ContinousBagOfWords model.
    """
    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, proba: bool = True) -> torch.Tensor:
        # B is batch size, E is embedding size
        # inputs shape: (B, N)
        # outputs shape: (B, 1)
        batch_size, _ = outputs.shape

        inputs_emb = torch.mean(self.embed_inputs(inputs), dim=1).view(batch_size, -1, 1)  # shape: (B, E, 1)
        outputs_emb = self.embed_outs(outputs)  # shape: (B, 1, E)

        scalars = torch.bmm(outputs_emb, inputs_emb).view(batch_size, -1)  # shape: (B, 1)
        if proba:
            scalars = torch.sigmoid(scalars)
        return scalars
