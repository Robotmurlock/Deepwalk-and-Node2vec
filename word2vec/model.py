import torch
from torch import nn


class W2VBase(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, max_norm: float = 1.0):
        super().__init__()
        self._input_embedding = nn.Embedding(vocab_size, embedding_size, max_norm=max_norm)
        self._output_embedding = nn.Embedding(vocab_size, embedding_size, max_norm=max_norm)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self._input_embedding.weight)
        torch.nn.init.xavier_uniform_(self._output_embedding.weight)

    def embed_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._input_embedding(inputs)

    def embed_outs(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._output_embedding(outputs)


class SkipGram(W2VBase):
    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, proba: bool = True) -> torch.Tensor:
        # B is batch size, V is vocabulary size and E is embedding size
        # inputs shape: (B, 1, V)
        # outputs shape: (B, N, V)
        batch_size, _, _ = outputs.shape

        inputs_emb = self.embed_inputs(inputs).view(batch_size, -1, 1)  # shape: (B, E, 1)
        outputs_emb = self.embed_outs(outputs)  # shape: (B, N, E)

        scalars = torch.bmm(outputs_emb, inputs_emb).view(batch_size, -1)  # shape: (B, N)
        if proba:
            scalars = torch.sigmoid(scalars)
        return scalars


class CBOW(nn.Module):
    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, proba: bool = True) -> torch.Tensor:
        # B is batch size, V is vocabulary size and E is embedding size
        # inputs shape: (B, N, V)
        # outputs shape: (B, 1, V)
        batch_size, _, _ = outputs.shape

        inputs_emb = torch.mean(self.embed_inputs(inputs), dim=1).view(batch_size, -1, 1)  # shape: (B, E, 1)
        outputs_emb = self.embed_outs(outputs)  # shape: (B, 1, E)

        scalars = torch.bmm(outputs_emb, inputs_emb).view(batch_size, -1)  # shape: (B, 1)
        if proba:
            scalars = torch.sigmoid(scalars)
        return scalars
