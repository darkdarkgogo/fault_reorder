"""Shared neural scorer for collapsed fault embeddings."""

import torch
from torch import nn


class FaultScorer(nn.Module):
    def __init__(self, input_dimension=257):
        super().__init__()
        self.input_dimension = int(input_dimension)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dimension),
            nn.Linear(self.input_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, embeddings):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.input_dimension:
            raise ValueError(
                "fault embeddings must have shape [N, {}]".format(
                    self.input_dimension
                )
            )
        scores = self.network(embeddings).squeeze(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("fault scorer produced non-finite scores")
        return scores
