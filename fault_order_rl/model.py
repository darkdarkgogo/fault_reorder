"""Dynamic fault features and shared neural scorer."""

import torch
from torch import nn


def build_dynamic_features(embeddings, remaining_rows):
    """Combine candidate embeddings with their current remaining-set context."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] != 257:
        raise ValueError("embeddings must have shape [N, 257] with N > 0")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values")
    rows = torch.as_tensor(remaining_rows, device=embeddings.device)
    if rows.ndim != 1 or rows.numel() == 0:
        raise ValueError("remaining rows must be a non-empty one-dimensional sequence")
    if rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("remaining rows must contain integer indices")
    if (rows < 0).any() or (rows >= embeddings.shape[0]).any():
        raise ValueError("remaining rows contain an out-of-range index")
    if torch.unique(rows).numel() != rows.numel():
        raise ValueError("remaining rows must be unique")
    candidates = embeddings[rows.long()]
    context = candidates.mean(dim=0, keepdim=True).expand(rows.numel(), -1)
    ratio = embeddings.new_full((rows.numel(), 1), rows.numel() / embeddings.shape[0])
    return torch.cat((candidates, context, ratio), dim=1)


class FaultScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dimension = 515
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dimension),
            nn.Linear(self.input_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.input_dimension:
            raise ValueError(
                "fault features must have shape [N, {}]".format(
                    self.input_dimension
                )
            )
        if not torch.isfinite(features).all():
            raise ValueError("fault features contain non-finite values")
        scores = self.network(features).squeeze(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("fault scorer produced non-finite scores")
        return scores
