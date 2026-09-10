"""Gumbel/Plackett-Luce listwise ranking operations."""

import numpy as np
import torch


def centered_logits(scores, temperature):
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return (scores - scores.mean()) / float(temperature)


def sample_permutation(scores, temperature, generator=None):
    logits = centered_logits(scores, temperature)
    uniform = torch.rand(
        logits.shape,
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    tiny = torch.finfo(uniform.dtype).tiny
    uniform = uniform.clamp(min=tiny, max=1.0 - torch.finfo(uniform.dtype).eps)
    gumbel = -torch.log(-torch.log(uniform))
    permutation = torch.argsort(logits + gumbel, descending=True)
    return permutation, logits


def deterministic_permutation(scores):
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    values = scores.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    rows = np.arange(values.shape[0], dtype=np.int64)
    order = np.lexsort((rows, -values))
    return torch.from_numpy(order).to(device=scores.device, dtype=torch.long)


def plackett_luce_log_prob(logits, permutation):
    if logits.ndim != 1 or permutation.ndim != 1:
        raise ValueError("logits and permutation must be one-dimensional")
    if logits.numel() == 0 or logits.numel() != permutation.numel():
        raise ValueError("permutation length must equal the non-empty logits length")
    ordered_indices = permutation.detach().cpu().numpy()
    if not np.array_equal(np.sort(ordered_indices), np.arange(logits.numel())):
        raise ValueError("permutation must contain every row exactly once")
    ordered = logits[permutation]
    reverse_denominators = torch.logcumsumexp(torch.flip(ordered, dims=(0,)), dim=0)
    denominators = torch.flip(reverse_denominators, dims=(0,))
    return (ordered - denominators).sum()
