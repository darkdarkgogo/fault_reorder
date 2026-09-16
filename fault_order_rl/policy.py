"""Dynamic categorical actions and legacy listwise ranking operations."""

import numpy as np
import torch

from fault_order_rl.model import build_dynamic_features


def centered_logits(scores, temperature):
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = (scores - scores.mean()) / float(temperature)
    if not torch.isfinite(logits).all():
        raise ValueError("temperature-scaled logits contain non-finite values")
    return logits


def select_categorical_action(scores, remaining_rows, temperature, stochastic,
                              generator=None):
    """Select a catalog row from current candidates, with stable evaluation ties."""
    logits = centered_logits(scores, temperature)
    rows = torch.as_tensor(remaining_rows, device=scores.device)
    if rows.ndim != 1 or rows.numel() != scores.numel():
        raise ValueError("remaining rows must match the one-dimensional scores")
    if rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("remaining rows must contain integer indices")
    if (rows < 0).any() or torch.unique(rows).numel() != rows.numel():
        raise ValueError("remaining rows must be non-negative and unique")
    if stochastic:
        local = torch.multinomial(torch.softmax(logits, 0), 1,
                                  generator=generator).item()
    else:
        maximum = torch.max(scores)
        tied_rows = rows[scores == maximum]
        return int(torch.min(tied_rows).item())
    return int(rows[local].item())


def trajectory_log_prob(model, embeddings, decisions, temperature):
    """Replay saved remaining sets under autograd and sum categorical log-probs."""
    if not decisions:
        raise ValueError("trajectory must contain at least one decision")
    step_log_probs = []
    for decision in decisions:
        rows = torch.as_tensor(decision["remaining_rows"], device=embeddings.device)
        features = build_dynamic_features(embeddings, rows)
        selected = decision["selected_row"]
        local_indices = (rows == selected).nonzero(as_tuple=True)[0]
        if local_indices.numel() != 1:
            raise ValueError("selected row must occur exactly once in remaining rows")
        logits = centered_logits(model(features), temperature)
        step_log_probs.append(torch.log_softmax(logits, 0)[local_indices[0]])
    return torch.stack(step_log_probs).sum()


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
