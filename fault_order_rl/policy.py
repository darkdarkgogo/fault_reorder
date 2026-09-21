"""Dynamic full rankings and executed-prefix policy statistics."""

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


def sample_ranking(scores, remaining_rows, temperature, stochastic,
                   generator=None):
    """Return every remaining catalog row in sampled or stable score order."""
    logits = centered_logits(scores, temperature)
    rows = torch.as_tensor(remaining_rows, device=scores.device, dtype=torch.long)
    if rows.ndim != 1 or rows.numel() != scores.numel():
        raise ValueError("remaining rows must match scores")
    if torch.unique(rows).numel() != rows.numel() or (rows < 0).any():
        raise ValueError("remaining rows must be non-negative and unique")
    if stochastic:
        local_order, _ = sample_permutation(scores, temperature, generator)
        return rows[local_order]
    values = scores.detach().cpu().numpy()
    catalog_rows = rows.detach().cpu().numpy()
    order = np.lexsort((catalog_rows, -values))
    return rows[torch.from_numpy(order).to(device=rows.device)]


def executed_prefix_stats(model, embeddings, remaining_rows, executed_rows,
                          temperature):
    """Recompute joint prefix log-probability, mean entropy and state value."""
    remaining = torch.as_tensor(
        remaining_rows, device=embeddings.device, dtype=torch.long)
    executed = torch.as_tensor(
        executed_rows, device=embeddings.device, dtype=torch.long)
    if executed.ndim != 1 or executed.numel() == 0:
        raise ValueError("executed prefix must be non-empty")
    if torch.unique(executed).numel() != executed.numel():
        raise ValueError("executed prefix must contain unique rows")
    features = build_dynamic_features(embeddings, remaining)
    scores, value = model(features)
    logits = centered_logits(scores, temperature)
    available = list(range(remaining.numel()))
    log_probs, entropies = [], []
    for selected_row in executed.tolist():
        matches = [index for index in available
                   if int(remaining[index]) == selected_row]
        if len(matches) != 1:
            raise ValueError("executed row is not available in remaining rows")
        selected_local = matches[0]
        available_tensor = torch.as_tensor(
            available, device=logits.device, dtype=torch.long)
        conditional = logits[available_tensor]
        conditional_log_probs = torch.log_softmax(conditional, dim=0)
        conditional_probs = torch.exp(conditional_log_probs)
        position = available.index(selected_local)
        log_probs.append(conditional_log_probs[position])
        entropies.append(-(conditional_probs * conditional_log_probs).sum())
        available.remove(selected_local)
    return (torch.stack(log_probs).sum(), torch.stack(entropies).mean(), value)


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
