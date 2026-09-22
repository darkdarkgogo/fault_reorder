"""Primary selection and BFS-filtered DTC ranking statistics."""

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


def _rows_tensor(rows, device, name):
    value = torch.as_tensor(rows, device=device, dtype=torch.long)
    if (value.ndim != 1 or value.numel() == 0
            or torch.unique(value).numel() != value.numel()
            or (value < 0).any()):
        raise ValueError("{} must be non-empty, non-negative and unique".format(name))
    return value


def _row_positions(remaining, selected, name):
    positions = {int(row): index for index, row in enumerate(remaining.tolist())}
    try:
        return torch.as_tensor(
            [positions[int(row)] for row in selected.tolist()],
            device=remaining.device, dtype=torch.long)
    except KeyError as exc:
        raise ValueError("{} contains a row outside remaining rows".format(name)) from exc


def sample_primary(scores, remaining_rows, temperature, stochastic,
                   generator=None):
    """Select one Primary catalog row from the complete remaining set."""
    logits = centered_logits(scores, temperature)
    remaining = _rows_tensor(remaining_rows, scores.device, "remaining rows")
    if remaining.numel() != scores.numel():
        raise ValueError("remaining rows must match scores")
    if stochastic:
        local = torch.multinomial(
            torch.softmax(logits, dim=0), 1, generator=generator).squeeze(0)
        return remaining[local]
    values = scores.detach().cpu().numpy()
    rows = remaining.detach().cpu().numpy()
    local = int(np.lexsort((rows, -values))[0])
    return remaining[local]


def sample_candidate_ranking(scores, remaining_rows, candidate_rows,
                             temperature, stochastic, generator=None):
    """Order exactly one native BFS candidate batch using cached scores."""
    centered_logits(scores, temperature)
    remaining = _rows_tensor(remaining_rows, scores.device, "remaining rows")
    if remaining.numel() != scores.numel():
        raise ValueError("remaining rows must match scores")
    candidates = _rows_tensor(candidate_rows, scores.device, "candidate rows")
    positions = _row_positions(remaining, candidates, "candidate rows")
    candidate_scores = scores[positions]
    if stochastic:
        local_order, _ = sample_permutation(
            candidate_scores, temperature, generator)
        return candidates[local_order]
    values = candidate_scores.detach().cpu().numpy()
    rows = candidates.detach().cpu().numpy()
    order = np.lexsort((rows, -values))
    return candidates[torch.from_numpy(order).to(device=candidates.device)]


def _conditional_prefix_stats(logits, available_rows, executed_rows):
    available = [int(row) for row in available_rows.tolist()]
    executed = [int(row) for row in executed_rows.tolist()]
    if len(executed) != len(set(executed)):
        raise ValueError("executed prefix must contain unique rows")
    log_probs = []
    entropies = []
    row_to_logit = {
        int(row): logits[index] for index, row in enumerate(available_rows.tolist())
    }
    for selected in executed:
        if selected not in available:
            raise ValueError("executed row is not available in its action mask")
        conditional_logits = torch.stack([row_to_logit[row] for row in available])
        conditional_log_probs = torch.log_softmax(conditional_logits, dim=0)
        conditional_probs = torch.exp(conditional_log_probs)
        selected_position = available.index(selected)
        log_probs.append(conditional_log_probs[selected_position])
        entropies.append(-(conditional_probs * conditional_log_probs).sum())
        available.remove(selected)
    return log_probs, entropies


def joint_action_stats_from_scores(scores, value, remaining_rows, primary_row,
                                   dtc_batches, temperature):
    """Compute one step's joint policy terms from one cached model output."""
    logits = centered_logits(scores, temperature)
    remaining = _rows_tensor(remaining_rows, scores.device, "remaining rows")
    if remaining.numel() != scores.numel():
        raise ValueError("remaining rows must match scores")
    primary = torch.as_tensor(
        [primary_row], device=scores.device, dtype=torch.long)
    primary_log_probs, entropies = _conditional_prefix_stats(
        logits, remaining, primary)
    log_probs = list(primary_log_probs)
    attempted = set()
    for batch in dtc_batches:
        candidates = _rows_tensor(
            batch["bfs_candidate_rows"], scores.device, "BFS candidate rows")
        requested = _rows_tensor(
            batch["requested_rows"], scores.device, "requested rows")
        executed_values = batch["executed_prefix_rows"]
        executed = torch.as_tensor(
            executed_values, device=scores.device, dtype=torch.long)
        if executed.ndim != 1 or torch.unique(executed).numel() != executed.numel():
            raise ValueError("executed prefix rows must be one-dimensional and unique")
        if (set(requested.tolist()) != set(candidates.tolist())
                or requested.numel() != candidates.numel()):
            raise ValueError("requested rows must permute the BFS candidate rows")
        if executed.tolist() != requested[:executed.numel()].tolist():
            raise ValueError("executed rows must be a requested-order prefix")
        if int(primary_row) in set(candidates.tolist()):
            raise ValueError("Primary row cannot be a DTC candidate")
        candidate_positions = _row_positions(
            remaining, candidates, "BFS candidate rows")
        if attempted & set(executed.tolist()):
            raise ValueError("DTC executed rows repeat across batches")
        attempted.update(executed.tolist())
        batch_log_probs, batch_entropies = _conditional_prefix_stats(
            logits[candidate_positions], candidates, executed)
        log_probs.extend(batch_log_probs)
        entropies.extend(batch_entropies)
    joint_log_probability = torch.stack(log_probs).sum()
    mean_entropy = torch.stack(entropies).mean()
    if (not torch.isfinite(joint_log_probability)
            or not torch.isfinite(mean_entropy)
            or not torch.isfinite(value)):
        raise ValueError("joint policy statistics must be finite")
    return joint_log_probability, mean_entropy, value


def joint_action_stats(model, embeddings, remaining_rows, primary_row,
                       dtc_batches, temperature):
    """Replay one complete Primary/DTC transition with one model forward."""
    remaining = _rows_tensor(remaining_rows, embeddings.device, "remaining rows")
    features = build_dynamic_features(embeddings, remaining)
    scores, value = model(features)
    return joint_action_stats_from_scores(
        scores, value, remaining_rows, primary_row, dtc_batches, temperature)


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
