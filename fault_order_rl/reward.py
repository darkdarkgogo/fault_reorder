"""Strict coverage-first reward, GAE and PPO helpers."""

import math

import torch


def _positive_count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + " must be a positive integer")


def step_reward(pattern_increment, newly_detected_eqv, initial_eqv, alpha=0.1):
    _positive_count(initial_eqv, "initial_eqv")
    if pattern_increment not in (0, 1):
        raise ValueError("pattern_increment must be 0 or 1")
    if (isinstance(newly_detected_eqv, bool)
            or not isinstance(newly_detected_eqv, int)
            or newly_detected_eqv < 0):
        raise ValueError("newly_detected_eqv must be a non-negative integer")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    return (-pattern_increment + alpha * newly_detected_eqv) / initial_eqv


def target_return(patterns_after_stc, initial_eqv, coverage_shortfall, beta=10.0):
    _positive_count(initial_eqv, "initial_eqv")
    for value, name in ((patterns_after_stc, "patterns_after_stc"),
                        (coverage_shortfall, "coverage_shortfall")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a non-negative integer")
    if patterns_after_stc > initial_eqv:
        raise ValueError("patterns_after_stc exceeds initial_eqv")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if coverage_shortfall:
        return -beta * coverage_shortfall / initial_eqv
    return 1.0 - patterns_after_stc / initial_eqv


def terminal_correction(step_rewards, target):
    values = list(step_rewards)
    if not values:
        raise ValueError("episode must contain at least one step reward")
    if not math.isfinite(target) or any(not math.isfinite(value) for value in values):
        raise ValueError("reward values must be finite")
    return target - sum(values)


def compute_gae(rewards, values, gamma=1.0, gae_lambda=0.95):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    if rewards.ndim != 1 or values.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards and values must be non-empty vectors")
    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have identical shapes")
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError("rewards and values must be finite")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must be in [0, 1]")
    advantages = torch.zeros_like(rewards)
    next_value = rewards.new_zeros(())
    next_advantage = rewards.new_zeros(())
    for index in range(rewards.numel() - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    returns = advantages + values
    return returns, advantages


def normalize_advantages(advantages, eps=1e-8):
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    if advantages.ndim != 1 or advantages.numel() == 0:
        raise ValueError("advantages must be a non-empty vector")
    if not torch.isfinite(advantages).all():
        raise ValueError("advantages must be finite")
    if advantages.numel() < 2:
        return advantages.clone()
    std = advantages.std(unbiased=False)
    if std <= eps:
        return advantages.clone()
    return (advantages - advantages.mean()) / (std + eps)


def ppo_objective(new_log_probs, old_log_probs, advantages, values, returns,
                  entropies, clip_epsilon=0.2, value_coef=0.5,
                  entropy_coef=0.01):
    tensors = [torch.as_tensor(value, dtype=torch.float32) for value in (
        new_log_probs, old_log_probs, advantages, values, returns, entropies)]
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("PPO inputs must be vectors")
    if len({tensor.numel() for tensor in tensors}) != 1 or not tensors[0].numel():
        raise ValueError("PPO inputs must have the same non-zero length")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("PPO inputs must be finite")
    new_log_probs, old_log_probs, advantages, values, returns, entropies = tensors
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    actor_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    critic_loss = torch.mean((values - returns) ** 2)
    entropy = entropies.mean()
    total = actor_loss + value_coef * critic_loss - entropy_coef * entropy
    approx_kl = torch.mean((ratio - 1.0) - log_ratio)
    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_epsilon).float())
    return total, {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }
