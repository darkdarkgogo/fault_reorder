"""Atomic local checkpoints and CPU random-state restoration."""

import inspect
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, state):
    atomic_write(path, lambda stream: torch.save(state, stream))


def load_checkpoint(path):
    # Training checkpoints include Python/NumPy RNG objects. Load only local,
    # trusted checkpoints produced by this package (not downloaded pickle files).
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    state = torch.load(str(path), **options)
    if isinstance(state, dict) and state.get("version") == 1:
        raise ValueError(
            "checkpoint schema 1 uses detected-only coverage; restart training"
        )
    if isinstance(state, dict) and state.get("version") == 2:
        raise ValueError(
            "checkpoint schema 2 uses a static permutation policy; retraining is required"
        )
    if not isinstance(state, dict) or state.get("version") != 3:
        raise ValueError("unsupported fault-order checkpoint")
    return state


def write_json(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(encoded))


def write_npz(path, **arrays):
    atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))
