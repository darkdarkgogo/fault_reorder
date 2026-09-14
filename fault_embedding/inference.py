"""Official DeepGate2 inference; no permissive or random-weight fallback."""

import hashlib
import importlib
from pathlib import Path
import random
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "third_party" / "python-deepgate"
DEFAULT_CHECKPOINT = DEFAULT_SOURCE / "deepgate" / "pretrained" / "model.pth"
OFFICIAL_REVISION = "173db7529cefc97f7b9b2b3fa97ec1bd5773754d"
OFFICIAL_CHECKPOINT_SHA256 = "9bc4a0c1f8fc57cc3aa0498dd8737af561ca71c26ca5332288d12a91a308f4d5"
OFFICIAL_SOURCE_SHA256 = "b88833d4bf6979909e33f92b76fb0101e66a6dcdc12ec3b539d0b4a3fa19312c"
_MODEL_CACHE = {}


def source_digest(source):
    digest = hashlib.sha256()
    for path in sorted((source / "deepgate").rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def infer_gate_embeddings(graph, checkpoint=DEFAULT_CHECKPOINT, source=DEFAULT_SOURCE, seed=0):
    checkpoint, source = Path(checkpoint).resolve(), Path(source).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Pretrained checkpoint not found: {checkpoint}")
    if not (source / "deepgate" / "model.py").is_file():
        raise ValueError(f"Official python-deepgate source not found: {source}")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32-1")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint is not the pinned official DeepGate2 model")
    current_source_sha = source_digest(source)
    if current_source_sha != OFFICIAL_SOURCE_SHA256:
        raise ValueError("DeepGate2 source does not match the pinned official revision")
    try:
        import torch
        from torch_geometric.data import Data
        source_package = (source / "deepgate").resolve()
        for name, module in list(sys.modules.items()):
            if name == "deepgate" or name.startswith("deepgate."):
                module_file = getattr(module, "__file__", None)
                if not module_file:
                    raise ValueError(f"DeepGate module has no verifiable source file: {name}")
                try:
                    Path(module_file).resolve().relative_to(source_package)
                except ValueError as exc:
                    raise ValueError(f"Another DeepGate module is already imported: {name}") from exc
        if "deepgate" in sys.modules:
            package = sys.modules["deepgate"]
            paths = [Path(path).resolve() for path in getattr(package, "__path__", [])]
            if (source / "deepgate") not in paths:
                raise ValueError("Another deepgate package is already imported")
        else:
            # Import only the official encoder module. Upstream
            # deepgate/__init__.py eagerly imports its AIG parser and an
            # unrelated optional `aiger` dependency.
            package = types.ModuleType("deepgate")
            package.__path__ = [str(source / "deepgate")]
            package.__file__ = str(source / "deepgate" / "__init__.py")
            sys.modules["deepgate"] = package
            for child in ("utils", "arch"):
                subpackage = types.ModuleType(f"deepgate.{child}")
                subpackage.__path__ = [str(source / "deepgate" / child)]
                subpackage.__file__ = str(source / "deepgate" / child / "__init__.py")
                sys.modules[f"deepgate.{child}"] = subpackage
        model_module = importlib.import_module("deepgate.model")
        if Path(model_module.__file__).resolve() != (source_package / "model.py"):
            raise ValueError("DeepGate model module is not the pinned official model.py")
    except ImportError as exc:
        raise RuntimeError(
            f"Missing DeepGate2 inference dependency: {exc}. "
            "See docs/fault-embeddings.md; prepare works without PyTorch."
        ) from exc
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cache_key = (str(checkpoint), str(source), checkpoint_sha, current_source_sha)
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = model_module.Model()
            # Upstream load() silently replaces missing or mismatched tensors with
            # random values. This export requires the complete official checkpoint.
            try:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:  # PyTorch before 2.0, including the project's d2l env.
                saved = torch.load(checkpoint, map_location="cpu")
            if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
                raise ValueError("Checkpoint must contain a state_dict")
            state = {}
            for key, tensor in saved["state_dict"].items():
                key = key[7:] if key.startswith("module.") else key
                if key in state:
                    raise ValueError(f"Duplicate checkpoint parameter {key}")
                state[key] = tensor
            model.load_state_dict(state, strict=True)
            model.eval()
            _MODEL_CACHE[cache_key] = model
        kinds = {"INPUT": 0, "AND": 1, "NOT": 2}
        nodes = graph["nodes"]
        edge_index = torch.tensor(graph["edges"], dtype=torch.long).reshape(-1, 2).T.contiguous()
        backward = [0] * len(nodes)
        for node in reversed(nodes):
            for parent in node["inputs"]:
                backward[parent] = max(backward[parent], backward[node["id"]] + 1)
        gate = torch.tensor([[kinds[node["kind"]]] for node in nodes], dtype=torch.long)
        data = Data(edge_index=edge_index, gate=gate,
                    x=torch.nn.functional.one_hot(gate.flatten(), 3).float(),
                    forward_level=torch.tensor([node["level"] for node in nodes]),
                    backward_level=torch.tensor(backward),
                    forward_index=torch.arange(len(nodes)), backward_index=torch.arange(len(nodes)),
                    num_nodes=len(nodes))
        with torch.inference_mode():
            hs, hf = model(data)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
    arrays = {"hs": hs.cpu().numpy(), "hf": hf.cpu().numpy()}
    for name, array in arrays.items():
        if array.shape != (len(nodes), model.dim_hidden) or not np.isfinite(array).all():
            raise ValueError(f"DeepGate2 produced invalid {name} embeddings")
    provenance = {"backend": "official-python-deepgate", "checkpoint": str(checkpoint),
                  "checkpoint_sha256": checkpoint_sha,
                  "is_known_official_checkpoint": True,
                  "source_path": str(source), "source_sha256": current_source_sha,
                  "downloaded_source_revision": OFFICIAL_REVISION,
                  "seed": seed, "device": "cpu", "torch_version": str(torch.__version__),
                  "numpy_version": np.__version__}
    return arrays, provenance
