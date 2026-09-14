"""Validated conversion of trusted DeepTPI graph NPZ files to BENCH."""

import hashlib
import heapq
from pathlib import Path
import re

import numpy as np


GATE_NAMES = {0: "INPUT", 1: "AND", 2: "NOT", 3: "BUFF"}
EXPECTED_FANIN = {0: 0, 1: 2, 2: 1, 3: 1}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graphs(path, limit=None):
    path = Path(path)
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("circuit limit must be a positive integer")
    try:
        with np.load(str(path), allow_pickle=True) as arrays:
            if set(arrays.files) != {"circuits"}:
                raise ValueError("DeepTPI NPZ must contain only the circuits field")
            circuits = arrays["circuits"].item()
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("cannot read DeepTPI graph NPZ {}: {}".format(path, exc))
    if not isinstance(circuits, dict) or not circuits:
        raise ValueError("DeepTPI circuits must be a non-empty dictionary")
    items = list(circuits.items())
    if limit is not None:
        if len(items) < limit:
            raise ValueError("DeepTPI NPZ has fewer than {} circuits".format(limit))
        items = items[:limit]
    for name, graph in items:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", str(name)):
            raise ValueError("unsafe DeepTPI circuit name: {}".format(name))
        validate_graph(str(name), graph)
    return [(str(name), graph) for name, graph in items]


def validate_graph(name, graph):
    if not isinstance(graph, dict) or set(graph) != {"x", "edge_index"}:
        raise ValueError("{} must contain exactly x and edge_index".format(name))
    x = np.asarray(graph["x"])
    edges = np.asarray(graph["edge_index"])
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError("{} x must have shape [N, >=2]".format(name))
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("{} x must be finite numeric data".format(name))
    node_ids = x[:, 0]
    if not np.array_equal(node_ids, np.arange(len(x))):
        raise ValueError("{} node IDs must equal their row indices".format(name))
    gate_values = x[:, 1]
    gate_types = gate_values.astype(np.int64)
    if not np.array_equal(gate_values, gate_types) or not set(gate_types).issubset(GATE_NAMES):
        raise ValueError("{} contains an unsupported gate type".format(name))
    if edges.ndim != 2 or edges.shape[1:] != (2,):
        raise ValueError("{} edge_index must have shape [E, 2]".format(name))
    if not np.issubdtype(edges.dtype, np.number) or not np.isfinite(edges).all():
        raise ValueError("{} edge_index must be finite numeric data".format(name))
    integer_edges = edges.astype(np.int64)
    if not np.array_equal(edges, integer_edges):
        raise ValueError("{} edge endpoints must be integers".format(name))
    if integer_edges.size and (integer_edges.min() < 0 or integer_edges.max() >= len(x)):
        raise ValueError("{} edge endpoint is outside the node range".format(name))
    edge_pairs = [tuple(values) for values in integer_edges.tolist()]
    if len(edge_pairs) != len(set(edge_pairs)):
        raise ValueError("{} contains duplicate edges".format(name))

    fanins = [[] for _ in range(len(x))]
    fanouts = [[] for _ in range(len(x))]
    for source, destination in edge_pairs:
        fanins[destination].append(source)
        fanouts[source].append(destination)
    for index, gate_type in enumerate(gate_types):
        expected = EXPECTED_FANIN[int(gate_type)]
        if len(fanins[index]) != expected:
            raise ValueError(
                "{} node {} ({}) has {} fanins; expected {}".format(
                    name, index, GATE_NAMES[int(gate_type)], len(fanins[index]), expected))

    indegree = [len(values) for values in fanins]
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topological = []
    while ready:
        source = heapq.heappop(ready)
        topological.append(source)
        for destination in fanouts[source]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                heapq.heappush(ready, destination)
    if len(topological) != len(x):
        raise ValueError("{} contains a combinational cycle".format(name))
    outputs = [index for index, values in enumerate(fanouts) if not values]
    if not outputs:
        raise ValueError("{} has no primary output".format(name))
    return gate_types, fanins, topological, outputs


def graph_to_bench(name, graph):
    gate_types, fanins, topological, outputs = validate_graph(name, graph)
    inputs = [index for index, gate_type in enumerate(gate_types) if gate_type == 0]
    lines = [
        "# DeepTPI graph converted for reorderATPG",
        "# Circuit: {}".format(name),
        "# {} inputs".format(len(inputs)),
        "# {} outputs".format(len(outputs)),
        "",
    ]
    lines.extend("INPUT(N{})".format(index) for index in inputs)
    lines.append("")
    lines.extend("OUTPUT(N{})".format(index) for index in outputs)
    lines.append("")
    for index in topological:
        gate_type = int(gate_types[index])
        if gate_type == 0:
            continue
        inputs_text = ", ".join("N{}".format(source) for source in fanins[index])
        lines.append("N{} = {}({})".format(index, GATE_NAMES[gate_type], inputs_text))
    lines.append("")
    return "\n".join(lines)
