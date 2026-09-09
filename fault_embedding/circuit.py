"""Strict binary BENCH reader and deterministic AND/NOT feature graph."""

from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Gate:
    name: str
    kind: str
    inputs: tuple[str, ...]


@dataclass
class Circuit:
    inputs: list[str]
    outputs: list[str]
    gates: dict[str, Gate]
    order: list[str]
    levels: dict[str, int]
    sha256: str

    def atpg_inputs(self, name):
        # PODEM/src/level.cpp swaps only when the first level is larger.
        return tuple(sorted(self.gates[name].inputs, key=self.levels.__getitem__))


def read_bench(path):
    raw = Path(path).read_bytes()
    inputs, outputs, gates = [], [], {}
    for line_no, raw_line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        port = re.fullmatch(r"(INPUT|OUTPUT)\(\s*([^\s(),=]+)\s*\)", line, re.I)
        if port:
            dest = inputs if port[1].upper() == "INPUT" else outputs
            if port[2] in dest:
                raise ValueError(f"Line {line_no}: duplicate port {port[2]}")
            dest.append(port[2])
            continue
        match = re.fullmatch(r"([^\s(),=]+)\s*=\s*(\w+)\(([^()]*)\)", line)
        if not match:
            raise ValueError(f"Line {line_no}: unsupported BENCH statement {line!r}")
        name, kind, args = match.groups()
        kind = {"BUFF": "BUF", "EQV": "XNOR"}.get(kind.upper(), kind.upper())
        pins = tuple(pin.strip() for pin in args.split(","))
        arity = 1 if kind in ("NOT", "BUF") else 2
        if kind not in ("AND", "NAND", "OR", "NOR", "XOR", "XNOR", "NOT", "BUF"):
            raise ValueError(f"Line {line_no}: unsupported gate {kind}")
        if len(pins) != arity or any(not re.fullmatch(r"[^\s(),=]+", p) for p in pins):
            raise ValueError(f"Line {line_no}: {kind} requires {arity} inputs; use a binary BENCH")
        if name in gates:
            raise ValueError(f"Line {line_no}: duplicate gate {name}")
        gates[name] = Gate(name, kind, pins)
    if not inputs or not outputs:
        raise ValueError("BENCH must have INPUT and OUTPUT declarations")
    if set(inputs) & gates.keys():
        raise ValueError("A signal cannot be both a primary input and a gate output")
    defined = set(inputs) | gates.keys()
    for name in outputs + [pin for gate in gates.values() for pin in gate.inputs]:
        if name not in defined:
            raise ValueError(f"Undefined signal {name}")
    # Kahn's algorithm avoids recursion limits on large scan circuits.
    rank = {name: i for i, name in enumerate(gates)}
    fanout = {name: [] for name in defined}
    remaining = {}
    for gate in gates.values():
        remaining[gate.name] = sum(pin in gates for pin in gate.inputs)
        for pin in gate.inputs:
            fanout[pin].append(gate.name)
    queue = [(rank[name], name) for name in gates if remaining[name] == 0]
    heapq.heapify(queue)
    order, levels = [], dict.fromkeys(inputs, 0)
    while queue:
        _, name = heapq.heappop(queue)
        order.append(name)
        levels[name] = 1 + max(levels[pin] for pin in gates[name].inputs)
        for target in fanout[name]:
            remaining[target] -= 1
            if remaining[target] == 0:
                heapq.heappush(queue, (rank[target], target))
    if len(order) != len(gates):
        raise ValueError("BENCH contains a combinational cycle")
    return Circuit(inputs, outputs, gates, order, levels, hashlib.sha256(raw).hexdigest())


def graph_digest(graph):
    content = {key: value for key, value in graph.items() if key != "graph_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_graph(circuit):
    """Keep every signal's anchor; only add AND/NOT feature nodes, never faults."""
    nodes, edges, anchors = [], [], {}

    def add(kind, pins, origin=None):
        idx = len(nodes)
        nodes.append({"id": idx, "name": f"aig_{idx}", "kind": kind,
                      "inputs": list(pins), "origin": origin,
                      "level": 0 if not pins else 1 + max(nodes[p]["level"] for p in pins)})
        edges.extend([p, idx] for p in pins)
        return idx

    def invert(pin):
        return add("NOT", [pin])

    def conjunction(left, right):
        return add("AND", [left, right])

    for name in circuit.inputs:
        anchors[name] = add("INPUT", [], name)
    for name in circuit.order:
        gate = circuit.gates[name]
        pins = [anchors[p] for p in gate.inputs]
        left = pins[0]
        if gate.kind == "NOT":
            result = invert(left)
        elif gate.kind == "BUF":
            result = invert(invert(left))
        else:
            right = pins[1]
            if gate.kind in ("AND", "NAND"):
                result = conjunction(left, right)
                if gate.kind == "NAND":
                    result = invert(result)
            elif gate.kind in ("OR", "NOR"):
                result = conjunction(invert(left), invert(right))
                if gate.kind == "OR":
                    result = invert(result)
            else:
                # XOR = NOT(AND(NOT(a AND NOT b), NOT(NOT a AND b))).
                first = conjunction(left, invert(right))
                second = conjunction(invert(left), right)
                result = conjunction(invert(first), invert(second))
                if gate.kind == "XOR":
                    result = invert(result)
        nodes[result]["origin"] = name
        anchors[name] = result
    # PODEM numbers dummy PI gates first, followed by dummy PO gates.
    aliases = {}
    for index, name in enumerate(circuit.inputs, 1):
        alias = f"dummy_gate{index}"
        if alias in anchors:
            raise ValueError(f"Signal name collides with PODEM boundary name {alias}")
        aliases[alias] = {"kind": "INPUT", "signal": name, "anchor": anchors[name]}
    for index, name in enumerate(circuit.outputs, len(circuit.inputs) + 1):
        alias = f"dummy_gate{index}"
        if alias in anchors:
            raise ValueError(f"Signal name collides with PODEM boundary name {alias}")
        # Independent PO boundary embedding for faults on a dummy PO's GI.
        anchor = invert(invert(anchors[name]))
        nodes[anchor]["origin"] = alias
        aliases[alias] = {"kind": "OUTPUT", "signal": name, "anchor": anchor}
    graph = {"schema": "fault_embedding_graph_v1", "bench_sha256": circuit.sha256,
             "nodes": nodes, "edges": edges, "anchors": anchors, "boundary_aliases": aliases,
             "inputs": circuit.inputs, "outputs": circuit.outputs}
    graph["graph_sha256"] = graph_digest(graph)
    return graph
