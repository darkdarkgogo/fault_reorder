import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PORT_RE = re.compile(
    r"^\s*(INPUT|OUTPUT)\s*\(\s*([^()]+?)\s*\)\s*(?:#.*)?$",
    re.IGNORECASE,
)
GATE_RE = re.compile(
    r"^\s*([A-Za-z0-9_.\[\]]+)\s*=\s*([A-Za-z][A-Za-z0-9_]*)"
    r"\s*\(([^()]*)\)\s*(?:#.*)?$"
)
EXPANDED_XOR_RE = re.compile(
    r"^\s*#\s*([A-Za-z0-9_.\[\]]+)\s*=\s*XOR\s*\(\s*"
    r"([A-Za-z0-9_.\[\]]+)\s*,\s*([A-Za-z0-9_.\[\]]+)\s*\)\s*$",
    re.IGNORECASE,
)
EXPANDED_XOR_CANDIDATE_RE = re.compile(
    r"^\s*#\s*[A-Za-z0-9_.\[\]]+\s*=\s*XOR\b",
    re.IGNORECASE,
)
SUPPORTED_TYPES = {"AND", "OR", "NAND", "NOR", "NOT", "BUF", "BUFF", "XOR", "EQV"}
ASSOCIATIVE_TYPES = {"AND", "OR", "NAND", "NOR"}
SYNTHETIC_PREFIX = "__smartatpg_bin_"


def _native_circuit_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    try:
        relative = os.path.relpath(resolved, Path.cwd())
    except ValueError:
        return str(resolved)
    return relative if relative.isascii() else str(resolved)


def catalog_cpp_podem(
    circuit_path: Path, fault_map_path: Optional[Path] = None
) -> dict[str, Any]:
    import cpp_podem

    return dict(cpp_podem.catalog_stuck_at(
        _native_circuit_path(circuit_path),
        _native_circuit_path(fault_map_path) if fault_map_path else "",
    ))


@dataclass(frozen=True)
class Gate:
    output: str
    gate_type: str
    inputs: tuple[str, ...]
    line_no: int


@dataclass(frozen=True)
class ExpandedXorCell:
    output: str
    inputs: tuple[str, str]
    private_outputs: tuple[str, str, str, str]
    line_no: int


def _fnv1a_file_hash(path: Path) -> str:
    value = 14695981039346656037
    for byte in path.read_bytes():
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _parse_bench(
    source: Path, allow_synthetic_names: bool = False
) -> tuple[list[str], list[str], list[Gate]]:
    ports: list[str] = []
    comments: list[str] = []
    gates: list[Gate] = []
    names: set[str] = set()
    outputs: set[str] = set()

    for line_no, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            comments.append(raw_line)
            continue
        port_match = PORT_RE.fullmatch(raw_line)
        if port_match:
            kind, name = port_match.groups()
            ports.append(f"{kind.upper()}({name})")
            names.add(name)
            continue
        gate_match = GATE_RE.fullmatch(raw_line)
        if not gate_match:
            raise ValueError(f"Malformed BENCH record on line {line_no}: {raw_line}")
        output, gate_type, input_text = gate_match.groups()
        gate_type = gate_type.upper()
        if gate_type == "BUF":
            gate_type = "BUFF"
        inputs = tuple(part.strip() for part in input_text.split(",") if part.strip())
        if gate_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported gate type '{gate_type}' on line {line_no}.")
        if not inputs:
            raise ValueError(f"Gate '{output}' has no inputs on line {line_no}.")
        if output in outputs:
            raise ValueError(f"Duplicate gate output '{output}' on line {line_no}.")
        expected_fanin = 1 if gate_type in {"NOT", "BUFF"} else None
        if expected_fanin is not None and len(inputs) != expected_fanin:
            raise ValueError(f"Gate '{output}' must have one input.")
        if gate_type in {"XOR", "EQV"} and len(inputs) != 2:
            raise ValueError(
                f"Gate '{output}' uses {len(inputs)}-input {gate_type}; only two-input "
                "XOR/EQV semantics are supported."
            )
        outputs.add(output)
        names.add(output)
        names.update(inputs)
        gates.append(Gate(output, gate_type, inputs, line_no))

    if not gates:
        raise ValueError(f"No gates found in {source}.")
    if not allow_synthetic_names and any(
        name.startswith(SYNTHETIC_PREFIX) for name in names
    ):
        raise ValueError(
            f"Source uses reserved synthetic prefix '{SYNTHETIC_PREFIX}'."
        )
    return comments, ports, gates


def _expanded_xor_cells(
    source: Path, ports: list[str], gates: list[Gate]
) -> list[ExpandedXorCell]:
    gates_by_output = {gate.output: gate for gate in gates}
    output_ports = {
        match.group(2)
        for port in ports
        if (match := PORT_RE.fullmatch(port)) and match.group(1).upper() == "OUTPUT"
    }
    fanouts: dict[str, list[str]] = {}
    for gate in gates:
        for input_name in gate.inputs:
            fanouts.setdefault(input_name, []).append(gate.output)

    cells: list[ExpandedXorCell] = []
    seen_outputs: set[str] = set()
    for line_no, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = EXPANDED_XOR_RE.fullmatch(raw_line)
        if not match:
            if EXPANDED_XOR_CANDIDATE_RE.match(raw_line):
                raise ValueError(
                    f"Malformed expanded XOR declaration on line {line_no}: "
                    f"{raw_line.strip()}"
                )
            continue
        output, first_input, second_input = match.groups()
        if output in seen_outputs:
            raise ValueError(
                f"Duplicate expanded XOR declaration for '{output}' on line {line_no}."
            )
        seen_outputs.add(output)
        if not output.startswith("G") or len(output) == 1:
            raise ValueError(
                f"Expanded XOR output '{output}' on line {line_no} must use a G suffix."
            )

        suffix = output[1:]
        w_name, z_name = f"W{suffix}", f"Z{suffix}"
        x_name, y_name = f"X{suffix}", f"Y{suffix}"
        expected = {
            w_name: ("NOT", (first_input,)),
            z_name: ("NOT", (second_input,)),
            x_name: ("NAND", (first_input, z_name)),
            y_name: ("NAND", (second_input, w_name)),
            output: ("NAND", (x_name, y_name)),
        }
        for gate_name, (gate_type, gate_inputs) in expected.items():
            gate = gates_by_output.get(gate_name)
            if (
                gate is None
                or gate.gate_type != gate_type
                or gate.inputs != gate_inputs
            ):
                raise ValueError(
                    f"Expanded XOR '{output}' on line {line_no} does not match its "
                    f"expected {gate_name} implementation."
                )

        expected_private_fanouts = {
            w_name: [y_name],
            z_name: [x_name],
            x_name: [output],
            y_name: [output],
        }
        for private_name, expected_fanouts in expected_private_fanouts.items():
            if (
                private_name in output_ports
                or fanouts.get(private_name, []) != expected_fanouts
            ):
                raise ValueError(
                    f"Expanded XOR '{output}' private wire '{private_name}' "
                    "is not private."
                )
        cells.append(
            ExpandedXorCell(
                output=output,
                inputs=(first_input, second_input),
                private_outputs=(w_name, z_name, x_name, y_name),
                line_no=line_no,
            )
        )
    return cells


def _filter_expanded_xor_faults(
    catalog: dict[str, Any], cells: list[ExpandedXorCell], gates: list[Gate],
    ports: list[str],
) -> tuple[dict[str, Any], int, int, int]:
    private_outputs = {
        output for cell in cells for output in cell.private_outputs
    }
    xor_outputs = {cell.output for cell in cells}
    source_faults = [dict(fault) for fault in catalog["faults"]]
    fanouts: dict[str, list[Gate]] = {}
    for gate in gates:
        for input_name in gate.inputs:
            fanouts.setdefault(input_name, []).append(gate)

    private_gate_outputs = private_outputs | xor_outputs
    logical_fanout_count: dict[str, int] = {}

    def add_logical_fanout(wire_name: str) -> None:
        logical_fanout_count[wire_name] = (
            logical_fanout_count.get(wire_name, 0) + 1
        )

    for gate in gates:
        if gate.output in private_gate_outputs:
            continue
        for input_name in gate.inputs:
            add_logical_fanout(input_name)
    for cell in cells:
        for input_name in cell.inputs:
            add_logical_fanout(input_name)
    for port in ports:
        match = PORT_RE.fullmatch(port)
        if match and match.group(1).upper() == "OUTPUT":
            add_logical_fanout(match.group(2))

    def find_output_fault(output: str, fault_type: int) -> Optional[int]:
        return next(
            (
                index
                for index, fault in enumerate(source_faults)
                if str(fault["node_name"]) == output
                and int(fault["io"]) == 1
                and int(fault["fault_type"]) == fault_type
            ),
            None,
        )

    def collapsed_representative(output: str, fault_type: int) -> int:
        wire_name = output
        stuck_value = fault_type
        visited: set[tuple[str, int]] = set()
        while (wire_name, stuck_value) not in visited:
            visited.add((wire_name, stuck_value))
            representative = find_output_fault(wire_name, stuck_value)
            if representative is not None:
                return representative
            consumers = fanouts.get(wire_name, [])
            if len(consumers) != 1:
                break
            consumer = consumers[0]
            if stuck_value == 0 and consumer.gate_type in {"AND", "BUFF"}:
                next_value = 0
            elif stuck_value == 0 and consumer.gate_type in {"NAND", "NOT"}:
                next_value = 1
            elif stuck_value == 1 and consumer.gate_type in {"OR", "BUFF"}:
                next_value = 1
            elif stuck_value == 1 and consumer.gate_type in {"NOR", "NOT"}:
                next_value = 0
            else:
                break
            wire_name = consumer.output
            stuck_value = next_value
        raise ValueError(
            "Cannot locate the collapsed representative for "
            f"{output}:GO:sa{fault_type}."
        )

    added_output_faults = 0
    for cell in cells:
        for fault_type in (0, 1):
            if find_output_fault(cell.output, fault_type) is not None:
                continue
            representative_index = collapsed_representative(
                cell.output, fault_type
            )
            representative = source_faults[representative_index]
            representative_weight = int(representative["eqv_fault_num"])
            # The final NAND's SA1 class also contains the SA0 classes of its
            # two private X/Y inputs. SA0 has no such internal equivalents.
            collapsed_class_weight = 3 if fault_type == 1 else 1
            if representative_weight <= collapsed_class_weight:
                raise ValueError(
                    f"Collapsed representative for {cell.output}:GO:sa{fault_type} "
                    "has no removable equivalent-fault weight."
                )
            representative["eqv_fault_num"] = (
                representative_weight - collapsed_class_weight
            )
            output_fault = {
                "fault_id": f"{cell.output}:GO:sa{fault_type}",
                "node_name": cell.output,
                "input_wire_name": "-",
                "io": 1,
                "input_occurrence": -1,
                "fault_type": fault_type,
                "eqv_fault_num": 1,
            }
            sibling_indices = [
                index
                for index, fault in enumerate(source_faults)
                if str(fault["node_name"]) == cell.output and int(fault["io"]) == 1
            ]
            if sibling_indices:
                insertion_index = (
                    min(sibling_indices)
                    if fault_type == 0
                    else max(sibling_indices) + 1
                )
            else:
                insertion_index = representative_index
            source_faults.insert(insertion_index, output_fault)
            added_output_faults += 1

    filtered_faults: list[dict[str, Any]] = []
    removed = 0

    for fault in source_faults:
        node_name = str(fault["node_name"])
        io = int(fault["io"])
        if node_name in private_outputs or (node_name in xor_outputs and io != 1):
            removed += 1
            continue
        if node_name in xor_outputs:
            fault["eqv_fault_num"] = 1
        filtered_faults.append(fault)

    retained_input_faults = 0
    for cell in cells:
        input_occurrences: dict[str, int] = {}
        for input_index, input_name in enumerate(cell.inputs):
            input_occurrence = input_occurrences.get(input_name, 0)
            input_occurrences[input_name] = input_occurrence + 1
            if logical_fanout_count.get(input_name, 0) <= 1:
                continue
            for fault_type in (0, 1):
                filtered_faults.append({
                    "fault_id": (
                        f"{cell.output}:GI{input_index}:sa{fault_type}"
                    ),
                    "node_name": cell.output,
                    "input_wire_name": input_name,
                    "io": 0,
                    "input_index": input_index,
                    "input_occurrence": input_occurrence,
                    "fault_type": fault_type,
                    "eqv_fault_num": 1,
                    "logical_xor_input": True,
                })
                retained_input_faults += 1

    for output in xor_outputs:
        output_faults = [
            fault
            for fault in filtered_faults
            if str(fault["node_name"]) == output and int(fault["io"]) == 1
        ]
        if (
            len(output_faults) != 2
            or {int(fault["fault_type"]) for fault in output_faults} != {0, 1}
            or any(int(fault["eqv_fault_num"]) != 1 for fault in output_faults)
        ):
            raise ValueError(
                f"Expanded XOR '{output}' must retain exactly GO-SA0 and GO-SA1."
            )

    return (
        {
            "faults": filtered_faults,
            "uncollapsed_total": sum(
                int(fault["eqv_fault_num"]) for fault in filtered_faults
            ),
        },
        removed,
        added_output_faults,
        retained_input_faults,
    )


def _binary_gate_lines(
    gate: Gate,
    gate_index: int,
) -> tuple[list[str], dict[tuple[str, str, int], tuple[str, str, int]], int]:
    if len(gate.inputs) <= 2:
        line = f"{gate.output} = {gate.gate_type}({','.join(gate.inputs)})"
        mapping = {}
        occurrences: dict[str, int] = {}
        for input_index, input_name in enumerate(gate.inputs):
            occurrence = occurrences.get(input_name, 0)
            occurrences[input_name] = occurrence + 1
            mapping[(gate.output, input_name, occurrence)] = (
                gate.output,
                input_name,
                occurrence,
            )
        return [line], mapping, 0
    if gate.gate_type not in ASSOCIATIVE_TYPES:
        raise ValueError(
            f"Gate '{gate.output}' cannot be decomposed as {gate.gate_type}."
        )

    base_type = "AND" if gate.gate_type in {"AND", "NAND"} else "OR"
    lines: list[str] = []
    mapping: dict[tuple[str, str, int], tuple[str, str, int]] = {}
    synthetic_count = 0

    source_occurrences: dict[str, int] = {}
    input_refs: list[tuple[str, int]] = []
    for input_name in gate.inputs:
        occurrence = source_occurrences.get(input_name, 0)
        source_occurrences[input_name] = occurrence + 1
        input_refs.append((input_name, occurrence))

    def build(inputs: tuple[tuple[str, int], ...], output: str, root: bool) -> str:
        nonlocal synthetic_count
        split = len(inputs) // 2
        child_wires: list[str] = []
        direct_inputs: list[tuple[str, int, int]] = []
        child_groups = (inputs[:split], inputs[split:])
        for group in child_groups:
            if len(group) == 1:
                input_index = len(child_wires)
                input_name, source_occurrence = group[0]
                child_wires.append(input_name)
                direct_inputs.append(
                    (input_name, source_occurrence, input_index)
                )
                continue
            synthetic_count += 1
            child_output = f"{SYNTHETIC_PREFIX}{gate_index}_{synthetic_count}"
            child_wires.append(build(group, child_output, False))
        for input_name, source_occurrence, input_index in direct_inputs:
            occurrence = child_wires[: input_index + 1].count(input_name) - 1
            mapping[(gate.output, input_name, source_occurrence)] = (
                output,
                input_name,
                occurrence,
            )
        current_type = gate.gate_type if root else base_type
        lines.append(f"{output} = {current_type}({','.join(child_wires)})")
        return output

    build(tuple(input_refs), gate.output, True)
    return lines, mapping, synthetic_count


def _write_fault_map(
    source: Path,
    destination: Path,
    fault_map_path: Path,
    catalog: dict[str, Any],
    gi_mapping: dict[tuple[str, str, int], tuple[str, str, int]],
) -> list[tuple[Any, ...]]:
    records: list[str] = []
    expected_records: list[tuple[Any, ...]] = []
    faults = list(catalog["faults"])
    has_logical_xor_inputs = any(
        bool(fault.get("logical_xor_input", False)) for fault in faults
    )
    for fault in faults:
        node_name = str(fault["node_name"])
        input_name = str(fault["input_wire_name"])
        input_occurrence = int(fault["input_occurrence"])
        if int(fault["io"]) == 0:
            node_name, input_name, input_occurrence = gi_mapping.get(
                (node_name, input_name, input_occurrence),
                (node_name, input_name, input_occurrence),
            )
        expected_records.append(
            (
                str(fault["fault_id"]),
                node_name,
                input_name,
                int(fault["io"]),
                input_occurrence,
                int(fault["fault_type"]),
                int(fault["eqv_fault_num"]),
            )
        )
        record = (
            "fault "
            f"{fault['fault_id']} {node_name} {input_name} {int(fault['io'])} "
            f"{input_occurrence} "
            f"{int(fault['fault_type'])} {int(fault['eqv_fault_num'])}"
        )
        if has_logical_xor_inputs:
            is_logical_xor_input = int(
                bool(fault.get("logical_xor_input", False))
            )
            logical_input_index = (
                int(fault["input_index"]) if is_logical_xor_input else -1
            )
            record += f" {is_logical_xor_input} {logical_input_index}"
        records.append(record)

    lines = [
        (
            "SMARTATPG_FAULT_MAP_V3"
            if has_logical_xor_inputs
            else "SMARTATPG_FAULT_MAP_V2"
        ),
        f"source_hash {_fnv1a_file_hash(source)}",
        f"circuit_hash {_fnv1a_file_hash(destination)}",
        f"count {len(records)}",
        f"uncollapsed_total {int(catalog['uncollapsed_total'])}",
        *records,
        "end",
        "",
    ]
    fault_map_path.parent.mkdir(parents=True, exist_ok=True)
    with fault_map_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines))
    return expected_records


def _verify_equivalence(
    source_ports: list[str],
    source_gates: list[Gate],
    binary_ports: list[str],
    binary_gates: list[Gate],
    vectors: int = 256,
) -> None:
    def port_names(ports: list[str], kind: str) -> list[str]:
        prefix = f"{kind}("
        return [port[len(prefix) : -1] for port in ports if port.startswith(prefix)]

    source_inputs = port_names(source_ports, "INPUT")
    source_outputs = port_names(source_ports, "OUTPUT")
    if source_inputs != port_names(binary_ports, "INPUT"):
        raise ValueError("Binary conversion changed primary inputs.")
    if source_outputs != port_names(binary_ports, "OUTPUT"):
        raise ValueError("Binary conversion changed primary outputs.")

    def evaluate(
        gates: list[Gate], input_values: dict[str, int], mask: int
    ) -> list[int]:
        drivers = {gate.output: gate for gate in gates}
        values = dict(input_values)
        visiting: set[str] = set()

        def wire_value(name: str) -> int:
            if name in values:
                return values[name]
            if name in visiting or name not in drivers:
                raise ValueError(f"Cannot resolve BENCH wire '{name}'.")
            visiting.add(name)
            gate = drivers[name]
            inputs = [wire_value(input_name) for input_name in gate.inputs]
            if gate.gate_type in {"AND", "NAND"}:
                value = mask
                for input_value in inputs:
                    value &= input_value
                if gate.gate_type == "NAND":
                    value = (~value) & mask
            elif gate.gate_type in {"OR", "NOR"}:
                value = 0
                for input_value in inputs:
                    value |= input_value
                if gate.gate_type == "NOR":
                    value = (~value) & mask
            elif gate.gate_type == "NOT":
                value = (~inputs[0]) & mask
            elif gate.gate_type == "BUFF":
                value = inputs[0]
            elif gate.gate_type == "XOR":
                value = inputs[0] ^ inputs[1]
            elif gate.gate_type == "EQV":
                value = (~(inputs[0] ^ inputs[1])) & mask
            else:
                raise ValueError(f"Cannot evaluate gate type '{gate.gate_type}'.")
            visiting.remove(name)
            values[name] = value
            return value

        return [wire_value(name) for name in source_outputs]

    rng = random.Random(14)
    remaining = vectors
    while remaining > 0:
        width = min(64, remaining)
        mask = (1 << width) - 1
        inputs = {name: rng.getrandbits(width) for name in source_inputs}
        if evaluate(source_gates, inputs, mask) != evaluate(binary_gates, inputs, mask):
            raise ValueError("Binary conversion failed random-vector PO equivalence.")
        remaining -= width


def convert_binary_bench(
    source: Path,
    destination: Optional[Path] = None,
    fault_map_path: Optional[Path] = None,
) -> dict[str, int]:
    source = source.resolve()
    destination = (
        destination.resolve()
        if destination
        else source.with_name(f"{source.stem}_binary{source.suffix}")
    )
    fault_map_path = (
        fault_map_path.resolve()
        if fault_map_path
        else destination.with_suffix(".faultmap")
    )
    if len({source, destination, fault_map_path}) != 3:
        raise ValueError(
            "Source, destination, and fault map must be different files."
        )

    comments, ports, gates = _parse_bench(source)
    expanded_xor_cells = _expanded_xor_cells(source, ports, gates)
    gate_lines: list[str] = []
    gi_mapping: dict[tuple[str, str, int], tuple[str, str, int]] = {}
    synthetic_gates = 0
    converted_gates = 0
    for gate_index, gate in enumerate(gates):
        lines, mapping, added = _binary_gate_lines(gate, gate_index)
        gate_lines.extend(lines)
        gi_mapping.update(mapping)
        synthetic_gates += added
        converted_gates += int(len(gate.inputs) > 2)

    output_lines = [
        "# Balanced two-input BENCH generated by convert_binary_bench.py",
        f"# Source: {source.name}",
        "# Original collapsed faults are restored through the companion .faultmap.",
        "",
        *comments,
        *ports,
        "",
        *gate_lines,
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(output_lines))

    _, binary_ports, binary_gates = _parse_bench(
        destination, allow_synthetic_names=True
    )
    if max(len(gate.inputs) for gate in binary_gates) > 2:
        raise ValueError("Binary conversion left a gate with fan-in greater than two.")
    _verify_equivalence(ports, gates, binary_ports, binary_gates)

    source_catalog = catalog_cpp_podem(source)
    (
        catalog,
        removed_xor_faults,
        added_xor_output_faults,
        retained_xor_input_faults,
    ) = (
        _filter_expanded_xor_faults(
            source_catalog, expanded_xor_cells, gates, ports
        )
    )
    expected_records = _write_fault_map(
        source, destination, fault_map_path, catalog, gi_mapping
    )
    mapped_catalog = catalog_cpp_podem(destination, fault_map_path)
    mapped_records = [
        (
            str(fault["fault_id"]),
            str(fault["node_name"]),
            str(fault["input_wire_name"]),
            int(fault["io"]),
            int(fault["input_occurrence"]),
            int(fault["fault_type"]),
            int(fault["eqv_fault_num"]),
        )
        for fault in mapped_catalog["faults"]
    ]
    if expected_records != mapped_records:
        raise ValueError("Mapped fault location, ID, or attributes changed.")
    if catalog["uncollapsed_total"] != mapped_catalog["uncollapsed_total"]:
        raise ValueError("Mapped uncollapsed fault total changed.")
    return {
        "original_gates": len(gates),
        "binary_gates": len(gate_lines),
        "converted_gates": converted_gates,
        "synthetic_gates": synthetic_gates,
        "expanded_xor_cells": len(expanded_xor_cells),
        "removed_xor_faults": removed_xor_faults,
        "added_xor_output_faults": added_xor_output_faults,
        "retained_xor_input_faults": retained_xor_input_faults,
        "source_collapsed_faults": len(source_catalog["faults"]),
        "source_uncollapsed_faults": int(source_catalog["uncollapsed_total"]),
        "collapsed_faults": len(catalog["faults"]),
        "uncollapsed_faults": int(catalog["uncollapsed_total"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert associative BENCH gates to balanced two-input trees."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--fault-map", type=Path)
    args = parser.parse_args()

    destination = args.destination or args.source.with_name(
        f"{args.source.stem}_binary{args.source.suffix}"
    )
    fault_map_path = args.fault_map or destination.with_suffix(".faultmap")
    stats = convert_binary_bench(args.source, destination, fault_map_path)
    print(
        "Converted binary BENCH: "
        f"gates={stats['original_gates']}->{stats['binary_gates']} "
        f"converted={stats['converted_gates']} synthetic={stats['synthetic_gates']} "
        f"xor_cells={stats['expanded_xor_cells']} "
        f"xor_faults_removed={stats['removed_xor_faults']} "
        f"xor_output_faults_added={stats['added_xor_output_faults']} "
        f"xor_input_faults_retained={stats['retained_xor_input_faults']} "
        f"faults={stats['collapsed_faults']}/{stats['uncollapsed_faults']}"
    )
    print(f"Circuit: {destination.resolve()}")
    print(f"Fault map: {fault_map_path.resolve()}")


if __name__ == "__main__":
    main()
