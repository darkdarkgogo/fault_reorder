"""Resolve preserved SMARTATPG collapsed faults on a binary BENCH."""

import hashlib
import json
from pathlib import Path
import re


BUNDLED_BINDINGS = Path(__file__).with_name("faultmap_bindings.json")
BUNDLED_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "PODEM" / "sample_circuits"


def _verify_binding(path, circuit, headers, faultmap_sha256):
    sidecar = Path(str(path) + ".binding.json")
    if sidecar.is_file():
        binding_path = sidecar
        binding = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        if Path(path).resolve().parent != BUNDLED_SAMPLE_DIR.resolve():
            raise ValueError(
                f"No trusted BENCH binding for {Path(path).name}; add {sidecar.name}"
            )
        binding_path = BUNDLED_BINDINGS
        document = json.loads(binding_path.read_text(encoding="utf-8"))
        binding = document.get("bindings", {}).get(Path(path).name)
    if not isinstance(binding, dict):
        raise ValueError(
            f"No trusted BENCH binding for {Path(path).name}; add {sidecar.name}"
        )
    expected = {
        "bench_sha256": circuit.sha256,
        "faultmap_sha256": faultmap_sha256,
        "source_hash": headers["source_hash"],
        "circuit_hash": headers["circuit_hash"],
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"Fault map binding mismatch for {key}")
    return {
        "status": "verified",
        "binding_path": str(binding_path.resolve()),
        "binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
    }


def read_faultmap(path, circuit, graph):
    raw = Path(path).read_bytes()
    lines = [line.strip() for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if (not lines or lines[0] not in ("SMARTATPG_FAULT_MAP_V2", "SMARTATPG_FAULT_MAP_V3")
            or lines[-1] != "end"):
        raise ValueError("Expected a complete SMARTATPG_FAULT_MAP_V2/V3 file ending in 'end'")
    has_logical_xor_inputs = lines[0] == "SMARTATPG_FAULT_MAP_V3"
    headers, records, identities, location_counts = {}, [], set(), {}
    for line_no, line in enumerate(lines[1:-1], 2):
        parts = line.split()
        if parts[0] != "fault":
            if len(parts) != 2 or parts[0] not in ("source_hash", "circuit_hash", "count", "uncollapsed_total"):
                raise ValueError(f"Fault map line {line_no}: unknown header")
            if parts[0] in headers or records:
                raise ValueError(f"Fault map line {line_no}: duplicate or misplaced header")
            headers[parts[0]] = parts[1]
            continue
        expected_fields = 10 if has_logical_xor_inputs else 8
        if len(parts) != expected_fields:
            raise ValueError(f"Fault map line {line_no}: expected {expected_fields} fields")
        _, fault_id, target, source, io_text, selector_text, sa_text, count_text, *logical_fields = parts
        io, selector, sa, count = map(int, (io_text, selector_text, sa_text, count_text))
        logical_xor_input = False
        logical_input_index = -1
        if has_logical_xor_inputs:
            logical_marker, logical_input_index = map(int, logical_fields)
            if not ((logical_marker == 0 and logical_input_index == -1)
                    or (logical_marker == 1 and logical_input_index in (0, 1))):
                raise ValueError(f"Fault map line {line_no}: invalid logical XOR marker")
            logical_xor_input = logical_marker == 1
        identity = re.fullmatch(r"(.+):(GO|GI(\d+)):sa([01])(?:@dup\d+)?", fault_id)
        if not identity or io not in (0, 1) or sa not in (0, 1) or count < 1:
            raise ValueError(f"Invalid fault record {fault_id}")
        if int(identity[4]) != sa or (identity[2] == "GO") != bool(io):
            raise ValueError(f"Fault identity and type disagree: {fault_id}")
        if fault_id in identities:
            raise ValueError(f"Duplicate fault ID {fault_id}")
        identities.add(fault_id)
        alias = graph["boundary_aliases"].get(target)
        if target not in circuit.gates and alias is None:
            raise ValueError(f"Fault {fault_id}: target gate {target} does not exist")
        bench_pin = pin = -1
        if io == 1:
            if source != "-" or selector != -1 or (alias and alias["kind"] != "INPUT"):
                raise ValueError(f"Fault {fault_id}: invalid GO location")
            signal = alias["signal"] if alias else target
            anchor = graph["anchors"][signal]
            connected_anchor = anchor
        else:
            if alias:
                if alias["kind"] != "OUTPUT" or source != alias["signal"]:
                    raise ValueError(f"Fault {fault_id}: invalid boundary GI location")
                bench_pin = pin = 0
                anchor = alias["anchor"]
            elif logical_xor_input:
                if circuit.gates[target].kind != "NAND":
                    raise ValueError(f"Fault {fault_id}: logical XOR target must be its final NAND")
                if int(identity[3]) != logical_input_index:
                    raise ValueError(f"Fault {fault_id}: logical XOR input index disagrees with ID")
                if source not in graph["anchors"]:
                    raise ValueError(f"Fault {fault_id}: unknown logical XOR input {source}")
                bench_pin = pin = logical_input_index
                anchor = graph["anchors"][target]
            else:
                bench_inputs = circuit.gates[target].inputs
                if source not in bench_inputs:
                    raise ValueError(f"Fault {fault_id}: {source} is not an input of {target}")
                matching_pins = [i for i, name in enumerate(bench_inputs) if name == source]
                if len(matching_pins) == 1:
                    bench_pin = matching_pins[0]
                    atpg_pin_order = sorted(range(len(bench_inputs)),
                                            key=lambda index: circuit.levels[bench_inputs[index]])
                    pin = atpg_pin_order.index(bench_pin)
                else:
                    # PODEM's fault generator assigns every tied-input record
                    # the last matching index; @dup IDs can therefore name the
                    # same actual location and intentionally share a feature.
                    pin = selector
                    if pin not in (0, 1):
                        raise ValueError(f"Fault {fault_id}: invalid tied-input selector {selector}")
                    bench_pin = pin
                anchor = graph["anchors"][target]
            if alias and selector != 0:
                raise ValueError(f"Fault {fault_id}: invalid boundary selector {selector}")
            signal = source
            connected_anchor = graph["anchors"][source]
        location = (target, io, pin, sa)
        location_duplicate_index = location_counts.get(location, 0)
        location_counts[location] = location_duplicate_index + 1
        position = 0 if io else pin + 1
        if position not in (0, 1, 2):
            raise ValueError(f"Fault {fault_id}: input does not fit the two-input encoding")
        records.append({"row": len(records), "fault_id": fault_id,
                        "original_gate": identity[1],
                        "original_pin_index": -1 if io else int(identity[3]),
                        "mapped_gate": target, "source_signal": signal,
                        "io": "GO" if io else "GI", "pin_index": pin,
                        "bench_pin_index": bench_pin, "legacy_map_selector": selector,
                        "logical_xor_input": logical_xor_input,
                        "logical_input_index": logical_input_index,
                        "mapped_location_duplicate_index": location_duplicate_index,
                        "sa_value": sa, "eqv_fault_num": count,
                        "anchor_node": anchor,
                        "gate_function_anchor_node": anchor,
                        "connected_function_anchor_node": connected_anchor,
                        "position_index": position})
    required = {"source_hash", "circuit_hash", "count", "uncollapsed_total"}
    if headers.keys() != required:
        raise ValueError("Missing fault map headers")
    if int(headers["count"]) != len(records):
        raise ValueError("Fault map count does not match the number of records")
    if int(headers["uncollapsed_total"]) != sum(f["eqv_fault_num"] for f in records):
        raise ValueError("Fault map equivalence counts do not sum to uncollapsed_total")
    for key in ("source_hash", "circuit_hash"):
        if not re.fullmatch(r"[0-9a-fA-F]{16}", headers[key]):
            raise ValueError(f"Invalid legacy {key}")
    faultmap_sha256 = hashlib.sha256(raw).hexdigest()
    binding = _verify_binding(path, circuit, headers, faultmap_sha256)
    metadata = {"schema": "collapsed_fault_positions_v1",
                "bench_sha256": circuit.sha256, "graph_sha256": graph["graph_sha256"],
                "faultmap_sha256": faultmap_sha256,
                "fault_count": len(records),
                "uncollapsed_total": int(headers["uncollapsed_total"]),
                "legacy_headers": headers,
                "faultmap_bench_binding": binding,
                "pin_order": "binary BENCH inputs sorted by logic level, stable for ties (PODEM)",
                "positions": ["GO", "GI0", "GI1"], "faults": records}
    return metadata
