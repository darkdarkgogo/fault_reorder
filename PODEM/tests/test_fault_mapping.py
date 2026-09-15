import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cpp_podem
from convert_binary_bench import catalog_cpp_podem, convert_binary_bench


def expanded_xor(extra_a_fanout: bool, extra_b_fanout: bool) -> str:
    extra_gates = []
    extra_outputs = []
    if extra_a_fanout:
        extra_gates.append("qa = NOT(a)")
        extra_outputs.append("OUTPUT(qa)")
    if extra_b_fanout:
        extra_gates.append("qb = NOT(b)")
        extra_outputs.append("OUTPUT(qb)")
    return "\n".join([
        "INPUT(a)",
        "INPUT(b)",
        "OUTPUT(G1)",
        *extra_outputs,
        "# G1 = XOR(a,b)",
        "W1 = NOT(a)",
        "Z1 = NOT(b)",
        "X1 = NAND(a,Z1)",
        "Y1 = NAND(b,W1)",
        "G1 = NAND(X1,Y1)",
        *extra_gates,
        "",
    ])


def fault_signature(catalog: dict) -> set[tuple[str, int]]:
    return {
        (str(fault["fault_id"]), int(fault["eqv_fault_num"]))
        for fault in catalog["faults"]
    }


class FaultMappingTests(unittest.TestCase):
    def convert(self, text: str):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "source.bench"
        binary = root / "binary.bench"
        fault_map = root / "binary.faultmap"
        source.write_text(text, encoding="ascii")
        stats = convert_binary_bench(source, binary, fault_map)
        return source, binary, fault_map, stats, catalog_cpp_podem(binary, fault_map)

    def test_multi_input_gate_uses_only_original_fault_ids(self):
        source, binary, _, stats, mapped = self.convert("\n".join([
            "INPUT(a)",
            "INPUT(b)",
            "INPUT(c)",
            "INPUT(d)",
            "OUTPUT(y)",
            "OUTPUT(q)",
            "y = AND(a,b,c,d)",
            "q = NOT(a)",
            "",
        ]))
        native = catalog_cpp_podem(source)
        binary_text = binary.read_text(encoding="utf-8")

        self.assertIn("__smartatpg_bin_", binary_text)
        self.assertEqual(fault_signature(mapped), fault_signature(native))
        self.assertEqual(mapped["uncollapsed_total"], native["uncollapsed_total"])
        self.assertEqual(stats["synthetic_gates"], 2)
        self.assertFalse(any(
            str(fault["fault_id"]).startswith("__smartatpg_bin_")
            for fault in mapped["faults"]
        ))

        mapped_input = next(
            fault for fault in mapped["faults"]
            if fault["fault_id"] == "y:GI0:sa1"
        )
        self.assertTrue(mapped_input["node_name"].startswith("__smartatpg_bin_"))
        self.assertEqual(mapped_input["input_wire_name"], "a")

    def test_output_paths_cannot_overwrite_source_or_each_other(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "source.bench"
        destination = root / "binary.bench"
        original = "INPUT(a)\nOUTPUT(y)\ny = BUF(a)\n"
        source.write_text(original, encoding="ascii")

        for fault_map in (source, destination):
            with self.subTest(fault_map=fault_map.name):
                with self.assertRaisesRegex(
                    ValueError, "must be different files"
                ):
                    convert_binary_bench(source, destination, fault_map)
                self.assertEqual(source.read_text(encoding="ascii"), original)
                self.assertFalse(destination.exists())

    def test_expanded_xor_without_other_fanout_keeps_inputs_collapsed(self):
        source, _, fault_map, stats, mapped = self.convert(
            expanded_xor(False, False)
        )
        native = Path(source.parent) / "native.bench"
        native.write_text(
            "INPUT(a)\nINPUT(b)\nOUTPUT(G1)\nG1 = XOR(a,b)\n",
            encoding="ascii",
        )

        self.assertEqual(
            fault_map.read_text(encoding="utf-8").splitlines()[0],
            "SMARTATPG_FAULT_MAP_V2",
        )
        self.assertFalse(any(
            str(fault["fault_id"]).startswith("G1:GI")
            for fault in mapped["faults"]
        ))
        self.assertEqual(stats["retained_xor_input_faults"], 0)
        self.assertEqual(fault_signature(mapped), fault_signature(catalog_cpp_podem(native)))

    def test_expanded_xor_preserves_each_fanout_branch_independently(self):
        for fanout_a, fanout_b, expected_ids in (
            (True, False, {"G1:GI0:sa0", "G1:GI0:sa1"}),
            (False, True, {"G1:GI1:sa0", "G1:GI1:sa1"}),
            (True, True, {
                "G1:GI0:sa0", "G1:GI0:sa1",
                "G1:GI1:sa0", "G1:GI1:sa1",
            }),
        ):
            with self.subTest(fanout_a=fanout_a, fanout_b=fanout_b):
                _, _, fault_map, stats, mapped = self.convert(
                    expanded_xor(fanout_a, fanout_b)
                )
                xor_input_faults = {
                    str(fault["fault_id"]): fault
                    for fault in mapped["faults"]
                    if str(fault["fault_id"]).startswith("G1:GI")
                }

                self.assertEqual(
                    fault_map.read_text(encoding="utf-8").splitlines()[0],
                    "SMARTATPG_FAULT_MAP_V3",
                )
                self.assertEqual(set(xor_input_faults), expected_ids)
                self.assertEqual(
                    stats["retained_xor_input_faults"], len(expected_ids)
                )
                self.assertFalse(any(
                    str(fault["fault_id"]).startswith(("W1:", "Z1:", "X1:", "Y1:"))
                    for fault in mapped["faults"]
                ))

    def test_expanded_xor_catalog_matches_native_xor(self):
        for fanout_a, fanout_b in ((False, False), (True, False), (True, True)):
            with self.subTest(fanout_a=fanout_a, fanout_b=fanout_b):
                source, _, _, _, mapped = self.convert(
                    expanded_xor(fanout_a, fanout_b)
                )
                native_lines = [
                    "INPUT(a)", "INPUT(b)", "OUTPUT(G1)",
                ]
                if fanout_a:
                    native_lines.extend(("OUTPUT(qa)", "qa = NOT(a)"))
                if fanout_b:
                    native_lines.extend(("OUTPUT(qb)", "qb = NOT(b)"))
                native_lines.append("G1 = XOR(a,b)")
                native = source.parent / "native.bench"
                native.write_text("\n".join(native_lines) + "\n", encoding="ascii")
                native_catalog = catalog_cpp_podem(native)

                self.assertEqual(fault_signature(mapped), fault_signature(native_catalog))
                self.assertEqual(
                    mapped["uncollapsed_total"], native_catalog["uncollapsed_total"]
                )

    def test_invalid_v3_logical_xor_markers_are_rejected(self):
        _, binary, fault_map, _, _ = self.convert(expanded_xor(True, False))
        original_lines = fault_map.read_text(encoding="utf-8").splitlines()
        logical_index = next(
            index for index, line in enumerate(original_lines)
            if line.startswith("fault G1:GI0:sa0 ")
        )
        regular_index = next(
            index for index, line in enumerate(original_lines)
            if line.startswith("fault ") and line.endswith(" 0 -1")
        )
        cases = (
            (logical_index, ("2", "0")),
            (logical_index, ("1", "2")),
            (regular_index, ("0", "0")),
        )
        for case, (line_index, marker) in enumerate(cases):
            with self.subTest(marker=marker):
                lines = list(original_lines)
                fields = lines[line_index].split()
                fields[-2:] = marker
                lines[line_index] = " ".join(fields)
                malformed = fault_map.with_name(f"malformed_{case}.faultmap")
                malformed.write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

                with self.assertRaisesRegex(
                    RuntimeError, "Invalid V3 logical XOR marker"
                ):
                    catalog_cpp_podem(binary, malformed)

    def test_ordered_atpg_returns_metrics_for_exact_permutations(self):
        _, binary, fault_map, _, catalog = self.convert("\n".join([
            "INPUT(a)",
            "INPUT(b)",
            "INPUT(c)",
            "OUTPUT(y)",
            "y = AND(a,b,c)",
            "",
        ]))
        fault_ids = [str(fault["fault_id"]) for fault in catalog["faults"]]
        expected_keys = {
            "pattern_count", "detected_collapsed_faults",
            "detected_equivalent_faults", "uncollapsed_faults",
            "aborted_faults", "redundant_faults",
            "redundant_equivalent_faults", "podem_calls",
            "total_backtracks",
        }

        native = cpp_podem.run_stuck_at_ordered(
            str(binary), str(fault_map), fault_ids
        )
        explicit_native = cpp_podem.run_stuck_at_ordered(
            str(binary), str(fault_map), fault_ids, 5000, 14
        )
        reversed_run = cpp_podem.run_stuck_at_ordered(
            str(binary), str(fault_map), list(reversed(fault_ids))
        )

        self.assertEqual(native, explicit_native)
        self.assertEqual(set(native), expected_keys)
        self.assertEqual(native["uncollapsed_faults"], catalog["uncollapsed_total"])
        self.assertGreater(native["pattern_count"], 0)
        self.assertGreater(native["detected_equivalent_faults"], 0)
        self.assertGreaterEqual(
            native["redundant_equivalent_faults"], native["redundant_faults"]
        )
        self.assertLessEqual(
            native["detected_equivalent_faults"]
            + native["redundant_equivalent_faults"],
            native["uncollapsed_faults"],
        )
        self.assertEqual(set(reversed_run), expected_keys)

    def test_redundant_equivalent_count_uses_uncollapsed_weights(self):
        binary = ROOT / "sample_circuits" / "c1908_binary.bench"
        fault_map = binary.with_suffix(".faultmap")
        catalog = cpp_podem.catalog_stuck_at(str(binary), str(fault_map))
        fault_ids = [str(fault["fault_id"]) for fault in catalog["faults"]]
        metrics = cpp_podem.run_stuck_at_ordered(
            str(binary), str(fault_map), fault_ids, 5000, 14
        )

        self.assertEqual(metrics["redundant_faults"], 36)
        self.assertEqual(metrics["redundant_equivalent_faults"], 88)

    def test_ordered_atpg_rejects_invalid_permutations(self):
        _, binary, fault_map, _, catalog = self.convert(
            "INPUT(a)\nOUTPUT(y)\ny = BUF(a)\n"
        )
        fault_ids = [str(fault["fault_id"]) for fault in catalog["faults"]]
        cases = (
            (fault_ids[:-1], "exactly"),
            (fault_ids[:-1] + [fault_ids[0]], "Duplicate"),
            (fault_ids[:-1] + ["__smartatpg_bin_fake:GO:sa0"], "Unknown"),
        )
        for ordered, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    cpp_podem.run_stuck_at_ordered(
                        str(binary), str(fault_map), ordered
                    )

    def test_incremental_session_matches_native_complete_run(self):
        _, binary, fault_map, _, catalog = self.convert(
            "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(y)\ny = AND(a,b,c)\n"
        )
        ids = [str(item["fault_id"]) for item in catalog["faults"]]
        expected = cpp_podem.run_stuck_at_ordered(
            str(binary), str(fault_map), ids, 5000, 14
        )
        session = cpp_podem.StuckAtSession(
            str(binary), str(fault_map), 5000, 14
        )
        self.assertEqual(
            [str(item["fault_id"]) for item in session.catalog()["faults"]],
            ids,
        )
        seen = []
        while session.remaining_fault_ids():
            selected = session.remaining_fault_ids()[0]
            before = set(session.remaining_fault_ids())
            step = session.step(selected)
            seen.append(selected)
            self.assertEqual(step["selected_fault_id"], selected)
            self.assertLess(set(step["remaining_fault_ids"]), before)
        self.assertTrue(seen)
        self.assertEqual(session.result(), expected)

    def test_incremental_session_rejects_non_selectable_fault(self):
        _, binary, fault_map, _, catalog = self.convert(
            "INPUT(a)\nOUTPUT(y)\ny = BUF(a)\n"
        )
        fault_id = str(catalog["faults"][0]["fault_id"])
        session = cpp_podem.StuckAtSession(
            str(binary), str(fault_map), 5000, 14
        )
        session.step(fault_id)
        with self.assertRaisesRegex(RuntimeError, "not selectable"):
            session.step(fault_id)
        with self.assertRaisesRegex(RuntimeError, "Unknown fault ID"):
            session.step("missing:GO:sa0")


if __name__ == "__main__":
    unittest.main()
