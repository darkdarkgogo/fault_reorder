import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier


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

    def make_three_input_and_fixture(self):
        _, binary, fault_map, _, _ = self.convert(
            "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(y)\ny = AND(a,b,c)\n"
        )
        return binary, fault_map

    def make_dtc_fixture(self, failure_only=False):
        # x:sa0 fixes a=1, leaving b and c unknown. z:sa0 can embed c=1;
        # y:sa0 is impossible for both b values, and z:sa1 conflicts with c=1.
        _, binary, fault_map, _, _ = self.convert(
            "INPUT(a)\nINPUT(b)\nINPUT(c)\n"
            "OUTPUT(x)\nOUTPUT(y)\nOUTPUT(z)\n"
            "x = BUF(a)\nn = NOT(b)\ny = AND(b,n)\nz = BUF(c)\n"
        )
        ids = ["x:GO:sa0", "y:GO:sa0"] if failure_only else [
            "x:GO:sa0", "z:GO:sa0", "y:GO:sa0", "z:GO:sa1",
        ]
        self.keep_dtc_faults(fault_map, ids)
        return binary, fault_map

    def keep_dtc_faults(self, fault_map, ids):
        lines = fault_map.read_text(encoding="ascii").splitlines()
        records = {line.split()[1]: line for line in lines if line.startswith("fault ")}
        selected = [records[fault_id] for fault_id in ids]
        fault_map.write_text("\n".join([
            *lines[:3], f"count {len(selected)}",
            f"uncollapsed_total {sum(int(line.split()[7]) for line in selected)}",
            *selected, "end", "",
        ]), encoding="ascii")

    def test_stuck_at_dtc_reports_secondary_attempts_without_primary_selection(self):
        binary, fault_map = self.make_dtc_fixture()
        session = cpp_podem.StuckAtSession(str(binary), str(fault_map))
        step = session.step("x:GO:sa0")
        self.assertEqual(step["target_status"], "detected")
        self.assertEqual(step["dtc_attempted_fault_ids"], [
            "z:GO:sa0", "y:GO:sa0", "z:GO:sa1",
        ])
        self.assertEqual(step["dtc_embedded_fault_ids"], ["z:GO:sa0"])
        self.assertEqual(step["current_dtc_secondary_calls"], 3)
        self.assertGreater(step["current_dtc_backtracks"], 0)
        self.assertLessEqual(step["current_dtc_backtracks"], 3 * 50)
        self.assertEqual(step["current_total_backtracks"],
                         step["current_primary_backtracks"] + step["current_dtc_backtracks"])
        self.assertEqual(step["generated_test_vector"][0], "1")
        self.assertEqual(step["generated_test_vector"][2], "1")
        self.assertEqual(set(step["newly_detected_fault_ids"]), {"x:GO:sa0", "z:GO:sa0"})
        self.assertEqual(step["remaining_fault_ids"], ["y:GO:sa0", "z:GO:sa1"])
        self.assertEqual(step["podem_calls"], 1)
        self.assertEqual(step["primary_podem_calls"], 1)
        self.assertEqual(step["dtc_secondary_calls"], 3)
        self.assertEqual(step["aborted_faults"], 0)
        self.assertEqual(step["redundant_faults"], 0)
        # A failed secondary is still eligible to become the next primary.
        next_step = session.step("z:GO:sa1")
        self.assertEqual(next_step["target_status"], "detected")
        self.assertEqual(next_step["generated_test_vector"][2], "0")
        self.assertEqual(next_step["dtc_attempted_fault_ids"], ["y:GO:sa0"])
        self.assertEqual(next_step["dtc_secondary_calls"], 4)
        self.assertEqual(next_step["current_dtc_secondary_calls"], 4)
        self.assertGreater(next_step["current_dtc_backtracks"], step["current_dtc_backtracks"])
        self.assertEqual(next_step["current_total_backtracks"],
                         next_step["current_primary_backtracks"] + next_step["current_dtc_backtracks"])
        self.assertEqual(next_step["total_backtracks"],
                         next_step["primary_backtracks"] + next_step["dtc_backtracks"])

    def test_failed_stuck_at_dtc_attempt_restores_primary_cube(self):
        binary, fault_map = self.make_dtc_fixture(failure_only=True)
        for seed in (0, 1, 14, 99):
            with self.subTest(seed=seed):
                enabled = cpp_podem.StuckAtSession(
                    str(binary), str(fault_map), seed=seed, dtc_enabled=True)
                disabled = cpp_podem.StuckAtSession(
                    str(binary), str(fault_map), seed=seed, dtc_enabled=False)
                actual = enabled.step("x:GO:sa0")
                baseline = disabled.step("x:GO:sa0")
                self.assertEqual(actual["target_status"], "detected")
                self.assertEqual(actual["dtc_attempted_fault_ids"], ["y:GO:sa0"])
                self.assertEqual(actual["dtc_embedded_fault_ids"], [])
                self.assertGreater(actual["current_dtc_backtracks"], 0)
                self.assertEqual(actual["generated_test_vector"], baseline["generated_test_vector"])
                self.assertEqual(actual["newly_detected_fault_ids"], baseline["newly_detected_fault_ids"])
                self.assertEqual(actual["remaining_fault_ids"], ["y:GO:sa0"])
                self.assertEqual(baseline["dtc_attempted_fault_ids"], [])
                self.assertEqual(baseline["dtc_secondary_calls"], 0)
                self.assertEqual(enabled.step("y:GO:sa0")["target_status"], "redundant")

    def test_limited_stuck_at_dtc_attempt_restores_cube_and_keeps_fault_selectable(self):
        # A seven-input parity requires all seven assignments. Its conjunction
        # with its own inverse is unsatisfiable and exceeds the 50-flip budget.
        lines = ["INPUT(a)", *[f"INPUT(b{i})" for i in range(7)],
                 "OUTPUT(x)", "OUTPUT(y)", "x = BUF(a)"]
        parity = "b0"
        for i in range(1, 7):
            lines.extend([
                f"n{i} = NAND({parity},b{i})",
                f"l{i} = NAND({parity},n{i})",
                f"r{i} = NAND(b{i},n{i})",
                f"p{i} = NAND(l{i},r{i})",
            ])
            parity = f"p{i}"
        lines.extend([f"inv = NOT({parity})", f"y = AND({parity},inv)", ""])
        _, binary, fault_map, _, _ = self.convert("\n".join(lines))
        self.keep_dtc_faults(fault_map, ["x:GO:sa0", "y:GO:sa0"])
        session = cpp_podem.StuckAtSession(str(binary), str(fault_map))
        actual = session.step("x:GO:sa0")
        baseline = cpp_podem.StuckAtSession(
            str(binary), str(fault_map), dtc_enabled=False).step("x:GO:sa0")
        self.assertEqual(actual["dtc_attempted_fault_ids"], ["y:GO:sa0"])
        self.assertEqual(actual["dtc_embedded_fault_ids"], [])
        self.assertEqual(actual["current_dtc_backtracks"], 50)
        self.assertEqual(actual["generated_test_vector"], baseline["generated_test_vector"])
        self.assertEqual(actual["remaining_fault_ids"], ["y:GO:sa0"])
        self.assertEqual(actual["aborted_faults"], 0)
        self.assertEqual(actual["redundant_faults"], 0)
        self.assertIn(session.step("y:GO:sa0")["target_status"], ("redundant", "aborted"))

    def test_stuck_at_dtc_preserves_expanded_xor_input_faults(self):
        _, binary, fault_map, _, catalog = self.convert(expanded_xor(True, True))
        for fault in catalog["faults"]:
            fault_id = fault["fault_id"]
            with self.subTest(primary=fault_id):
                session = cpp_podem.StuckAtSession(str(binary), str(fault_map))
                step = session.step(fault_id)
                self.assertEqual(step["target_status"], "detected")
                self.assertIn(fault_id, step["newly_detected_fault_ids"])
                self.assertLessEqual(set(step["dtc_embedded_fault_ids"]),
                                     set(step["newly_detected_fault_ids"]))

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
            "primary_podem_calls", "dtc_secondary_calls",
            "primary_backtracks", "dtc_backtracks",
        }

        native = cpp_podem.run_stuck_at_ordered(
            str(binary), str(fault_map), fault_ids
        )
        explicit_native = cpp_podem.run_stuck_at_ordered(
            str(binary), str(fault_map), fault_ids, 200, 14
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
        _, binary, fault_map, _, catalog = self.convert(
            "INPUT(a)\n"
            "INPUT(b)\n"
            "OUTPUT(y)\n"
            "n1 = AND(a,b)\n"
            "y = OR(a,n1)\n"
        )
        weights = {
            str(fault["fault_id"]): int(fault["eqv_fault_num"])
            for fault in catalog["faults"]
        }
        session = cpp_podem.StuckAtSession(
            str(binary), str(fault_map), 5000, 14
        )
        redundant_ids = []
        while session.remaining_fault_ids():
            step = session.step(session.remaining_fault_ids()[0])
            if step["target_status"] == "redundant":
                redundant_ids.append(step["selected_fault_id"])
        metrics = session.result()

        self.assertTrue(redundant_ids)
        self.assertTrue(any(weights[fault_id] > 1 for fault_id in redundant_ids))
        self.assertEqual(metrics["redundant_faults"], len(redundant_ids))
        self.assertEqual(
            metrics["redundant_equivalent_faults"],
            sum(weights[fault_id] for fault_id in redundant_ids),
        )

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
            str(binary), str(fault_map), ids, 200, 14
        )
        session = cpp_podem.StuckAtSession(
            str(binary), str(fault_map), 200, 14
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
            str(binary), str(fault_map), 200, 14
        )
        session.step(fault_id)
        with self.assertRaisesRegex(RuntimeError, "not selectable"):
            session.step(fault_id)
        with self.assertRaisesRegex(RuntimeError, "Unknown fault ID"):
            session.step("missing:GO:sa0")

    def test_stuck_at_session_reports_fixed_protocol(self):
        _, binary, fault_map, _, _ = self.convert(
            "INPUT(a)\nINPUT(b)\nOUTPUT(y)\ny = AND(a,b)\n"
        )
        session = cpp_podem.StuckAtSession(str(binary), str(fault_map))
        self.assertEqual(session.config(), {
            "primary_backtrack_limit": 200,
            "primary_seed": 14,
            "attempts_per_primary_fault": 1,
            "dtc_enabled": True,
            "dtc_secondary_backtrack_limit": 50,
            "stc_enabled": True,
            "stc_reverse_order_enabled": True,
            "stc_shuffle_seed": 7,
            "stc_no_improvement_limit": 5,
            "scoap_enabled": False,
        })
        snapshot = session.config()
        snapshot["primary_seed"] = 99
        snapshot["dtc_enabled"] = False
        self.assertEqual(session.config()["primary_seed"], 14)
        self.assertTrue(session.config()["dtc_enabled"])

    def test_interleaved_sessions_have_identical_primary_traces(self):
        binary, fault_map = self.make_three_input_and_fixture()
        left = cpp_podem.StuckAtSession(str(binary), str(fault_map))
        right = cpp_podem.StuckAtSession(str(binary), str(fault_map))
        left_trace, right_trace = [], []
        while left.remaining_fault_ids():
            fault_id = left.remaining_fault_ids()[0]
            self.assertEqual(fault_id, right.remaining_fault_ids()[0])
            left_trace.append(left.step(fault_id))
            right_trace.append(right.step(fault_id))
        self.assertEqual(left_trace, right_trace)
        self.assertEqual(right.remaining_fault_ids(), [])

    def test_random_fill_is_repeatable_interleaved_and_concurrent(self):
        # Each independent output leaves other inputs unknown in the primary
        # cube, so fault dropping observes the random fill directly.
        _, binary, fault_map, _, _ = self.convert(
            "INPUT(a)\nINPUT(b)\nINPUT(c)\nINPUT(d)\n"
            "OUTPUT(w)\nOUTPUT(x)\nOUTPUT(y)\nOUTPUT(z)\n"
            "w = BUF(a)\nx = BUF(b)\ny = BUF(c)\nz = BUF(d)\n"
        )

        def new_session(seed=14):
            return cpp_podem.StuckAtSession(
                str(binary), str(fault_map), 200, seed, False, False
            )

        def collect_trace(session):
            trace = []
            while session.remaining_fault_ids():
                trace.append(session.step(session.remaining_fault_ids()[0]))
            return trace

        # Guard against a fixture which never consumes RNG state.
        first_drops = []
        for seed in range(4):
            session = new_session(seed)
            first_drops.append(tuple(session.step(
                session.remaining_fault_ids()[0]
            )["newly_detected_fault_ids"]))
        self.assertGreater(len(set(first_drops)), 1)

        expected = collect_trace(new_session())
        self.assertEqual(collect_trace(new_session()), expected)
        left, right = new_session(), new_session()
        left_trace, right_trace = [], []
        while left.remaining_fault_ids():
            fault_id = left.remaining_fault_ids()[0]
            self.assertEqual(right.remaining_fault_ids()[0], fault_id)
            left_trace.append(left.step(fault_id))
            right_trace.append(right.step(fault_id))
        self.assertEqual(left_trace, expected)
        self.assertEqual(right_trace, expected)
        self.assertEqual(right.remaining_fault_ids(), [])
        with ThreadPoolExecutor(max_workers=2) as pool:
            traces = list(pool.map(collect_trace, [new_session(), new_session()]))
        self.assertEqual(traces, [expected, expected])

    def test_incremental_status_updates_are_exact(self):
        binary, fault_map = self.make_three_input_and_fixture()
        detected = cpp_podem.StuckAtSession(
            str(binary), str(fault_map), 200, 14, False, False
        )
        before = detected.result()
        step = detected.step("dummy_gate1:GO:sa1")
        self.assertEqual(step["target_status"], "detected")
        self.assertTrue(step["generated_pattern"])
        self.assertEqual(step["pattern_count"], before["pattern_count"] + 1)

        _, redundant_binary, redundant_map, _, _ = self.convert(
            "INPUT(a)\n"
            "INPUT(b)\n"
            "OUTPUT(y)\n"
            "n1 = AND(a,b)\n"
            "y = OR(a,n1)\n"
        )
        redundant = cpp_podem.StuckAtSession(
            str(redundant_binary), str(redundant_map), 200, 14, False, False
        )
        for fault_id in (
            "dummy_gate1:GO:sa0",
            "dummy_gate1:GO:sa1",
            "n1:GI0:sa1",
        ):
            redundant.step(fault_id)
        before = redundant.result()
        step = redundant.step("dummy_gate2:GO:sa1")
        self.assertEqual(step["target_status"], "redundant")
        self.assertFalse(step["generated_pattern"])
        self.assertEqual(step["pattern_count"], before["pattern_count"])

        aborted = cpp_podem.StuckAtSession(
            str(binary), str(fault_map), 0, 14, False, False
        )
        before = aborted.result()
        step = aborted.step("dummy_gate1:GO:sa1")
        self.assertEqual(step["target_status"], "aborted")
        self.assertFalse(step["generated_pattern"])
        self.assertEqual(step["pattern_count"], before["pattern_count"])

        for session, selected_id in (
            (detected, "dummy_gate1:GO:sa1"),
            (redundant, "dummy_gate2:GO:sa1"),
            (aborted, "dummy_gate1:GO:sa1"),
        ):
            before_errors = session.result()
            remaining = session.remaining_fault_ids()
            self.assertNotIn(selected_id, remaining)
            with self.assertRaisesRegex(RuntimeError, "not selectable"):
                session.step(selected_id)
            with self.assertRaisesRegex(RuntimeError, "Unknown fault ID"):
                session.step("missing:GO:sa0")
            self.assertEqual(session.result(), before_errors)
            self.assertEqual(session.remaining_fault_ids(), remaining)

    def test_incremental_session_matches_independent_de86cdd_goldens(self):
        # Captured from run_stuck_at_ordered in an isolated de86cdd worktree
        # before this refactor, seed=14, compression disabled. Do not regenerate
        # these constants through the current incremental implementation.
        cases = (
            (
                "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(y)\n"
                "y = AND(a,b,c)\n",
                200,
                [
                    "dummy_gate1:GO:sa1",
                    "dummy_gate2:GO:sa1",
                    "dummy_gate3:GO:sa1",
                    "y:GO:sa0",
                    "y:GO:sa1",
                ],
                {
                    "pattern_count": 4,
                    "detected_collapsed_faults": 5,
                    "detected_equivalent_faults": 8,
                    "uncollapsed_faults": 8,
                    "aborted_faults": 0,
                    "redundant_faults": 0,
                    "redundant_equivalent_faults": 0,
                    "podem_calls": 4,
                    "total_backtracks": 0,
                },
            ),
            (
                "INPUT(a)\nINPUT(b)\nOUTPUT(y)\n"
                "n1 = AND(a,b)\ny = OR(a,n1)\n",
                200,
                [
                    "dummy_gate1:GO:sa0",
                    "dummy_gate1:GO:sa1",
                    "y:GI0:sa0",
                    "n1:GI0:sa1",
                    "dummy_gate2:GO:sa1",
                    "n1:GO:sa0",
                    "y:GO:sa0",
                    "y:GO:sa1",
                ],
                {
                    "pattern_count": 3,
                    "detected_collapsed_faults": 6,
                    "detected_equivalent_faults": 8,
                    "uncollapsed_faults": 12,
                    "aborted_faults": 0,
                    "redundant_faults": 2,
                    "redundant_equivalent_faults": 4,
                    "podem_calls": 5,
                    "total_backtracks": 1,
                },
            ),
            (
                "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(y)\n"
                "y = AND(a,b,c)\n",
                0,
                [
                    "dummy_gate1:GO:sa1",
                    "dummy_gate2:GO:sa1",
                    "dummy_gate3:GO:sa1",
                    "y:GO:sa0",
                    "y:GO:sa1",
                ],
                {
                    "pattern_count": 1,
                    "detected_collapsed_faults": 1,
                    "detected_equivalent_faults": 4,
                    "uncollapsed_faults": 8,
                    "aborted_faults": 4,
                    "redundant_faults": 0,
                    "redundant_equivalent_faults": 0,
                    "podem_calls": 5,
                    "total_backtracks": 0,
                },
            ),
        )
        for text, backtrack_limit, expected_ids, expected_result in cases:
            with self.subTest(backtrack_limit=backtrack_limit, text=text):
                _, binary, fault_map, _, _ = self.convert(text)
                session = cpp_podem.StuckAtSession(
                    str(binary), str(fault_map), backtrack_limit, 14,
                    False, False,
                )
                self.assertEqual(
                    [
                        str(item["fault_id"])
                        for item in session.catalog()["faults"]
                    ],
                    expected_ids,
                )
                while session.remaining_fault_ids():
                    session.step(session.remaining_fault_ids()[0])
                result = session.result()
                self.assertEqual({key: result[key] for key in expected_result}, expected_result)
                self.assertEqual(result["primary_podem_calls"], expected_result["podem_calls"])
                self.assertEqual(result["primary_backtracks"], expected_result["total_backtracks"])
                self.assertEqual(result["dtc_secondary_calls"], 0)
                self.assertEqual(result["dtc_backtracks"], 0)

    def test_same_session_concurrent_steps_are_serialized(self):
        binary, fault_map = self.make_three_input_and_fixture()
        session = cpp_podem.StuckAtSession(str(binary), str(fault_map))
        fault_id = session.remaining_fault_ids()[0]
        barrier = Barrier(2)

        def attempt_step():
            barrier.wait(timeout=10)
            try:
                return session.step(fault_id)
            except RuntimeError as error:
                return str(error)

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: attempt_step(), range(2)))

        self.assertEqual(sum(isinstance(item, dict) for item in outcomes), 1)
        self.assertEqual(
            sum("not selectable" in item for item in outcomes if isinstance(item, str)),
            1,
        )
        self.assertEqual(session.result()["podem_calls"], 1)


if __name__ == "__main__":
    unittest.main()
