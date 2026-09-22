"""Validated Python boundary for the fixed compressed PODEM protocol."""

import importlib
from pathlib import Path
import sys


PROTOCOL_CONFIG = {
    "primary_backtrack_limit": 100, "primary_seed": 14,
    "attempts_per_primary_fault": 1, "dtc_enabled": True,
    "dtc_secondary_backtrack_limit": 50, "stc_enabled": True,
    "stc_reverse_order_enabled": True, "stc_shuffle_seed": 7,
    "stc_no_improvement_limit": 5, "scoap_enabled": False,
    "dtc_bfs_small_input_threshold": 32,
    "dtc_bfs_small_select_fault_try": 15,
    "dtc_bfs_default_select_fault_try": 100,
    "dtc_rollback_algorithm": "accepted_pi_cube_resim_v1",
}
RESULT_FIELDS = (
    "pattern_count", "current_pattern_count", "patterns_before_stc",
    "patterns_after_stc", "detected_collapsed_faults",
    "detected_equivalent_faults", "uncollapsed_faults", "aborted_faults",
    "redundant_faults", "redundant_equivalent_faults", "podem_calls",
    "primary_podem_calls", "dtc_secondary_calls", "primary_backtracks",
    "dtc_backtracks", "total_backtracks", "stc_removed_patterns",
    "stc_shuffle_attempts", "stc_coverage_preserved", "finalized",
)
STEP_FIELDS = (
    "selected_fault_id", "target_status", "generated_pattern",
    "generated_test_vector", "dtc_attempted_fault_ids",
    "dtc_embedded_fault_ids", "current_dtc_secondary_calls",
    "current_primary_backtracks", "current_dtc_backtracks",
    "newly_detected_fault_ids", "remaining_fault_ids",
    "current_podem_calls", "current_total_backtracks",
)
COVERAGE_FIELDS = (
    "detected_collapsed_faults", "detected_equivalent_faults",
    "uncollapsed_faults", "aborted_faults", "redundant_faults",
    "redundant_equivalent_faults",
)


def load_cpp_podem(module_dir=None):
    if module_dir is not None:
        resolved = str(Path(module_dir).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    try:
        module = importlib.import_module("cpp_podem")
    except ImportError as exc:
        raise RuntimeError("cpp_podem is unavailable; build PODEM/setup.py for this Python") from exc
    if not all(hasattr(module, name) for name in (
        "catalog_stuck_at", "run_stuck_at_ordered", "StuckAtSession"
    )):
        raise RuntimeError("cpp_podem does not provide the compressed ATPG API")
    return module


def _number(raw, field):
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("PODEM returned invalid {}".format(field))
    return value


def _validate_result(raw, catalog_size, require_finalized):
    missing = [field for field in RESULT_FIELDS if field not in raw]
    if missing:
        raise RuntimeError("PODEM result is missing fields: {}".format(missing))
    result = {}
    for field in RESULT_FIELDS:
        if field in ("finalized", "stc_coverage_preserved"):
            if not isinstance(raw[field], bool):
                raise RuntimeError("PODEM returned invalid {}".format(field))
            result[field] = raw[field]
        else:
            result[field] = _number(raw, field)
    if result["finalized"] is not require_finalized:
        raise RuntimeError("PODEM returned invalid finalized state")
    if result["podem_calls"] != result["primary_podem_calls"]:
        raise RuntimeError("PODEM podem_calls disagree with primary_podem_calls")
    if result["total_backtracks"] != (result["primary_backtracks"]
                                        + result["dtc_backtracks"]):
        raise RuntimeError("PODEM total_backtracks disagree with component counts")
    if result["detected_collapsed_faults"] > catalog_size:
        raise RuntimeError("PODEM collapsed detection count exceeds the catalog")
    if result["redundant_faults"] > catalog_size:
        raise RuntimeError("PODEM collapsed redundant count exceeds the catalog")
    covered = (result["detected_equivalent_faults"]
               + result["redundant_equivalent_faults"])
    if covered > result["uncollapsed_faults"]:
        raise RuntimeError("PODEM covered count exceeds the fault total")
    if require_finalized:
        if result["pattern_count"] != result["patterns_after_stc"]:
            raise RuntimeError("PODEM pattern_count disagrees with patterns_after_stc")
        if result["patterns_after_stc"] > result["patterns_before_stc"]:
            raise RuntimeError("PODEM STC increased the pattern count")
        if result["patterns_before_stc"] != result["current_pattern_count"]:
            raise RuntimeError("PODEM STC input disagrees with raw pattern count")
        if result["stc_removed_patterns"] != (result["patterns_before_stc"]
                                               - result["patterns_after_stc"]):
            raise RuntimeError("PODEM STC removed-pattern count is inconsistent")
        if not result["stc_coverage_preserved"]:
            raise RuntimeError("PODEM STC coverage was not preserved")
    elif result["pattern_count"] != result["current_pattern_count"]:
        raise RuntimeError("PODEM unfinalized pattern count is inconsistent")
    result["covered_equivalent_faults"] = covered
    result["fault_coverage"] = (covered / result["uncollapsed_faults"]
                                if result["uncollapsed_faults"] else 0.0)
    return result


def _ids(raw, field, catalog):
    if field not in raw:
        raise RuntimeError("PODEM step is missing {}".format(field))
    value = raw[field]
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(identifier, str) for identifier in value
    ) or len(value) != len(set(value)) or not set(value) <= catalog:
        raise RuntimeError("PODEM returned invalid {}".format(field))
    return tuple(value)


class PodemSession:
    """One validated, stateful interaction with a native ATPG session."""

    def __init__(self, native):
        self._native = native
        self.config = dict(native.config())
        if self.config != PROTOCOL_CONFIG or any(
            type(self.config[field]) is not type(expected)
            for field, expected in PROTOCOL_CONFIG.items()
        ):
            raise RuntimeError("PODEM session protocol config differs from fixed production config")
        self.catalog = dict(native.catalog())
        try:
            self.initial_fault_ids = tuple(item["fault_id"] for item in self.catalog["faults"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("PODEM catalog has invalid fault IDs") from exc
        if any(not isinstance(identifier, str) for identifier in self.initial_fault_ids):
            raise RuntimeError("PODEM catalog has invalid fault IDs")
        if len(self.initial_fault_ids) != len(set(self.initial_fault_ids)):
            raise RuntimeError("PODEM catalog has duplicate fault IDs")
        self._catalog_set = set(self.initial_fault_ids)
        self.remaining_fault_ids = _ids(
            {"remaining_fault_ids": native.remaining_fault_ids()},
            "remaining_fault_ids", self._catalog_set)
        if set(self.remaining_fault_ids) != self._catalog_set:
            raise RuntimeError("PODEM initial remaining IDs differ from catalog")
        self._pattern_count = 0
        self._calls = 0
        self._dtc_calls = 0
        self._primary_backtracks = 0
        self._dtc_backtracks = 0
        if "uncollapsed_total" not in self.catalog:
            raise RuntimeError("PODEM catalog is missing uncollapsed_total")
        self._coverage = dict.fromkeys(COVERAGE_FIELDS, 0)
        self._coverage["uncollapsed_faults"] = _number(self.catalog, "uncollapsed_total")
        self._detected_ids = set()
        self._final = None
        self._pending = None

    def step(self, fault_id, rank_dtc_candidates=None):
        if self._final is not None or fault_id not in self.remaining_fault_ids:
            raise ValueError("selected fault ID is not remaining")
        if rank_dtc_candidates is not None and not callable(rank_dtc_candidates):
            raise TypeError("rank_dtc_candidates must be callable or None")
        before = set(self.remaining_fault_ids)
        if self._pending is None:
            state = dict(self._native.begin_step(fault_id))
            batches = []
            attempted_so_far = []
        else:
            if self._pending["fault_id"] != fault_id:
                raise RuntimeError("PODEM session is awaiting a DTC ranking for another Primary")
            state = self._pending["state"]
            batches = self._pending["batches"]
            attempted_so_far = self._pending["attempted"]

        while state.get("phase") == "dtc":
            if state.get("selected_fault_id") != fault_id:
                raise RuntimeError("PODEM DTC phase changed the selected fault ID")
            candidates = _ids(
                state, "dtc_candidate_fault_ids", self._catalog_set)
            if fault_id in candidates or not set(candidates) <= before - {fault_id}:
                raise RuntimeError("PODEM returned invalid BFS DTC candidates")
            batch_index = _number(state, "dtc_batch_index")
            if batch_index != len(batches):
                raise RuntimeError("PODEM returned a non-sequential DTC batch index")
            select_fault_try = _number(state, "select_fault_try")
            visited_wire_count = _number(state, "visited_wire_count")
            if select_fault_try not in (
                    self.config["dtc_bfs_small_select_fault_try"],
                    self.config["dtc_bfs_default_select_fault_try"]
            ) or visited_wire_count > select_fault_try:
                raise RuntimeError("PODEM returned invalid DTC BFS visit accounting")
            unknown_po_id = state.get("unknown_po_id")
            if not isinstance(unknown_po_id, str) or not unknown_po_id:
                raise RuntimeError("PODEM returned invalid unknown PO ID")
            metadata = {
                "unknown_po_id": unknown_po_id,
                "dtc_batch_index": batch_index,
                "select_fault_try": select_fault_try,
                "visited_wire_count": visited_wire_count,
            }
            requested_value = (candidates if rank_dtc_candidates is None
                               else rank_dtc_candidates(candidates, dict(metadata)))
            try:
                requested = tuple(requested_value)
            except TypeError as exc:
                self._pending = {
                    "fault_id": fault_id, "state": state, "batches": batches,
                    "attempted": attempted_so_far,
                }
                raise ValueError("DTC ranker must return an iterable permutation") from exc
            if (len(requested) != len(candidates)
                    or any(not isinstance(identifier, str) for identifier in requested)
                    or len(set(requested)) != len(requested)
                    or set(requested) != set(candidates)):
                self._pending = {
                    "fault_id": fault_id, "state": state, "batches": batches,
                    "attempted": attempted_so_far,
                }
                raise ValueError("DTC ranker must return exactly the current BFS candidate permutation")
            response = dict(self._native.rank_dtc_candidates(list(requested)))
            executed = _ids(
                response, "last_dtc_attempted_fault_ids", self._catalog_set)
            embedded_batch = _ids(
                response, "last_dtc_embedded_fault_ids", self._catalog_set)
            if executed != requested[:len(executed)]:
                raise RuntimeError("PODEM DTC attempted IDs are not the requested batch prefix")
            if set(executed) & set(attempted_so_far):
                raise RuntimeError("PODEM repeated a DTC secondary across batches")
            if not set(embedded_batch) <= set(executed):
                raise RuntimeError("PODEM embedded a DTC fault that was not attempted")
            embedded_positions = [executed.index(identifier)
                                  for identifier in embedded_batch]
            if any(left >= right for left, right in zip(
                    embedded_positions, embedded_positions[1:])):
                raise RuntimeError("PODEM DTC embedded IDs are not in attempted order")
            batches.append({
                **metadata,
                "bfs_candidate_fault_ids": candidates,
                "requested_fault_ids": requested,
                "executed_prefix_fault_ids": executed,
                "embedded_fault_ids": embedded_batch,
            })
            attempted_so_far.extend(executed)
            state = response
            self._pending = {
                "fault_id": fault_id, "state": state, "batches": batches,
                "attempted": attempted_so_far,
            }

        if state.get("phase") != "complete":
            raise RuntimeError("PODEM returned an invalid stuck-at session phase")
        self._pending = None
        raw = state
        missing = [field for field in STEP_FIELDS if field not in raw]
        if missing:
            raise RuntimeError("PODEM step is missing fields: {}".format(missing))
        result = _validate_result(raw, len(self.initial_fault_ids), False)
        remaining = _ids(raw, "remaining_fault_ids", self._catalog_set)
        newly = _ids(raw, "newly_detected_fault_ids", self._catalog_set)
        attempted = _ids(raw, "dtc_attempted_fault_ids", self._catalog_set)
        embedded = _ids(raw, "dtc_embedded_fault_ids", self._catalog_set)
        if not set(remaining) < before or fault_id in remaining:
            raise RuntimeError("PODEM remaining IDs must be a strict subset")
        # A later vector can detect an earlier aborted target, which is no
        # longer selectable but still participates in fault simulation.
        if set(newly) & set(remaining):
            raise RuntimeError("PODEM newly_detected IDs conflict with remaining IDs")
        if set(newly) & self._detected_ids:
            raise RuntimeError("PODEM newly_detected IDs were already detected")
        if result["detected_collapsed_faults"] != len(self._detected_ids) + len(newly):
            raise RuntimeError("PODEM detected count disagrees with newly_detected IDs")
        for field, previous in self._coverage.items():
            if result[field] < previous or (
                field == "uncollapsed_faults" and result[field] != previous
            ):
                raise RuntimeError("PODEM coverage count changed incorrectly: {}".format(field))
        if not set(attempted) <= before - {fault_id}:
            raise RuntimeError("PODEM dtc_attempted IDs were not secondary candidates")
        if attempted != tuple(attempted_so_far):
            raise RuntimeError("PODEM aggregate DTC attempts disagree with batch prefixes")
        if not set(embedded) <= set(attempted):
            raise RuntimeError("PODEM dtc_embedded IDs were not attempted")
        embedded_positions = [attempted.index(identifier) for identifier in embedded]
        if any(left >= right for left, right in zip(
                embedded_positions, embedded_positions[1:])):
            raise RuntimeError("PODEM dtc_embedded IDs are not in attempted order")
        if raw["selected_fault_id"] != fault_id:
            raise RuntimeError("PODEM selected fault ID changed")
        if not isinstance(raw["generated_pattern"], bool):
            raise RuntimeError("PODEM generated_pattern is not boolean")
        if not isinstance(raw["generated_test_vector"], str):
            raise RuntimeError("PODEM generated_test_vector is invalid")
        statuses = {"detected": True, "redundant": False, "aborted": False}
        if (not isinstance(raw["target_status"], str)
                or raw["target_status"] not in statuses
                or raw["generated_pattern"] is not statuses[raw["target_status"]]):
            raise RuntimeError("PODEM target status and generated pattern disagree")
        if not raw["generated_pattern"] and newly:
            raise RuntimeError("PODEM newly_detected IDs require a generated pattern")
        # test_tried removes the primary regardless of its simulation outcome.
        # Every other removal must be explained by this step's fault simulation.
        removed = before - set(remaining)
        if removed != {fault_id} | (before & set(newly)):
            raise RuntimeError("PODEM removed IDs disagree with selected and newly_detected IDs")
        if result["current_pattern_count"] != self._pattern_count + int(raw["generated_pattern"]):
            raise RuntimeError("PODEM raw pattern count changed incorrectly")
        if result["primary_podem_calls"] != self._calls + 1 or _number(raw, "current_podem_calls") != result["podem_calls"]:
            raise RuntimeError("PODEM primary calls changed incorrectly")
        if (result["dtc_secondary_calls"] != self._dtc_calls + len(attempted)
                or _number(raw, "current_dtc_secondary_calls") != result["dtc_secondary_calls"]):
            raise RuntimeError("PODEM DTC calls changed incorrectly")
        if result["primary_backtracks"] < self._primary_backtracks or result["dtc_backtracks"] < self._dtc_backtracks:
            raise RuntimeError("PODEM backtracks went backwards")
        for step_field, summary_field in (
            ("current_primary_backtracks", "primary_backtracks"),
            ("current_dtc_backtracks", "dtc_backtracks"),
            ("current_total_backtracks", "total_backtracks"),
        ):
            if _number(raw, step_field) != result[summary_field]:
                raise RuntimeError("PODEM current backtracks disagree with cumulative result")
        self.remaining_fault_ids = remaining
        self._pattern_count = result["current_pattern_count"]
        self._calls = result["primary_podem_calls"]
        self._dtc_calls = result["dtc_secondary_calls"]
        self._primary_backtracks = result["primary_backtracks"]
        self._dtc_backtracks = result["dtc_backtracks"]
        self._coverage = {field: result[field] for field in COVERAGE_FIELDS}
        self._detected_ids.update(newly)
        return {**result, **{field: raw[field] for field in STEP_FIELDS},
                "remaining_fault_ids": remaining, "newly_detected_fault_ids": newly,
                "dtc_attempted_fault_ids": attempted, "dtc_embedded_fault_ids": embedded,
                "dtc_batches": tuple(batches)}

    def finish(self):
        if self._final is not None:
            return dict(self._final)
        if self.remaining_fault_ids:
            raise RuntimeError("PODEM cannot finish with remaining faults")
        result = _validate_result(dict(self._native.result()),
                                  len(self.initial_fault_ids), True)
        if any(
            result[field] != expected for field, expected in self._coverage.items()
        ):
            raise RuntimeError("PODEM final coverage counts changed during STC")
        if result["current_pattern_count"] != self._pattern_count:
            raise RuntimeError("PODEM final raw pattern count changed")
        for field, expected in (("primary_podem_calls", self._calls),
                                ("dtc_secondary_calls", self._dtc_calls),
                                ("primary_backtracks", self._primary_backtracks),
                                ("dtc_backtracks", self._dtc_backtracks)):
            if result[field] != expected:
                raise RuntimeError("PODEM final {} changed".format(field))
        self._final = result
        return dict(result)


class PodemEnvironment:
    def __init__(self, module_dir=None, backtrack_limit=100, seed=14):
        if (type(backtrack_limit) is not int or type(seed) is not int
                or backtrack_limit != 100 or seed != 14):
            raise ValueError("PODEM production protocol requires backtrack_limit=100 and seed=14")
        self.module = load_cpp_podem(module_dir)
        self.backtrack_limit = 100
        self.seed = 14

    def catalog(self, bench_path, faultmap_path):
        self._check_files(bench_path, faultmap_path)
        return dict(self.module.catalog_stuck_at(
            str(bench_path), self._faultmap_argument(faultmap_path)))

    def start_session(self, bench_path, faultmap_path):
        self._check_files(bench_path, faultmap_path)
        native = self.module.StuckAtSession(
            str(bench_path), self._faultmap_argument(faultmap_path),
            100, 14, True, True)
        return PodemSession(native)

    def run(self, bench_path, faultmap_path, ordered_fault_ids):
        self._check_files(bench_path, faultmap_path)
        ids = tuple(ordered_fault_ids)
        raw = dict(self.module.run_stuck_at_ordered(
            str(bench_path), self._faultmap_argument(faultmap_path),
            list(ids), 100, 14, True, True))
        return _validate_result(raw, len(ids), True)

    @staticmethod
    def _check_files(bench_path, faultmap_path):
        if not Path(bench_path).is_file():
            raise ValueError("missing PODEM input: {}".format(bench_path))
        if faultmap_path is not None and not Path(faultmap_path).is_file():
            raise ValueError("missing PODEM input: {}".format(faultmap_path))

    @staticmethod
    def _faultmap_argument(faultmap_path):
        return "" if faultmap_path is None else str(faultmap_path)
