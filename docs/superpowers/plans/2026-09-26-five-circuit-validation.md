# Five-Circuit Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the stuck-at wire budget at 15 for every circuit and remove `b17_C` from all default fault-order RL validation runs.

**Architecture:** Replace the six-circuit default manifest with a five-circuit manifest and update every current runtime/default reference. Teach the manifest generator to exclude `b17_C` while retaining its dataset files, and extract single-manifest generation into a function that removes stale generated JSON files.

**Tech Stack:** Python 3.9, JSON manifests, pytest, Bash launch scripts, Markdown

**Spec:** `docs/superpowers/specs/2026-09-26-five-circuit-validation-design.md`

## Global Constraints

- Default RL validation contains exactly `b12_C`, `b15_C`, `b20_C`, `b21_C`, and `b22_C`.
- `b17_C` dataset/AIG artifacts and `scripts/diagnose_b17_atpg.py` remain available.
- DeepTPI manifests and TDF code remain unchanged.
- Both stuck-at DTC protocol wire-budget fields remain exactly 15.
- Historical files under `docs/superpowers` and `docs/experiments` are not rewritten.

---

### Task 1: Make the default RL validation set exclude b17_C

**Files:**
- Create: `configs/anchor_validation_5.json`
- Create: `tests/test_anchor_rl_manifests.py`
- Delete: `configs/anchor_validation_6.json`
- Delete: `configs/anchor_validation_single/b17_C.json`
- Modify: `scripts/generate_anchor_rl_manifests.py`
- Modify: `fault_order_rl/trainer.py:34-36`
- Modify: `tests/test_fault_order_rl.py:93-105`
- Modify: `scripts/run_anchor_linux.sh:14-18`
- Modify: `scripts/evaluate_anchor_linux.sh:7-9`
- Modify: `docs/fault-order-rl.md`
- Modify: `docs/fault-reorder-algorithm.md:119-123`

**Interfaces:**
- Consumes: the complete six-circuit artifact set in `datasets/validation` and `datasets/validation_AIG`.
- Produces: `DEFAULT_VALIDATION_MANIFEST -> configs/anchor_validation_5.json`, `generate_manifest(split, expected_count, output, excluded_stems=())`, and `write_single_manifests(validation, single_dir)`.

- [ ] **Step 1: Add failing manifest and protocol tests**

Create `tests/test_anchor_rl_manifests.py` with these assertions:

```python
import json
from pathlib import Path

from fault_order_rl.trainer import DEFAULT_VALIDATION_MANIFEST
from scripts import generate_anchor_rl_manifests as manifests


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["b12_C", "b15_C", "b20_C", "b21_C", "b22_C"]


def test_default_validation_manifest_has_five_circuits_without_b17():
    assert DEFAULT_VALIDATION_MANIFEST.name == "anchor_validation_5.json"
    payload = json.loads(DEFAULT_VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    assert [entry["name"] for entry in payload["circuits"]] == EXPECTED
    assert sorted(path.stem for path in
                  (ROOT / "configs" / "anchor_validation_single").glob("*.json")) == EXPECTED


def test_validation_generator_excludes_b17_and_removes_stale_single(tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(manifests, "ROOT", tmp_path)
    source_dir = tmp_path / "datasets" / "validation"
    aig_dir = tmp_path / "datasets" / "validation_AIG"
    config_dir = tmp_path / "configs"
    source_dir.mkdir(parents=True)
    aig_dir.mkdir(parents=True)
    config_dir.mkdir()
    for name in ["b12_C", "b15_C", "b17_C", "b20_C", "b21_C", "b22_C"]:
        (source_dir / f"{name}.bench").write_text("", encoding="utf-8")
        for suffix in (".bench", ".aigmap.json", ".fault_embeddings.npz",
                       ".faults.json"):
            (aig_dir / f"{name}{suffix}").write_text("", encoding="utf-8")
    output = config_dir / "anchor_validation_5.json"
    manifests.generate_manifest(
        "validation", 5, output, excluded_stems={"b17_C"})
    payload = json.loads(output.read_text(encoding="utf-8"))
    single_dir = config_dir / "anchor_validation_single"
    single_dir.mkdir()
    (single_dir / "b17_C.json").write_text("stale", encoding="utf-8")
    manifests.write_single_manifests(payload, single_dir)
    assert [entry["name"] for entry in payload["circuits"]] == EXPECTED
    assert sorted(path.stem for path in single_dir.glob("*.json")) == EXPECTED
```

In `tests/test_fault_order_rl.py::test_fixed_protocol_and_training_defaults`, add:

```python
assert PROTOCOL_CONFIG["dtc_bfs_small_select_fault_try"] == 15
assert PROTOCOL_CONFIG["dtc_bfs_default_select_fault_try"] == 15
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest `
  tests/test_anchor_rl_manifests.py `
  tests/test_fault_order_rl.py::test_fixed_protocol_and_training_defaults `
  -q -p no:cacheprovider --basetemp tmp/pytest-five-validation-red
```

Expected: manifest tests fail because `anchor_validation_5.json`, the generator exclusion argument, and single-manifest helper do not exist. The wire-budget assertions pass.

- [ ] **Step 3: Replace the combined and single validation manifests**

Create `configs/anchor_validation_5.json` from the existing combined manifest with the complete `b17_C` object removed. Delete `configs/anchor_validation_6.json` and `configs/anchor_validation_single/b17_C.json`; leave all five other single manifests unchanged.

- [ ] **Step 4: Make manifest regeneration preserve the five-circuit subset**

In `scripts/generate_anchor_rl_manifests.py`, define:

```python
VALIDATION_EXCLUDED_STEMS = frozenset({"b17_C"})
SPLITS = {
    "train": (1024, ROOT / "configs" / "anchor_train_1024.json", frozenset()),
    "validation": (
        5, ROOT / "configs" / "anchor_validation_5.json",
        VALIDATION_EXCLUDED_STEMS),
}
```

Change `generate_manifest` to accept `excluded_stems=()` and filter before the expected-count check:

```python
sources = sorted(
    (source for source in source_dir.glob("*.bench")
     if source.stem not in excluded_stems),
    key=lambda path: path.name,
)
```

Extract `write_single_manifests(validation, single_dir)`. It computes the expected `<name>.json` set, unlinks existing JSON files outside that set, and writes the adjusted five manifests using the existing path-prefix logic. Update `main()` to pass each split's exclusions, read `SPLITS["validation"][1]`, and call the helper.

- [ ] **Step 5: Update runtime defaults, launchers, and current docs**

Change every current reference from `anchor_validation_6.json` to `anchor_validation_5.json` in:

```text
fault_order_rl/trainer.py
scripts/run_anchor_linux.sh
scripts/evaluate_anchor_linux.sh
docs/fault-order-rl.md
docs/fault-reorder-algorithm.md
```

Do not replace historical references under `docs/superpowers` or `docs/experiments`.

- [ ] **Step 6: Run focused tests and generator validation**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest `
  tests/test_anchor_rl_manifests.py `
  tests/test_fault_order_rl.py::test_fixed_protocol_and_training_defaults `
  -q -p no:cacheprovider --basetemp tmp/pytest-five-validation-green
C:\Users\acer\.conda\envs\d2l\python.exe scripts/generate_anchor_rl_manifests.py
```

Expected: tests pass; generator reports five validation and five validation-single circuits; regenerated files keep `b17_C` excluded.

- [ ] **Step 7: Run regression and scope checks**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest `
  tests/test_fault_order_rl.py tests/test_anchor_rl_manifests.py PODEM/tests `
  -q -p no:cacheprovider --basetemp tmp/pytest-five-validation-full
rg -n "anchor_validation_6" fault_order_rl scripts configs `
  docs/fault-order-rl.md docs/fault-reorder-algorithm.md tests
rg -n '"name": "b17_C"' configs/anchor_validation_5.json `
  configs/anchor_validation_single
git diff --exit-code -- PODEM/src/tdfatpg.cpp `
  datasets/validation datasets/validation_AIG `
  configs/deeptpi_itc22_test_9.json scripts/diagnose_b17_atpg.py
git diff --check
```

Expected: all tests pass; the two searches return no matches; protected TDF/b17/DeepTPI paths have no diff; formatting check is clean.

- [ ] **Step 8: Commit the implementation**

```powershell
git add -- configs/anchor_validation_5.json `
  configs/anchor_validation_6.json `
  configs/anchor_validation_single/b17_C.json `
  scripts/generate_anchor_rl_manifests.py scripts/run_anchor_linux.sh `
  scripts/evaluate_anchor_linux.sh fault_order_rl/trainer.py `
  tests/test_anchor_rl_manifests.py tests/test_fault_order_rl.py `
  docs/fault-order-rl.md docs/fault-reorder-algorithm.md
git commit -m "perf: exclude b17 from default validation"
```

