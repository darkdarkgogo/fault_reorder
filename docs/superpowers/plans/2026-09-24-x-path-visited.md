# X-path Visited Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent `trace_unknown_path()` from repeatedly expanding reconvergent fanout subgraphs during one X-path existence query.

**Architecture:** Keep the existing one-argument method as the top-level query boundary. It creates a fresh local `unordered_set<wptr>` and delegates to an overloaded recursive helper, which inserts each wire before exploring it and prunes repeated visits without caching across PODEM calls.

**Tech Stack:** C++11/14, Python pytest source regression, setuptools native-extension build

**Spec:** `docs/superpowers/specs/2026-09-24-x-path-visited-design.md`

## Global Constraints

- Preserve X-path existence semantics and current fanout traversal order.
- Visit each wire at most once per top-level query, giving `O(V + E)` traversal.
- Never reuse visited state across PODEM assignments, iterations, or backtracks.
- Do not change wire values, objective selection, fault status, or ATPG budgets.

---

### Task 1: Deduplicate one X-path traversal

**Files:**
- Create: `PODEM/tests/test_podem_source.py`
- Modify: `PODEM/src/atpg.h:377`
- Modify: `PODEM/src/podem.cpp:540-566`

**Interfaces:**
- Consumes: existing `bool ATPG::trace_unknown_path(wptr)` callers in `test_possible()` and `find_propagate_gate()`.
- Produces: unchanged `bool trace_unknown_path(wptr)` top-level interface plus private helper `bool trace_unknown_path(wptr, unordered_set<wptr> &visited)`.

- [ ] **Step 1: Write the failing source regression**

Create `PODEM/tests/test_podem_source.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_trace_unknown_path_deduplicates_each_top_level_query():
    source = (ROOT / "PODEM" / "src" / "podem.cpp").read_text(
        encoding="utf-8"
    )
    header = (ROOT / "PODEM" / "src" / "atpg.h").read_text(
        encoding="utf-8"
    )

    assert "bool trace_unknown_path(wptr, unordered_set<wptr> &);" in header
    assert "unordered_set<wptr> visited;" in source
    assert "return trace_unknown_path(w, visited);" in source
    assert "if (!visited.insert(w).second)" in source
    assert "trace_unknown_path(output, visited)" in source
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
python -m pytest PODEM/tests/test_podem_source.py -q
```

Expected: FAIL because the header overload and per-call visited traversal do not exist yet.

- [ ] **Step 3: Add the recursive helper declaration**

In `PODEM/src/atpg.h`, keep the current declaration and add the overload immediately after it:

```cpp
bool trace_unknown_path(wptr);
bool trace_unknown_path(wptr, unordered_set<wptr> &);
```

- [ ] **Step 4: Implement per-query visited traversal**

Replace the current recursive implementation in `PODEM/src/podem.cpp` with:

```cpp
bool ATPG::trace_unknown_path(const wptr w)
{
    unordered_set<wptr> visited;
    return trace_unknown_path(w, visited);
}

bool ATPG::trace_unknown_path(const wptr w, unordered_set<wptr> &visited)
{
    if (!visited.insert(w).second)
        return false;

    if (w->is_output())
        return true;

    for (int i = 0, nout = w->onode.size(); i < nout; i++)
    {
        wptr output = w->onode[i]->owire.front();
        if (output->value == U && trace_unknown_path(output, visited))
            return true;
    }
    return false;
}
```

- [ ] **Step 5: Run the focused test and source checks**

Run:

```powershell
python -m pytest PODEM/tests/test_podem_source.py -q
git diff --check
```

Expected: the focused test passes and `git diff --check` emits no errors.

- [ ] **Step 6: Rebuild the native extension**

Run:

```powershell
python PODEM/setup.py build_ext --inplace
```

Expected: `cpp_podem` compiles and links successfully with the new overload.

- [ ] **Step 7: Run ATPG regression tests**

Run:

```powershell
python -m pytest PODEM/tests -q
python -m pytest tests/test_diagnose_b17_atpg.py -q
```

Expected: all tests pass, demonstrating unchanged public behavior and diagnostics.

- [ ] **Step 8: Commit the implementation**

```powershell
git add -- PODEM/src/atpg.h PODEM/src/podem.cpp PODEM/tests/test_podem_source.py
git commit -m "perf: deduplicate reconvergent X-path traversal"
```
