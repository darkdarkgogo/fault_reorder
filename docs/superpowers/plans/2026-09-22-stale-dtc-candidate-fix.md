# Stale DTC Candidate Bugfix Implementation Plan

> **For agentic workers:** Implement inline in the current session. Do not dispatch subagents; the repository task only needs one focused native change and its regression test.

**Goal:** 防止已被前一条完整测试向量检测并 drop 的 fault 通过静态 `wire->udflist` 再次进入后续 BFS DTC batch。

**Architecture:** 保留 `udflist` 作为稳定结构索引，不在每个 PO 重建列表。候选发现时增加运行状态过滤，排除 `detect == TRUE`；Python 的 remaining-set 校验继续作为协议边界保护。

**Tech Stack:** C++14、pybind11、Python 3、pytest、setuptools。

**Spec:** `docs/superpowers/specs/2026-09-22-bfs-filtered-ranked-dtc-design.md`

## Global Constraints

- 不改变 BFS 顺序、15/100 wire 预算、PO-stop、RL 排序或 PI-cube 回滚。
- 不在每个 PO 扫描或重建完整 fault list。
- 回归必须证明跨 Primary 不会返回已经检测并 drop 的 fault。
- 只运行聚焦测试和单个 `b12_C` 集成，不运行完整 validation 套件。

---

### Task 1: 排除 stale detected fault

**Files:**
- Modify: `PODEM/src/saf_compaction.cpp:148-156`
- Test: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Consumes: `FAULT::detect`、`TRUE`、静态 `WIRE::udflist`。
- Produces: `find_next_stuck_at_dtc_batch()` 只返回当前 remaining/selectable faults。

- [x] **Step 1: 写失败回归测试**

  构造一个小电路，执行第一条启发式 DTC step；记录其 `newly_detected_fault_ids`。继续执行下一 Primary 并遍历其所有 DTC phase，断言每个
  `dtc_candidate_fault_ids` 都与此前 detected 集合不相交。

- [x] **Step 2: 确认测试在旧实现失败**

  Run: `python -m pytest PODEM/tests/test_fault_mapping.py -k stale_detected -q`

  Expected: FAIL，候选 batch 含前一条向量已经 drop 的 fault。

- [x] **Step 3: 实施最小修复**

  在 `find_next_stuck_at_dtc_batch()` 的现有资格判断中加入：

  ```cpp
  fault->detect == TRUE
  ```

  满足该条件时 `continue`。保留现有 Primary、`test_tried`、`REDUNDANT` 和 attempted-ID 条件。

- [x] **Step 4: 编译并运行聚焦回归**

  Run: `python PODEM/setup.py build_ext --inplace`

  Run: `python -m pytest PODEM/tests/test_fault_mapping.py -q`

  Expected: 该回归及全部原生测试通过。

- [x] **Step 5: 运行 Python 边界测试**

  Run: `C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py tests/test_runtime_progress.py -q`

  Expected: Python remaining-set 校验及 RL 测试通过。

- [x] **Step 6: 复测 `b12_C`**

  使用 BFS 原顺序、DTC 开启、STC 关闭完成一次运行；断言每个 batch 都是当前 remaining set 的子集，并记录 patterns、DTC calls、coverage 和 backtracks。之前受 stale candidates 影响的 547-pattern 数据作废。

- [x] **Step 7: 提交**

  ```text
  git add PODEM/src/saf_compaction.cpp PODEM/tests/test_fault_mapping.py docs/superpowers/specs/2026-09-22-bfs-filtered-ranked-dtc-design.md docs/superpowers/plans/2026-09-22-stale-dtc-candidate-fix.md
  git commit -m "fix: exclude detected faults from DTC batches"
  ```

## 验证结果

- 修复前聚焦回归稳定失败：`q1:GO:sa0` 已在第一个 pattern 中检测，但再次进入下一个 Primary 的 batch。
- 修复后原生回归：30 passed，37 subtests passed。
- Python/RL 边界回归：17 passed。
- `b12_C`、STC 关闭：DTC 开启为 152 patterns / 7,322 secondary calls；DTC 关闭为 215 patterns。
