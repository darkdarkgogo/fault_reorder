# TDF Lazy Heuristic Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让所有 heuristic/native baseline 严格使用 TDF 的 lazy `q_wire`/`q_fault` 调度，同时保持 RL 对预算内完整 BFS batch 排序。

**Architecture:** 保留 `find_next_stuck_at_dtc_batch()` 专供显式 RL 阶段协议。将单个 secondary fault 的 PODEMX、preserved-fault 检查、cube 回滚与 PO 检查抽为共享 helper，再新增 TDF 同构的 lazy baseline 循环；`step_stuck_at()` 走 lazy 路径，`begin_stuck_at_step()`/`rank_stuck_at_dtc_candidates()` 仍走完整 batch 路径。

**Tech Stack:** C++14 ATPG core、pybind11、Python 3.9+、pytest、setuptools。

**Spec:** `docs/superpowers/specs/2026-09-23-tdf-lazy-baseline-design.md`

## Global Constraints

- Baseline 只在 `q_fault` 为空时继续展开 `q_wire`。
- 每尝试一个 secondary 后立即恢复 accepted good cube 并检查当前 PO。
- RL 继续返回预算内完整 candidate batch，且只接受完整 permutation。
- `ncktin <= 32` 的 wire 展开预算为 15，否则为 100。
- 不改变 Primary 选择、PODEMX、STC、fault simulation 或 Python 公开签名。
- 保留已检测/冗余/已尝试 fault 过滤和 reconvergence 去重，不重现旧 TDF 的缺陷。

---

### Task 1: 用聚焦回归锁定 lazy baseline 与 ranked RL 的差异

**Files:**
- Modify: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Consumes: `cpp_podem.StuckAtSession.begin_step()`、`rank_dtc_candidates()`、`step()` 和 `run_stuck_at_ordered()`。
- Produces: 能区分“baseline lazy 展开”与“RL 完整 batch”的真实小电路测试。

- [ ] **Step 1: 增加 lazy DTC fixture**

构造 Primary 成功后保留多个 `U` PI 的小电路，使近层 wire 候选能解析 PO，而深层 wire 仍有 eligible fault。返回 `binary, fault_map, primary, near_fault, deep_fault`。

- [ ] **Step 2: 编写 baseline lazy 与入口一致性测试**

```python
def test_native_dtc_is_tdf_lazy_and_ordered_runner_matches_session(self):
    binary, fault_map, primary, near_fault, deep_fault = self.make_lazy_dtc_fixture()
    session = cpp_podem.StuckAtSession(str(binary), str(fault_map), stc_enabled=False)
    step = session.step(primary)
    self.assertEqual(step["dtc_attempted_fault_ids"], [near_fault])
    self.assertNotIn(deep_fault, step["dtc_attempted_fault_ids"])
```

用同一 Primary 顺序跑完 session 和 `run_stuck_at_ordered()`，断言最终 summary 一致。

- [ ] **Step 3: 编写 RL 完整 batch 与逐 fault PO-stop 测试**

```python
state = session.begin_step(primary)
candidates = list(state["dtc_candidate_fault_ids"])
self.assertIn(near_fault, candidates)
self.assertIn(deep_fault, candidates)
ranking = [near_fault] + [fault for fault in candidates if fault != near_fault]
state = session.rank_dtc_candidates(ranking)
self.assertEqual(state["last_dtc_attempted_fault_ids"], [near_fault])
```

- [ ] **Step 4: 重建扩展并确认 baseline 用例先失败**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM\tests\test_fault_mapping.py -k "tdf_lazy or ranked_dtc_still" -q
```

Expected: RL 用例通过，baseline lazy 用例因 `step_stuck_at()` 仍预收集完整 batch 而失败。

### Task 2: 实现共享 secondary 原子操作和 TDF lazy baseline

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/atpg.cpp`
- Modify: `PODEM/src/saf_compaction.cpp`
- Test: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Consumes: `stuck_at_podemx_secondary(fptr, int&)`、`restore_stuck_at_good_cube()`、`stuck_at_cube_detects()` 和现有 active-step state。
- Produces: `attempt_stuck_at_dtc_secondary(fptr, wptr) -> bool`、`run_stuck_at_lazy_dtc() -> void`、`begin_stuck_at_step_impl(const string&, bool) -> StuckAtPhaseResult`。

- [ ] **Step 1: 声明共享 helper 和内部 begin mode**

```cpp
StuckAtPhaseResult begin_stuck_at_step_impl(
    const string &fault_id, bool expose_ranked_dtc);
bool attempt_stuck_at_dtc_secondary(fptr secondary, wptr unknown_po);
void run_stuck_at_lazy_dtc();
```

`begin_stuck_at_step()` 调用 `begin_stuck_at_step_impl(fault_id, true)`；`step_stuck_at()` 调用 `begin_stuck_at_step_impl(fault_id, false)`。

- [ ] **Step 2: 从 ranked 循环抽出单 fault 尝试**

```cpp
bool ATPG::attempt_stuck_at_dtc_secondary(fptr secondary, wptr unknown_po)
{
    const string identifier = fault_identifier(secondary);
    if (!stuck_at_attempted_ids.insert(identifier).second)
        throw runtime_error("A stuck-at DTC secondary was attempted more than once");
    // 依次记录计数、运行 PODEMX、检查 preserved faults、
    // 接受或回滚 cube，最后恢复 good circuit 并检查 PO。
    return unknown_po->value != U;
}
```

`rank_stuck_at_dtc_candidates()` 仍先验证完整 permutation，循环内改为调用 helper；helper 返回 `true` 时立即停止。

- [ ] **Step 3: 按 `tdf_podemx_bt()` 实现 baseline lazy 循环**

```cpp
for (wptr unknown_po : cktout) {
    if (unknown_po->value != U)
        continue;
    queue<wptr> q_wire;
    queue<fptr> q_fault;
    q_wire.push(unknown_po);
    int expanded = 0;
    while (unknown_po->value == U) {
        while (q_fault.empty() && expanded < budget) {
            if (q_wire.empty())
                break;
            // 只展开一个 unique U wire，按 fan-in 顺序入 q_wire，
            // 按 udflist 顺序把 eligible faults 入 q_fault。
        }
        if (q_fault.empty())
            break;
        fptr secondary = q_fault.front();
        q_fault.pop();
        if (attempt_stuck_at_dtc_secondary(secondary, unknown_po))
            break;
    }
}
```

候选资格与 ranked BFS 一致：排除 Primary、`test_tried`、`TRUE`、`REDUNDANT` 和已尝试 ID，并保留 `udflist` 顺序。

- [ ] **Step 4: 分离 ranked begin 与 lazy step**

```cpp
if (dynamic_test_compression && expose_ranked_dtc) {
    // 现有完整 batch 发现与 dtc phase 返回。
} else if (dynamic_test_compression) {
    run_stuck_at_lazy_dtc();
}
```

lazy 路径直接 `complete_stuck_at_step()`；ranked 路径的阶段错误、permutation 校验和 metadata 保持不变。

- [ ] **Step 5: 重建并运行聚焦测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM\tests\test_fault_mapping.py -k "tdf_lazy or ranked_dtc" -q
```

Expected: PASS，baseline 只尝试 lazy prefix，RL 仍看到完整 batch 并在每个 fault 后停于已解析 PO。

### Task 3: 更新文档并运行必要回归

**Files:**
- Modify: `docs/fault-reorder-algorithm.md`
- Modify: `docs/fault-order-rl.md`
- Test: `PODEM/tests/test_fault_mapping.py`
- Test: `tests/test_fault_order_rl.py`
- Test: `tests/test_runtime_progress.py`

**Interfaces:**
- Consumes: Task 2 的 baseline/RL 分叉语义。
- Produces: 与实现一致的算法文档和回归证据。

- [ ] **Step 1: 更新两份算法文档**

```text
Heuristic/native baseline 使用 TDF lazy q_wire/q_fault 调度；
RL 使用预算内完整 BFS batch 排序。
两者在每次 secondary 后都立即检查当前 PO。
```

- [ ] **Step 2: 运行原生完整回归**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM\tests\test_fault_mapping.py -q
```

- [ ] **Step 3: 运行 Python RL 协议与日志回归**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests\test_fault_order_rl.py tests\test_runtime_progress.py -q
```

- [ ] **Step 4: 检查最终 diff 和工作树**

```powershell
git diff --check
git status --short
```

Expected: 全部测试通过，无 whitespace 错误，只有本功能的 C++、测试和文档变更。
