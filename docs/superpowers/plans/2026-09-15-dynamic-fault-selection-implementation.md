# 动态故障选择与 Stuck-at 测试压缩实施计划

> **面向智能体执行者：** 必须使用子技能 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项执行本计划。各步骤使用复选框（`- [ ]`）跟踪状态。

**目标：** 将固定完整排列策略改为逐步动态故障选择，并让 native baseline、训练和评估统一使用主 PODEM 回溯上限 200、stuck-at PODEMX DTC 以及确定性 STC。

**架构：** C++ 会话每次处理一个 primary fault；TRUE 后先用 PODEMX 将 secondary fault 约束加入测试 cube，再进行一次 fault simulation，全部 primary 尝试结束后执行 reverse-order 与固定 seed shuffle 的 STC。Python 每一步根据候选 fault embedding、剩余集合均值和剩余比例构造 515 维特征；回合结束后重放 masked categorical 轨迹，并用 STC 后 pattern 数计算 REINFORCE 奖励。

**技术栈：** C++14、pybind11、Python 3.9+、PyTorch、NumPy、pytest。

**设计规格：** `docs/superpowers/specs/2026-09-15-dynamic-fault-selection-design.md`

## 全局约束

- DeepGate2 fault embedding 始终冻结且严格为 257 维。
- 评分器输入严格为 `[fault embedding (257), remaining mean (257), remaining ratio (1)]`，共 515 维。
- 每次 TRUE、FALSE 或 MAYBE 的 primary 尝试后都重新计算 remaining mask、mean、ratio 和候选分数。
- primary PODEM 每个 fault 的回溯上限固定为 200，seed 固定为 14，每个 fault 只尝试一次。
- stuck-at DTC 始终开启；secondary fault 回溯上限固定为 50，按当前 catalog 行号升序尝试，不使用 SCOAP。
- stuck-at STC 始终开启；先 reverse-order compaction，再用独立 seed 7 shuffle，连续 5 次无改进后停止。
- `podem_calls == primary_podem_calls`；`total_backtracks == primary_backtracks + dtc_backtracks`。
- 训练 reward、best 比较和最终评估使用 `patterns_after_stc`；逐步轨迹使用单调递增的 `current_pattern_count`。
- STC 前后已检测 fault 集合及 equivalent coverage 必须完全一致，否则当前轮失败并回滚。
- 轨迹 log probability 按初始 catalog fault 数归一化，不能按实际决策数归一化。
- 保留 coverage guard、EMA advantage、Adam、温度调度、小批次事务和 RNG 恢复语义。
- Schema 2 静态检查点和压缩协议不匹配的检查点都必须被明确拒绝；新运行使用 Schema 3。
- 保留用户已有的 `README.md` 和 `docs/fault-reorder-algorithm.md` 改动。

---

### 任务 1：加固增量会话、固定求解协议并隔离随机数

**涉及文件：**
- 修改：`PODEM/src/atpg.h`
- 修改：`PODEM/src/atpg.cpp`
- 修改：`PODEM/src/podem.cpp`
- 修改：`PODEM/src/python_bindings.cpp`
- 测试：`PODEM/tests/test_fault_mapping.py`

**接口：**
- 使用：当前分支已有的 `ATPG::step_stuck_at` 和 `cpp_podem.StuckAtSession`。
- 产出：`StuckAtProtocolConfig`、`StuckAtSession.config()`、会话级填充 RNG，以及线程安全且可重复的基础步进会话。

- [ ] **步骤 1：从重构前版本取得独立 golden 结果**

在临时 Git worktree 中检出 `de86cdd`，使用测试内生成的 TRUE、FALSE 和受限回溯 MAYBE 微型电路运行旧 `run_stuck_at_ordered()`。把完整结果字典、故障 ID 顺序和固定 seed 输出记录为测试常量；不得恢复或依赖 `c1908_binary.bench`。完成后移除临时 worktree。

- [ ] **步骤 2：为固定协议和可重复会话编写预期失败的测试**

增加以下测试；微型 BENCH 继续使用测试内生成的 fixture，不依赖已废弃的 sample circuit：

```python
def test_stuck_at_session_reports_fixed_protocol(self):
    _, binary, fault_map, _, _ = self.convert(
        "INPUT(a)\nINPUT(b)\nOUTPUT(y)\ny = AND(a,b)\n"
    )
    session = cpp_podem.StuckAtSession(str(binary), str(fault_map))
    assert session.config() == {
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
    }

def test_interleaved_sessions_have_identical_primary_traces(self):
    binary, fault_map = self.make_three_input_and_fixture()
    left = cpp_podem.StuckAtSession(str(binary), str(fault_map))
    right = cpp_podem.StuckAtSession(str(binary), str(fault_map))
    left_trace, right_trace = [], []
    while left.remaining_fault_ids():
        fault_id = left.remaining_fault_ids()[0]
        assert fault_id == right.remaining_fault_ids()[0]
        left_trace.append(left.step(fault_id))
        right_trace.append(right.step(fault_id))
    assert left_trace == right_trace
```

补充精确状态测试：TRUE 必须增加一个 raw pattern；FALSE/MAYBE 不增加；同一 ID 第二次调用和未知 ID 在进入 PODEM 前失败。native-first 会话在关闭新压缩的回归模式下必须匹配步骤 1 的独立 golden，而不能只与复用同一实现路径的 `run_stuck_at_ordered()` 相互比较。

- [ ] **步骤 3：运行测试并确认固定协议接口缺失**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -k "fixed_protocol or interleaved or incremental_status" -v
```

预期结果：FAIL，因为 `config()`、200 默认值和会话级 RNG 尚未实现。

- [ ] **步骤 4：定义协议配置并移除全局随机数依赖**

在 `atpg.h` 中定义：

```cpp
struct StuckAtProtocolConfig {
    int primary_backtrack_limit{200};
    int primary_seed{14};
    int attempts_per_primary_fault{1};
    bool dtc_enabled{true};
    int dtc_secondary_backtrack_limit{50};
    bool stc_enabled{true};
    bool stc_reverse_order_enabled{true};
    int stc_shuffle_seed{7};
    int stc_no_improvement_limit{5};
    bool scoap_enabled{false};
};
```

`configure_ordered_stuck_at` 接收并保存该结构。为 `ATPG` 增加两个独立的 `std::mt19937`：primary cube 填充 RNG 用 seed 14，STC shuffle RNG 用 seed 7。将 `podem()` 中的随机填充移出搜索函数；`podem()` 成功时保留 U 值，由 `step_stuck_at()` 在 DTC 结束后统一填充。不得再调用共享的 `srand/rand`。

- [ ] **步骤 5：让绑定使用固定默认值并串行化单会话操作**

`StuckAtSession` 默认构造固定协议，并用实例 mutex 保护 `remaining_fault_ids()`、`step()` 和 `result()`。GIL 可以在 C++ 重计算期间释放，但同一个会话对象不能同时执行两个可变操作。绑定 `config()` 返回只读协议字典。`run_stuck_at_ordered` 和 `StuckAtSession` 的公开默认回溯上限都改为 200；仅测试用回归路径可显式关闭 DTC/STC。

- [ ] **步骤 6：构建扩展并运行完整 fault-mapping 测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -v
```

预期结果：全部 PASS；重复和交错会话得到相同 primary trace。

- [ ] **步骤 7：提交基础会话加固**

```powershell
git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/podem.cpp PODEM/src/python_bindings.cpp PODEM/tests/test_fault_mapping.py
git commit -m "fix: harden incremental stuck-at sessions"
```

---

### 任务 2：实现 Stuck-at PODEMX 动态测试压缩

**涉及文件：**
- 创建：`PODEM/src/saf_compaction.cpp`
- 修改：`PODEM/setup.py`
- 修改：`PODEM/src/atpg.h`
- 修改：`PODEM/src/atpg.cpp`
- 修改：`PODEM/src/python_bindings.cpp`
- 测试：`PODEM/tests/test_fault_mapping.py`

**接口：**
- 使用：任务 1 保留 U 输入的 primary test cube 和固定协议。
- 产出：`ATPG::run_stuck_at_dtc(fptr primary) -> DtcResult`、每步 DTC IDs/统计，以及 fault simulation 后的真实 drop 集合。

- [ ] **步骤 1：编写 DTC 成功、失败回滚和计数测试**

```python
def test_stuck_at_dtc_reports_secondary_attempts_without_primary_selection(self):
    binary, fault_map = self.make_dtc_fixture()
    session = cpp_podem.StuckAtSession(str(binary), str(fault_map))
    primary = session.remaining_fault_ids()[0]
    step = session.step(primary)
    assert step["target_status"] == "detected"
    assert primary not in step["dtc_attempted_fault_ids"]
    assert set(step["dtc_embedded_fault_ids"]) <= set(step["dtc_attempted_fault_ids"])
    assert step["current_dtc_secondary_calls"] == len(step["dtc_attempted_fault_ids"])
    assert step["current_total_backtracks"] == (
        step["current_primary_backtracks"] + step["current_dtc_backtracks"]
    )

def test_failed_stuck_at_dtc_attempt_restores_primary_cube(self):
    enabled = self.run_failure_only_dtc_fixture(dtc_enabled=True)
    disabled = self.run_failure_only_dtc_fixture(dtc_enabled=False)
    assert enabled["target_status"] == "detected"
    assert enabled["dtc_attempted_fault_ids"]
    assert enabled["dtc_embedded_fault_ids"] == []
    assert enabled["generated_test_vector"] == disabled["generated_test_vector"]
```

fixture 必须同时包含：一个能加入当前 cube 的 secondary fault，以及一个在 50 次回溯内失败的 secondary fault。

- [ ] **步骤 2：运行 DTC 测试并确认字段和实现缺失**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -k "stuck_at_dtc" -v
```

预期结果：FAIL，因为 DTC runner 和诊断字段尚不存在。

- [ ] **步骤 3：实现 secondary PODEMX 搜索与事务回滚**

在 `saf_compaction.cpp` 中定义：

```cpp
struct DtcResult {
    int secondary_calls{};
    int backtracks{};
    vector<string> attempted_fault_ids;
    vector<string> embedded_fault_ids;
};
```

`run_stuck_at_dtc` 排除 primary，按 catalog 行号升序扫描当前 selectable fault。每次尝试先保存全部 PI 值和 PODEM 决策标志，只允许为 U 输入赋值，最多回溯 50 次。成功后保留新增约束；FALSE/MAYBE 时恢复本次快照。每次成功后重新验证 primary 和此前 embedded secondary 都仍能传播到 PO；验证失败同样回滚。

- [ ] **步骤 4：把 DTC 接入 TRUE 步骤**

`step_stuck_at()` 的 TRUE 分支顺序严格为：

```text
primary PODEM cube
-> run_stuck_at_dtc(primary)
-> 使用 primary RNG 填充剩余 U
-> 生成一个完整 vector
-> fault_sim_a_vector 一次
-> 由 fault simulation 前后差集生成 newly_detected_fault_ids
```

secondary 尝试不得设置 `test_tried`、REDUNDANT 或 aborted；只有最终 fault simulation 能把它从 remaining 集合 drop。向 `AtpgRunResult` 增加 `primary_podem_calls`、`dtc_secondary_calls`、`primary_backtracks` 和 `dtc_backtracks`，并保持 `podem_calls == primary_podem_calls`、`total_backtracks == primary_backtracks + dtc_backtracks`。向 `AtpgStepResult` 和 Python 字典增加 `generated_test_vector`、`dtc_attempted_fault_ids`、`dtc_embedded_fault_ids`、`current_dtc_secondary_calls`、`current_primary_backtracks` 和 `current_dtc_backtracks`。

- [ ] **步骤 5：运行 DTC、增量会话和确定性测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -k "dtc or incremental or deterministic" -v
```

预期结果：全部 PASS；DTC 失败不改变 primary vector，成功 secondary 只以 fault-sim drop 身份退出。

- [ ] **步骤 6：提交 stuck-at DTC**

```powershell
git add PODEM/setup.py PODEM/src/saf_compaction.cpp PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/python_bindings.cpp PODEM/tests/test_fault_mapping.py
git commit -m "feat: add stuck-at dynamic test compression"
```

---

### 任务 3：实现确定性 Stuck-at 静态测试压缩

**涉及文件：**
- 修改：`PODEM/src/saf_compaction.cpp`
- 修改：`PODEM/src/atpg.h`
- 修改：`PODEM/src/atpg.cpp`
- 修改：`PODEM/src/python_bindings.cpp`
- 测试：`PODEM/tests/test_fault_mapping.py`

**接口：**
- 使用：任务 2 生成的完整 stuck-at vectors 和最终求解状态。
- 产出：幂等的 `ATPG::finalize_stuck_at_session()`，以及区分 STC 前后 pattern 的最终摘要。

- [ ] **步骤 1：编写 reverse、shuffle、覆盖保持和幂等测试**

```python
def test_stc_finalization_is_deterministic_idempotent_and_coverage_preserving(self):
    session = self.complete_compressible_session()
    first = session.result()
    second = session.result()
    assert first == second
    assert first["finalized"] is True
    assert first["pattern_count"] == first["patterns_after_stc"]
    assert first["patterns_after_stc"] <= first["patterns_before_stc"]
    assert first["stc_removed_patterns"] == (
        first["patterns_before_stc"] - first["patterns_after_stc"]
    )
    assert first["stc_coverage_preserved"] is True
```

另测 remaining 非空时 `result()["finalized"] is False`，且此时 `pattern_count == current_pattern_count`，不提前运行 STC。

- [ ] **步骤 2：运行 STC 测试并确认失败**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -k "stc or finalization" -v
```

预期结果：FAIL，因为 stuck-at STC finalization 尚不存在。

- [ ] **步骤 3：实现 coverage-preserving reverse-order compaction**

在不覆盖 REDUNDANT、aborted、calls 或 backtracks 状态的前提下，对 vectors 反向扫描。每个候选 vector 只在它对完整 pre-STC detected fault 集合仍有独占贡献时保留。压缩使用独立的检测位集或显式保存/恢复 fault 状态，不能复用会破坏主会话状态的旧 TDF `reverse_order_fault_sim()`。

- [ ] **步骤 4：实现固定 seed shuffle compaction 与原子 finalize**

使用会话级 `std::mt19937 stc_rng_{7}` 对当前 vectors shuffle；每次执行与 reverse 阶段相同的覆盖保持删除。连续 5 次不能减少向量时停止。`finalize_stuck_at_session()` 只在 selectable 为空时执行一次，并缓存：

```text
finalized
patterns_before_stc
patterns_after_stc
pattern_count
stc_removed_patterns
stc_shuffle_attempts
stc_coverage_preserved
```

若压缩前后 detected catalog-ID 集合或 equivalent count 不一致，抛出异常；不得返回 coverage 降低的正常结果。

- [ ] **步骤 5：运行完整 C++ 绑定测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -v
```

预期结果：全部 PASS，重复 `result()` 不会再次 shuffle。

- [ ] **步骤 6：提交 stuck-at STC**

```powershell
git add PODEM/src/saf_compaction.cpp PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/python_bindings.cpp PODEM/tests/test_fault_mapping.py
git commit -m "feat: compact stuck-at test patterns"
```

---

### 任务 4：构造 515 维动态特征与类别策略

**涉及文件：**
- 修改：`fault_order_rl/model.py`
- 修改：`fault_order_rl/policy.py`
- 修改：`tests/test_fault_order_rl.py`

**接口：**
- 使用：已校验的 `[N,257]` 冻结 embedding 和 catalog 行索引。
- 产出：`build_dynamic_features`、单步 categorical action helper 和 `trajectory_log_prob`。

- [ ] **步骤 1：编写 515 维特征测试**

```python
def test_dynamic_features_append_remaining_mean_and_ratio():
    embeddings = torch.arange(4 * 257, dtype=torch.float32).reshape(4, 257)
    rows = torch.tensor([1, 3])
    features = build_dynamic_features(embeddings, rows)
    assert features.shape == (2, 515)
    assert torch.equal(features[:, :257], embeddings[rows])
    mean = embeddings[rows].mean(dim=0)
    assert torch.equal(features[:, 257:514], mean.expand(2, -1))
    assert torch.equal(features[:, 514], torch.full((2,), 0.5))
```

同时测试空、重复、越界及非一维 rows 抛出 `ValueError`，`FaultScorer` 拒绝 `[N,257]` 并接受 `[N,515]`。

- [ ] **步骤 2：运行测试并确认特征构造器缺失**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_features or scorer" -v
```

- [ ] **步骤 3：实现 515 维非线性评分器**

保持以下结构：

```text
LayerNorm(515) -> Linear(515,256) -> ReLU
-> Linear(256,128) -> ReLU -> Linear(128,1)
```

`build_dynamic_features` 不修改或克隆源 embedding；ratio 为 `K/N`，mean 和 ratio 扩展到 K 个候选。

- [ ] **步骤 4：编写 categorical 采样与轨迹重放的预期失败测试**

```python
def test_dynamic_trajectory_log_prob_matches_direct_steps():
    embeddings = torch.randn(4, 257, generator=torch.Generator().manual_seed(9))
    model = FaultScorer()
    decisions = [
        {"remaining_rows": (0, 1, 2, 3), "selected_row": 2},
        {"remaining_rows": (1, 3), "selected_row": 3},
    ]
    actual = trajectory_log_prob(model, embeddings, decisions, 0.7)
    direct = []
    for state in decisions:
        rows = torch.tensor(state["remaining_rows"])
        logits = centered_logits(model(build_dynamic_features(embeddings, rows)), 0.7)
        local = state["remaining_rows"].index(state["selected_row"])
        direct.append(torch.log_softmax(logits, 0)[local])
    assert torch.allclose(actual, torch.stack(direct).sum())
```

- [ ] **步骤 5：实现 categorical action helper 与轨迹重放**

训练选择使用 `torch.multinomial(torch.softmax(logits, 0), 1, generator=generator)`；评估选择稳定的第一个最大值。重放时校验 selected row 在保存的 remaining rows 中恰好出现一次。

- [ ] **步骤 6：运行策略测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_features or dynamic_trajectory or categorical or scorer" -v
```

预期结果：全部 PASS，并产生有限的非零 scorer 梯度。

- [ ] **步骤 7：提交动态模型和策略计算**

```powershell
git add fault_order_rl/model.py fault_order_rl/policy.py tests/test_fault_order_rl.py
git commit -m "feat: add dynamic fault context policy"
```

---

### 任务 5：实现带压缩协议校验的 Python 会话环境

**涉及文件：**
- 修改：`fault_order_rl/environment.py`
- 修改：`tests/test_fault_order_rl.py`

**接口：**
- 使用：任务 1～3 的 `cpp_podem.StuckAtSession`。
- 产出：`PodemEnvironment.start_session(...) -> PodemSession`，以及经严格校验的 step/final result。

- [ ] **步骤 1：为 step 和 final result 编写预期失败的校验测试**

假的绑定会话第一步把 `("f0","f1","f2")` 变为 `("f2",)`，并报告 `newly_detected_fault_ids=("f0","f1")`。测试：

```python
def test_session_wrapper_validates_compressed_protocol(fake_binding):
    session = PodemEnvironment(fake_binding).start_session("c.bench", "c.map")
    assert session.config["primary_backtrack_limit"] == 200
    step = session.step("f0")
    assert step["remaining_fault_ids"] == ("f2",)
    final = session.finish()
    assert final["pattern_count"] == final["patterns_after_stc"]
    assert final["patterns_after_stc"] <= final["patterns_before_stc"]
    assert final["stc_coverage_preserved"] is True
```

分别构造并拒绝：重复/未知 remaining ID、remaining 集合增长、TRUE 不增加 raw pattern、FALSE/MAYBE 增加 pattern、calls/backtracks 倒退、`total_backtracks` 分解错误、提前 finalized、STC 后 coverage 改变，以及缺失结果字段。

- [ ] **步骤 2：运行包装层测试并确认失败**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "session_wrapper or compressed_protocol" -v
```

预期结果：FAIL，因为 `start_session`、`PodemSession` 和压缩字段校验尚未完成。

- [ ] **步骤 3：实现共享结果校验与状态机**

提取 `_validate_result(raw, catalog_size, require_finalized)`。`PodemSession` 保存不可变初始 IDs、上一步 remaining tuple、上一步 raw pattern/calls/backtracks，并验证：

```text
selected ID 属于上一步 remaining
新 remaining 是已知、唯一的严格子集
newly_detected IDs 已知且不再 remaining
TRUE 恰好增加一个 current_pattern_count
FALSE/MAYBE 不改变 current_pattern_count
primary calls 每步加一
primary/dtc backtracks 各自不减
total_backtracks 等于两者之和
finish 时 remaining 为空且 finalized=true
pattern_count 等于 patterns_after_stc
patterns_after_stc 不大于 patterns_before_stc
STC coverage preserved 为真
```

`PodemEnvironment.run` 保留兼容支持，并将完整字典送入同一个 validator。所有创建路径固定使用设计规格中的协议参数，不接受训练 CLI 覆盖。

将最终结果字段扩展为：

```python
RESULT_FIELDS = (
    "pattern_count", "patterns_before_stc", "patterns_after_stc",
    "detected_collapsed_faults", "detected_equivalent_faults",
    "uncollapsed_faults", "aborted_faults", "redundant_faults",
    "redundant_equivalent_faults", "podem_calls", "primary_podem_calls",
    "dtc_secondary_calls", "primary_backtracks", "dtc_backtracks",
    "total_backtracks", "stc_removed_patterns", "stc_shuffle_attempts",
    "stc_coverage_preserved", "finalized",
)
```

- [ ] **步骤 4：运行环境测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "environment or session_wrapper or compressed_protocol" -v
```

预期结果：全部 PASS。

- [ ] **步骤 5：提交 Python 环境边界**

```powershell
git add fault_order_rl/environment.py tests/test_fault_order_rl.py
git commit -m "feat: validate compressed PODEM sessions"
```

---

### 任务 6：实现动态训练回合与压缩后奖励

**涉及文件：**
- 修改：`fault_order_rl/trainer.py`
- 修改：`tests/test_fault_order_rl.py`

**接口：**
- 使用：任务 4 的动态策略函数和任务 5 的 `PodemEnvironment.start_session`。
- 产出：动态 `_run_native`、`_run_policy`、可重放 decision states，以及按初始故障数归一化的 REINFORCE 更新。

- [ ] **步骤 1：把训练 fixture 改为带 DTC/STC 的假会话**

`FakeEnvironment.start_session` 返回确定性会话。第一次选择偶数行 fault 时，同时 drop 下一行，模拟 DTC/fault-sim；`finish()` 返回：

```python
{
    "finalized": True,
    "patterns_before_stc": raw_patterns,
    "patterns_after_stc": max(0, raw_patterns - 1),
    "pattern_count": max(0, raw_patterns - 1),
    "stc_coverage_preserved": True,
    "detected_collapsed_faults": detected,
    "detected_equivalent_faults": detected,
    "uncollapsed_faults": len(ids),
    "aborted_faults": 0,
    "redundant_faults": 0,
    "redundant_equivalent_faults": 0,
    "podem_calls": primary_calls,
    "primary_podem_calls": primary_calls,
    "dtc_secondary_calls": dtc_calls,
    "primary_backtracks": primary_backtracks,
    "dtc_backtracks": dtc_backtracks,
    "total_backtracks": primary_backtracks + dtc_backtracks,
    "stc_removed_patterns": min(1, raw_patterns),
    "stc_shuffle_attempts": 5,
}
```

- [ ] **步骤 2：编写动态回合、STC 奖励和归一化测试**

```python
def test_dynamic_episode_rebuilds_state_after_fault_drop(trainer, circuit):
    metrics, elapsed, decisions, trace = trainer._run_policy(
        circuit, trainer.model, temperature=1.0, stochastic=False
    )
    assert decisions[0]["remaining_rows"] == tuple(range(circuit.fault_count))
    assert len(decisions[1]["remaining_rows"]) < circuit.fault_count - 1
    assert trace[1]["remaining_ratio"] == (
        len(decisions[1]["remaining_rows"]) / circuit.fault_count
    )
    assert metrics["pattern_count"] == metrics["patterns_after_stc"]

def test_loss_uses_initial_fault_count_not_decision_count():
    log_prob = torch.tensor(2.0, requires_grad=True)
    loss = _normalized_policy_loss(
        advantage=3.0,
        log_prob=log_prob,
        initial_fault_count=5,
        batch_size=1,
    )
    assert torch.allclose(loss, torch.tensor(-1.2))
```

另测 reward 使用 `patterns_after_stc` 而不是 `patterns_before_stc`；若 `stc_coverage_preserved=False`，整轮必须失败且参数、optimizer、baseline、EMA、RNG 均不提交。

- [ ] **步骤 3：运行训练器测试并确认仍在请求静态排列**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_episode or stc_reward or initial_fault_normalization" -v
```

- [ ] **步骤 4：实现 native 和 policy 会话 runner**

`_run_native` 每次选择 session 返回的第一个 ID。`_run_policy` 将 session IDs 映射到不可变 catalog rows，每一步重新构造 `[K,515]` 特征，采样或确定性选择一个局部候选，执行一次 step，并记录：

```python
{"remaining_rows": tuple(rows), "selected_row": selected_row}
```

人工 trace 另存 selected ID/score、remaining count/ratio、target status、测试向量、DTC attempted/embedded IDs、newly detected IDs、拆分后的 calls/backtracks 和 raw pattern count。循环结束后调用 `finish()` 获取 STC 最终摘要。

- [ ] **步骤 5：实现 no-grad 收集与 autograd 重放**

同一 batch 的所有 episode 使用一个不变的候选模型快照在 `torch.no_grad()` 下收集。随后逐回合调用 `trajectory_log_prob`，严格计算：

```python
loss = (
    -record["advantage"]
    * log_prob
    / circuit.fault_count
    / len(batch)
)
```

收集与重放之间不得更新模型。reward 和 previous-pattern baseline 使用 `patterns_after_stc`。
将公式封装为 `_normalized_policy_loss`，对 `initial_fault_count <= 0` 或
`batch_size <= 0` 抛出 `ValueError`，供精确单元测试复用。

- [ ] **步骤 6：保存紧凑动态轨迹和压缩统计**

每个电路在 round NPZ 中保存 `<name>__selected_rows`、`<name>__remaining_offsets`、`<name>__remaining_rows`；在 round metrics 中保存 `patterns_before_stc`、`patterns_after_stc`、DTC calls、primary/DTC backtracks 和 STC shuffle attempts。所有数组必须支持 `allow_pickle=False`。

- [ ] **步骤 7：运行恢复、小批次和事务测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "shared_update or minibatch or failed_round or dynamic_episode or stc_reward or initial_fault_normalization" -v
```

预期结果：连续与恢复运行生成逐字节相同的轨迹数组、相同模型参数、baseline 和 EMA。

- [ ] **步骤 8：提交动态训练器**

```powershell
git add fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: train on compressed dynamic trajectories"
```

---

### 任务 7：升级 Schema 3 检查点与动态确定性评估

**涉及文件：**
- 修改：`fault_order_rl/checkpoint.py`
- 修改：`fault_order_rl/trainer.py`
- 修改：`tests/test_fault_order_rl.py`

**接口：**
- 使用：任务 6 的确定性 `_run_policy(..., stochastic=False)` 和压缩后摘要。
- 产出：带协议身份的 Schema 3 检查点、初始排序诊断 NPZ 和 `<circuit>.trajectory.jsonl`。

- [ ] **步骤 1：编写检查点迁移和协议身份测试**

```python
with pytest.raises(ValueError, match="detected-only coverage"):
    load_checkpoint(schema1)
with pytest.raises(ValueError, match="static 257-dimensional policy"):
    load_checkpoint(schema2)

payload = make_schema3_payload()
assert payload["policy_version"] == "dynamic_remaining_mean_v1"
assert payload["input_dimension"] == 515
assert payload["solver_protocol"] == {
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
    "compression_algorithm_version": "stuck_at_podemx_reverse_shuffle_v1",
}
```

分别修改每个协议字段并断言 resume/evaluate 明确拒绝。

- [ ] **步骤 2：升级 checkpoint payload 与兼容性校验**

`load_checkpoint` 只接受 Schema 3。latest 和 derived-best payload 都保存动态策略身份、515 维输入、完整 solver protocol、模型/optimizer/baseline/EMA/RNG 状态。Schema 1 保留原错误信息，Schema 2 给出必须重新训练的错误；协议身份不一致不得部分加载。

- [ ] **步骤 3：编写动态评估产物测试**

每个电路的 ranking NPZ 必须包含：

```text
fault_ids, scores, ranks, permutation,
selected_rows, selected_scores, selected_steps,
exit_steps, exit_reasons,
patterns_before_stc, patterns_after_stc,
primary_podem_calls, dtc_secondary_calls,
primary_backtracks, dtc_backtracks, total_backtracks
```

`scores/ranks/permutation` 只代表初始状态诊断。被其他 pattern drop 的 fault 必须 `selected_steps == -1`。JSONL 同一步必须区分 `selected_fault_id`、`dtc_attempted_fault_ids`、`dtc_embedded_fault_ids` 和 `newly_detected_fault_ids`，最后增加一条 STC summary。

- [ ] **步骤 4：实现确定性动态评估与原子产物**

删除静态 `model(circuit.embeddings)` 加完整顺序运行。通过动态会话逐步选择；只用第一状态生成初始 ranking。根据 trace 派生 selected/exit arrays，exit reason 使用 Unicode dtype，保证 `np.load(..., allow_pickle=False)`。ranking NPZ、trajectory JSONL 和 metrics JSON 都使用原子写入。

外部 manifest 的可恢复评估只有在三类文件都存在，且 checkpoint/evaluation/solver-protocol identity 全部匹配时才视为完成。

- [ ] **步骤 5：运行 checkpoint 与评估测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "checkpoint or evaluation or latest or best or solver_protocol" -v
```

预期结果：全部 PASS；重新确定性评估与保存的 best 指标完全一致。

- [ ] **步骤 6：提交评估和检查点迁移**

```powershell
git add fault_order_rl/checkpoint.py fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: evaluate compressed dynamic trajectories"
```

---

### 任务 8：真实求解器集成、文档和完整验证

**涉及文件：**
- 修改：`tests/test_fault_order_rl.py`
- 修改：`docs/fault-order-rl.md`

**接口：**
- 使用：完成的 Schema 3 动态压缩策略与重建后的 `cpp_podem`。
- 产出：真实微型电路上的端到端证据和与实现一致的运行文档。

- [ ] **步骤 1：扩展真实 PODEM 冒烟测试**

在测试内转换微型 BENCH，运行一轮真实动态训练和确定性评估，断言：

```python
assert metrics["patterns_after_stc"] <= metrics["patterns_before_stc"]
assert metrics["stc_coverage_preserved"] is True
assert metrics["total_backtracks"] == (
    metrics["primary_backtracks"] + metrics["dtc_backtracks"]
)
assert round_npz_contains_dynamic_masks
assert ranking_npz_exists
assert trajectory_jsonl_exists
assert native_resolved_coverage_is_preserved
```

- [ ] **步骤 2：运行真实求解器冒烟测试**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k real_podem -v
```

预期结果：真实 C++ session 完成 primary PODEM、DTC、fault simulation、STC 和有限的 optimizer update。

- [ ] **步骤 3：更新运行文档**

在 `docs/fault-order-rl.md` 中记录：515 维布局、remaining mask、逐步采样/确定性选择、轨迹 log probability、初始 fault 数归一化、主回溯 200、DTC 回溯 50、DTC secondary 语义、STC reverse/shuffle 参数、压缩后奖励、Schema 3 重启要求，以及初始 ranking 与真实执行轨迹的区别。删除“每回合生成一个完整 permutation”和“STC/DTC 关闭”的现行描述。

- [ ] **步骤 4：运行格式和过时配置检查**

```powershell
git diff --check
rg -n "backtrack limit=5000|backtrack_limit=5000|STC、DTC、SCOAP.*关闭|complete permutation" docs/fault-order-rl.md fault_order_rl PODEM/src
```

预期结果：diff 无格式错误；命中项只能描述历史迁移背景，不能代表现行协议。

- [ ] **步骤 5：运行完整相关测试套件**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py PODEM/tests/test_fault_mapping.py -v
```

预期结果：全部 PASS。

- [ ] **步骤 6：校验一个生产 manifest，不启动训练**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m fault_order_rl validate --manifest configs/anchor_smoke_train.json
```

预期结果：manifest、catalog、embedding 维度、ID、等价故障数和 provenance 全部通过。

- [ ] **步骤 7：审查完整改动**

调用 `requesting-code-review` 技能。解决所有正确性问题；先运行最小受影响测试，再运行完整套件。确认 `git status --short` 没有构建产物或无关的已暂存改动。

- [ ] **步骤 8：提交文档与集成测试**

```powershell
git add tests/test_fault_order_rl.py docs/fault-order-rl.md
git commit -m "docs: describe compressed dynamic fault selection"
```

- [ ] **步骤 9：报告验证证据**

报告准确的通过命令、微型动态压缩电路、检查点不兼容情况、已知限制，以及设计规格和实施计划路径。冒烟测试只能证明流程正确，不能用于宣称真实训练已经减少 pattern 数。
