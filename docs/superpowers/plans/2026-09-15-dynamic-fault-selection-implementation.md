# 动态故障选择实施计划

> **面向智能体执行者：** 必须使用子技能 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项执行本计划。各步骤使用复选框（`- [ ]`）跟踪状态。

**目标：** 将固定的完整故障排列策略替换为有状态策略；每次执行 PODEM／故障仿真后，基于每个候选故障的 embedding、当前剩余故障 embedding 的均值和剩余比例重新计算分数。

**架构：** 有状态的 C++ PODEM 会话对外提供“每次处理一个主故障”的步进接口，以及当前可选故障 ID。Python 为每个状态构造 515 维动态特征，并从带 mask 的类别分布中选择一个动作；求解回合结束后，在 autograd 下重放已记录的 mask，应用现有的、受覆盖率门控的 REINFORCE 奖励，并按电路的初始故障数归一化。

**技术栈：** C++11/14、pybind11、Python 3.9+、PyTorch、NumPy、pytest。

**设计规格：** `docs/superpowers/specs/2026-09-15-dynamic-fault-selection-design.md`

## 全局约束

- 每个 DeepGate2 故障 embedding 必须保持冻结，并且维度严格为 257。
- 评分器输入严格为 `[fault embedding (257), remaining mean (257), remaining ratio (1)]`，总计 515 维。
- 每次对主故障尝试得到 TRUE、FALSE 或 MAYBE 后，都必须重新计算上下文和分数。
- 可选故障必须同时满足：尚未检测、不是冗余故障，并且在当前单次尝试回合中尚未尝试过。
- 保持 PODEM 随机种子为 14、回溯上限为 5000、尝试次数为 1，并禁用 STC/DTC/SCOAP/TDF。
- 保留现有的测试向量数量奖励、原生 resolved-coverage 保护条件、EMA advantage、Adam 优化器、温度调度、小批次事务机制和最优模型选择键。
- 轨迹 log probability 必须按 catalog 的初始故障数归一化，不能按实际决策步数归一化。
- 禁止将 Schema 2 的静态策略检查点加载到动态模型中；必须重新开始 Schema 3 训练。
- 除非文档任务明确需要修改重叠内容，否则保留用户在 `README.md` 和 `docs/fault-reorder-algorithm.md` 中已有的改动。

---

### 任务 1：增量式 PODEM 核心与 pybind 会话接口

**涉及文件：**
- 修改：`PODEM/src/atpg.h`
- 修改：`PODEM/src/atpg.cpp`
- 修改：`PODEM/src/python_bindings.cpp`
- 测试：`PODEM/tests/test_fault_mapping.py`

**接口：**
- 使用：现有的 `ATPG::podem`、`ATPG::fault_sim_a_vector`、故障 catalog 和有序固定型故障配置。
- 产出：`ATPG::get_selectable_fault_ids() const -> vector<string>`、`ATPG::step_stuck_at(const string&) -> AtpgStepResult`、`ATPG::get_stuck_at_result() const -> AtpgRunResult`，以及提供 `catalog()`、`remaining_fault_ids()`、`step(fault_id)`、`result()` 的 Python `cpp_podem.StuckAtSession`。

- [ ] **步骤 1：为增量式会话编写预期失败的绑定测试**

在现有的有序 ATPG 测试旁添加以下测试：

```python
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
```

增加非法调用的拒绝测试：

```python
def test_incremental_session_rejects_non_selectable_fault(self):
    _, binary, fault_map, _, catalog = self.convert(
        "INPUT(a)\nOUTPUT(y)\ny = BUF(a)\n"
    )
    fault_id = str(catalog["faults"][0]["fault_id"])
    session = cpp_podem.StuckAtSession(str(binary), str(fault_map), 5000, 14)
    session.step(fault_id)
    with self.assertRaisesRegex(RuntimeError, "not selectable"):
        session.step(fault_id)
    with self.assertRaisesRegex(RuntimeError, "Unknown fault ID"):
        session.step("missing:GO:sa0")
```

- [ ] **步骤 2：运行新测试，确认因缺少会话类而失败**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -k incremental -v
```

预期结果：FAIL，因为 `cpp_podem.StuckAtSession` 尚不存在。

- [ ] **步骤 3：增加步进累计状态，并重构完整运行流程**

在 `atpg.h` 中新增 `AtpgStepResult`，其中包含所选 ID、`target_status`、`generated_pattern`、`newly_detected_fault_ids`、`remaining_fault_ids` 和累计的 `AtpgRunResult`。新增私有累计计数器，并由 `configure_ordered_stuck_at` 初始化。

在 `atpg.cpp` 中实现以下行为：

```cpp
vector<string> ATPG::get_selectable_fault_ids() const {
    vector<string> result;
    for (fptr fault : flist_undetect) {
        if (!fault->test_tried && fault->detect != REDUNDANT)
            result.push_back(fault_identifier(fault));
    }
    return result;
}
```

`step_stuck_at` 必须精确定位一个当前可选的 ID，调用一次 `podem`，仅在结果为 TRUE 时运行 `fault_sim_a_vector`，将目标标记为已尝试，更新累计计数器，通过比较故障仿真前后的 `flist_undetect` 确定被 drop 的 ID，并返回新的可选故障列表。首次步进时设置 `rand()` 种子，并执行一次“不支持物理 XOR/EQV”的校验。重构 `run_stuck_at`，使其反复调用 `step_stuck_at(get_selectable_fault_ids().front())`；这样旧的完整运行 API 与新会话将共用同一条求解路径。

- [ ] **步骤 4：绑定有状态会话**

在 `python_bindings.cpp` 中增加一个持有底层对象的轻量包装类：

```cpp
class StuckAtSession {
public:
    StuckAtSession(const string &circuit, const string &faultmap,
                   int backtrack_limit, int seed);
    py::dict catalog() const;
    vector<string> remaining_fault_ids() const;
    py::dict step(const string &fault_id);
    py::dict result() const;
private:
    ATPG atpg_;
    vector<ATPG::FaultCatalogEntry> catalog_;
};
```

在电路初始化和每次 `step` 期间释放 GIL，但只能在持有 GIL 时构造 Python 字典。旧接口和会话接口的结果应复用同一个 `summary_to_dict` 辅助函数。

- [ ] **步骤 5：构建并运行 C++／绑定测试**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -v
```

预期结果：所有故障映射、完整有序运行和增量会话测试均 PASS；会话按原生顺序执行得到的结果必须与旧版原生完整排列结果完全一致。

- [ ] **步骤 6：提交增量式求解器边界改动**

```powershell
git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/python_bindings.cpp PODEM/tests/test_fault_mapping.py
git commit -m "feat: expose incremental PODEM fault sessions"
```

---

### 任务 2：动态特征构造与类别策略计算

**涉及文件：**
- 修改：`fault_order_rl/model.py`
- 修改：`fault_order_rl/policy.py`
- 修改：`tests/test_fault_order_rl.py`

**接口：**
- 使用：已校验的 `[N,257]` 冻结 embedding，以及 catalog 行索引。
- 产出：`build_dynamic_features(embeddings, remaining_rows) -> Tensor[K,515]`、`sample_action(scores, temperature, generator=None) -> (local_index, logits)`、`deterministic_action(scores) -> local_index`，以及 `trajectory_log_prob(model, embeddings, decisions, temperature) -> scalar Tensor`。

- [ ] **步骤 1：为 515 维输入编写预期失败的测试**

添加使用可人工核对 tensor 的测试：

```python
def test_dynamic_features_append_remaining_mean_and_ratio():
    embeddings = torch.arange(4 * 257, dtype=torch.float32).reshape(4, 257)
    rows = torch.tensor([1, 3])
    features = build_dynamic_features(embeddings, rows)
    assert features.shape == (2, 515)
    assert torch.equal(features[:, :257], embeddings[rows])
    expected_mean = embeddings[rows].mean(dim=0)
    assert torch.equal(features[:, 257:514], expected_mean.expand(2, -1))
    assert torch.equal(features[:, 514], torch.full((2,), 0.5))
```

同时断言：空的、重复的、越界的或非一维的行索引 tensor 会抛出 `ValueError`；`FaultScorer()` 拒绝 `[N,257]` 输入，但接受 `[N,515]` 输入。

- [ ] **步骤 2：运行特征测试并确认失败**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_features or scorer" -v
```

预期结果：FAIL，因为动态特征构造函数尚不存在，评分器仍然要求 257 维输入。

- [ ] **步骤 3：实现动态评分器输入**

将 `FaultScorer` 的默认输入维度改为 515，同时保留当前的非线性网络结构：

```text
LayerNorm(515) -> Linear(515,256) -> ReLU
-> Linear(256,128) -> ReLU -> Linear(128,1)
```

实现 `build_dynamic_features`，不得修改或克隆源 embedding 矩阵。按 `K / N` 计算剩余比例，并将共享均值和标量扩展到全部 `K` 个候选行。

- [ ] **步骤 4：为带 mask 的类别策略和轨迹重放编写预期失败的测试**

添加固定随机种子的采样测试，以及按公式直接计算的重放测试：

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

- [ ] **步骤 5：实现类别动作辅助函数和轨迹重放**

训练时使用 `torch.multinomial(torch.softmax(logits, 0), 1, generator=generator)` 选择动作。评估时使用稳定的“第一个最大值”选择规则。在 `trajectory_log_prob` 中，校验每个选中行在其保存的剩余行中恰好出现一次；随后为每个状态重新构造动态特征，并累加被选动作对应的 `log_softmax` 项。

- [ ] **步骤 6：运行策略测试**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_features or dynamic_trajectory or categorical or scorer" -v
```

预期结果：所有筛选出的测试均 PASS，并且通过评分器得到的梯度都是有限值。

- [ ] **步骤 7：提交模型和策略计算改动**

```powershell
git add fault_order_rl/model.py fault_order_rl/policy.py tests/test_fault_order_rl.py
git commit -m "feat: add dynamic fault context policy"
```

---

### 任务 3：带状态校验的 Python 会话环境

**涉及文件：**
- 修改：`fault_order_rl/environment.py`
- 修改：`tests/test_fault_order_rl.py`

**接口：**
- 使用：任务 1 产出的 `cpp_podem.StuckAtSession`。
- 产出：`PodemEnvironment.start_session(bench_path, faultmap_path) -> PodemSession`；以及 `PodemSession.remaining_fault_ids() -> tuple[str,...]`、`step(fault_id) -> dict` 和 `result() -> dict`。

- [ ] **步骤 1：为包装层校验编写预期失败的测试**

创建一个假的绑定会话：第一次步进将 `("f0","f1","f2")` 变为 `("f2",)`，并报告 `newly_detected_fault_ids=("f0","f1")`。断言 `PodemEnvironment.start_session` 会传递路径、回溯上限 5000 和随机种子 14；同时断言包装层会拒绝以下情况：剩余 ID 重复或未知、剩余集合变大、TRUE 步骤却没有增加测试向量，以及最终结果缺少任意 `RESULT_FIELDS` 字段。

- [ ] **步骤 2：运行包装层测试并确认失败**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "session_wrapper" -v
```

预期结果：FAIL，因为 `start_session` 和 `PodemSession` 尚不存在。

- [ ] **步骤 3：实现包装层并复用结果校验逻辑**

将当前的最终结果检查提取为 `_validate_result(raw, catalog_size)`。`PodemSession` 保存不可变的初始 ID，以及上一步的剩余故障 tuple。每次步进都要校验：

```text
所选 ID 存在于上一步的剩余集合中
新的剩余 ID 唯一、已知，并且构成严格子集
新检测到的 ID 均为已知 ID，且不在新的剩余集合中
generated_pattern 为真时，累计 pattern_count 恰好增加 1
未生成测试向量的步骤不能改变累计 pattern_count
calls 增加 1，backtracks 永不减少
```

继续支持 `PodemEnvironment.run`，并让其聚合结果字典也经过同一个 `_validate_result` 辅助函数。

- [ ] **步骤 4：运行环境层测试**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "environment or session_wrapper" -v
```

预期结果：所有筛选出的测试均 PASS。

- [ ] **步骤 5：提交 Python 环境边界改动**

```powershell
git add fault_order_rl/environment.py tests/test_fault_order_rl.py
git commit -m "feat: validate incremental PODEM sessions"
```

---

### 任务 4：动态训练回合与按初始故障数归一化

**涉及文件：**
- 修改：`fault_order_rl/trainer.py`
- 修改：`tests/test_fault_order_rl.py`

**接口：**
- 使用：任务 2～3 产出的策略辅助函数和 `PodemEnvironment.start_session`。
- 产出：`Trainer._run_native(circuit)`、`Trainer._run_policy(circuit, model, temperature, stochastic)`、保存的逐步决策状态，以及动态 REINFORCE 更新。

- [ ] **步骤 1：用假的步进会话替换假的完整运行环境**

在 trainer fixture 中，让 `FakeEnvironment.start_session` 针对测试电路的 ID 返回确定性会话。当第一次选择编号为偶数的故障时，还必须同时 drop 下一个可选故障，以便测试能够证明：第二次决策的剩余集合并非只移除了上一步选中的目标。完整的聚合指标仍应由所选行轨迹确定性地产生。

- [ ] **步骤 2：编写预期失败的动态回合测试**

断言：

```python
metrics, elapsed, decisions, trace = trainer._run_policy(
    circuit, trainer.model, temperature=1.0, stochastic=False
)
assert decisions[0]["remaining_rows"] == tuple(range(circuit.fault_count))
assert len(decisions[1]["remaining_rows"]) < circuit.fault_count - 1
assert trace[1]["remaining_ratio"] == (
    len(decisions[1]["remaining_rows"]) / circuit.fault_count
)
```

增加一个损失测试：为包含 5 个故障的电路构造 3 次决策，检查分母严格为 `5`，而不是 `3`。

- [ ] **步骤 3：运行训练器测试并确认失败**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_episode or initial_fault_normalization" -v
```

预期结果：FAIL，因为训练器仍然一次性请求完整排列。

- [ ] **步骤 4：实现原生会话运行器和策略会话运行器**

`_run_native` 反复选择会话返回的第一个 ID。`_run_policy` 将会话 ID 映射回不可变的 catalog 行；每轮重新构造 `[K,515]` 特征，采样或确定性选择一个局部候选，严格执行一次会话步进，并记录：

```python
{"remaining_rows": tuple(rows), "selected_row": selected_row}
```

供人阅读的轨迹另行记录：所选 ID／分数、剩余数量／比例、目标状态、是否生成测试向量、新检测到的 ID，以及累计指标。会话状态中若出现未知 ID、重复 ID、重排后指向未知项，或剩余集合不缩小，必须拒绝。

- [ ] **步骤 5：替换静态回合收集和梯度计算流程**

在每个 batch 内，使用同一个不发生变化的候选模型快照，在 `torch.no_grad()` 下收集所有电路回合。随后针对每个保存的决策列表调用 `trajectory_log_prob`，并严格按下式计算：

```python
loss = (
    -record["advantage"]
    * log_prob
    / circuit.fault_count
    / len(batch)
)
```

对于同一 batch 中的所有回合，在收集与重放之间不得更新模型参数。保持事务式异常处理和随机数生成器状态恢复逻辑不变。

- [ ] **步骤 6：以紧凑格式编码回合轨迹**

每个电路在回合 NPZ 中保存三个数值数组：`<name>__selected_rows`、`<name>__remaining_offsets` 和 `<name>__remaining_rows`。offset 从 0 开始，并以扁平化剩余行数组的长度结束，从而无需 pickle 即可重建每个决策状态。

- [ ] **步骤 7：运行训练恢复和小批次测试**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "shared_update or minibatch or failed_round or dynamic_episode or initial_fault_normalization" -v
```

预期结果：连续运行与断点恢复运行生成逐字节相同的已选行数组和剩余行数组，并得到相同的模型参数与 baseline。

- [ ] **步骤 8：提交动态训练器行为改动**

```powershell
git add fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: train on dynamic fault trajectories"
```

---

### 任务 5：动态确定性评估与 Schema 3 检查点

**涉及文件：**
- 修改：`fault_order_rl/checkpoint.py`
- 修改：`fault_order_rl/trainer.py`
- 修改：`tests/test_fault_order_rl.py`

**接口：**
- 使用：确定性执行的 `_run_policy(..., stochastic=False)`，以及任务 4 产出的动态轨迹。
- 产出：Schema 3 检查点、初始排序诊断 NPZ 文件，以及 `<circuit>.trajectory.jsonl` 执行轨迹。

- [ ] **步骤 1：编写预期失败的检查点迁移测试**

保存最小化的版本 1 和版本 2 字典，并断言二者产生不同的错误：

```python
with pytest.raises(ValueError, match="detected-only coverage"):
    load_checkpoint(schema1)
with pytest.raises(ValueError, match="static 257-dimensional policy"):
    load_checkpoint(schema2)
```

断言 Schema 3 payload 包含 `policy_version == "dynamic_remaining_mean_v1"` 和 `input_dimension == 515`。

- [ ] **步骤 2：升级检查点校验与 payload**

让 `load_checkpoint` 仅接受版本 3。保留 Schema 1 的错误消息，并为 Schema 2 增加明确的“需要重新训练”消息。将上述两个动态策略身份字段加入 latest 和派生 best payload，并在恢复训练／评估的兼容性检查中校验它们。

- [ ] **步骤 3：为动态评估产物编写预期失败的测试**

对每个参与评估的电路，断言其 NPZ 包含：

```text
fault_ids, scores, ranks, permutation,
selected_rows, selected_scores, selected_steps,
exit_steps, exit_reasons
```

将 `scores/ranks/permutation` 视为初始状态诊断信息。对于被另一个选中目标 drop 的故障，断言其 `selected_steps == -1`；同时断言 JSONL 轨迹在同一步中标明执行选择的目标及被 drop 的 ID。

- [ ] **步骤 4：实现动态评估及其输出产物**

将所有静态的 `model(circuit.embeddings)` 加完整顺序运行替换为确定性的动态会话。初始诊断排序只能根据第一个动态状态生成。根据步进轨迹推导 selected 和 exit 数组；退出原因使用 Unicode 字符串，以保证 `np.load(..., allow_pickle=False)` 仍然可用。在每个排序 NPZ 旁以原子方式写入 JSONL。

更新外部 manifest 的可恢复评估逻辑：只有当 metrics JSON、ranking NPZ 和 trajectory JSONL 全部存在，并且具有预期的检查点／评估身份信息时，才将一个电路视为评估完成。

- [ ] **步骤 5：运行检查点和评估测试**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "checkpoint or evaluation or latest or best" -v
```

预期结果：所有筛选出的测试均 PASS；重新进行一次确定性评估时，其结果与已保存的 best 指标完全一致。

- [ ] **步骤 6：提交评估和检查点迁移改动**

```powershell
git add fault_order_rl/checkpoint.py fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: evaluate dynamic fault trajectories"
```

---

### 任务 6：真实求解器集成、文档更新与完整验证

**涉及文件：**
- 修改：`tests/test_fault_order_rl.py`
- 修改：`docs/fault-order-rl.md`

**接口：**
- 使用：已完成的 Schema 3 动态策略，以及重新构建的 `cpp_podem` 扩展。
- 产出：经过验证的真实电路动态训练／评估行为，以及与当前实现一致的用户文档。

- [ ] **步骤 1：增加真实动态流程的冒烟断言**

扩展现有的真实 PODEM 冒烟测试，在其转换得到的微型电路上运行一轮动态训练。断言回合 NPZ 包含已选行和剩余集合 offset；至少有一个相邻步骤的剩余集合发生缩小；确定性评估会同时写出排序产物与轨迹产物，并保持原生 resolved coverage 不变。

- [ ] **步骤 2：运行聚焦的真实求解器冒烟测试**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k real_podem -v
```

预期结果：PASS；测试使用真实的增量式 C++ 会话，并产生有限值的优化器更新。

- [ ] **步骤 3：更新算法与运行文档**

记录以下内容：515 维特征布局、可选剩余故障 mask 的语义、逐步确定性选择和采样选择、轨迹 log probability、按初始故障数归一化、有状态 PODEM 边界、Schema 3 必须重新训练的要求，以及初始排序与实际选择轨迹之间的区别。删除这份受版本控制的运行指南中“模型每个回合生成一个完整排列”的表述。不要修改用户已有的 `README.md` 改动和未跟踪的 `docs/fault-reorder-algorithm.md`。

- [ ] **步骤 4：运行格式和占位内容检查**

运行：

```powershell
git diff --check
rg -n "complete permutation" docs/fault-order-rl.md fault_order_rl PODEM/src
```

预期结果：`git diff --check` 无报错；任何剩余的 `complete permutation` 只能用于描述旧行为或迁移背景。

- [ ] **步骤 5：运行完整的相关测试套件**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py PODEM/tests/test_fault_mapping.py -v
```

预期结果：所有测试均 PASS。

- [ ] **步骤 6：在不训练的情况下校验一个生产 manifest**

运行：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m fault_order_rl validate --manifest configs/anchor_smoke_train.json
```

预期结果：manifest、catalog、embedding 维度、ID、等价故障计数和来源信息全部通过校验。

- [ ] **步骤 7：审查完整改动**

调用 `requesting-code-review` 技能。解决所有正确性问题；先重新运行受影响的最小测试，再运行完整的相关测试套件；最后确认 `git status --short` 中没有构建产物或无关的已暂存改动。

- [ ] **步骤 8：提交已验证的文档与集成测试**

```powershell
git add tests/test_fault_order_rl.py docs/fault-order-rl.md
git commit -m "docs: describe dynamic fault selection"
```

- [ ] **步骤 9：报告验证证据**

报告准确的测试通过命令、动态冒烟测试所用电路、检查点不兼容情况、仍然存在的已知限制，以及设计文档和实施计划的路径。不得依据冒烟运行宣称测试向量数量已经改善；要证明改善必须进行真实训练实验。
