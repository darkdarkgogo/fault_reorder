# BFS 筛选与缓存分数 Ranked-DTC 实施计划

> **供执行者使用：** 必须按任务逐项实施并使用复选框（`- [ ]`）跟踪进度。若采用代理式执行，应使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`。

**目标：** stuck-at DTC 使用原始的 unknown-PO 反向 BFS 发现候选，并采用 TDF 的 `select_fault_try` 预算；RL 只使用 Primary 前向计算得到的缓存分数对每个候选批次排序；每尝试一个 secondary fault 后检查目标 PO；同时用 accepted PI cube 恢复替换逐候选的全 wire 快照。

**架构：** C++ 通过分阶段会话负责 Primary PODEM、canonical good-circuit 状态、BFS 候选资格、候选执行、PO 终止判断和最终 fault simulation。Python 负责协议校验，并提供原样 BFS 顺序或 RL 排序回调。Actor-Critic 每个 Primary step 只运行一次；PPO 使用同一个 score tensor 保存并重放 Primary 选择以及各 BFS 批次的实际执行前缀。

**技术栈：** C++14 ATPG core、pybind11、Python 3、PyTorch、NumPy、pytest、setuptools。

**设计规范：** `docs/superpowers/specs/2026-09-22-bfs-filtered-ranked-dtc-design.md`

## 全局约束

- Primary 回溯上限保持 `100`，DTC secondary 回溯上限保持 `50`，Primary seed 保持 `14`，STC seed 保持 `7`。
- Secondary 候选资格只能由 C++ 的 unknown-PO 反向 BFS 决定。Python 和 RL 可以改变批次内顺序，但不得增加、删除或重复使用候选。
- 当 `ncktin <= 32` 时使用 `select_fault_try = 15`，否则使用 `100`。预算只统计实际出队并展开的 wire，并在切换到每个 unknown PO 时重置。
- Rollout 中每个 Primary step 只能执行一次 Actor-Critic 前向计算；PPO replay 中每个 transition 也只能执行一次。
- 每次 secondary 尝试结束后，必须恢复 accepted fault-free PI cube、重建 canonical good-circuit implication，再判断当前目标 PO 是否仍为 `U`。
- 删除逐候选的 `vector<Snapshot>` 全 wire 保存/恢复路径。持久回滚状态必须为 O(PI 数量)，候选局部日志可以随实际搜索改动量增长。
- 不修改 reward、GAE、PPO 超参数、fault-simulation drop 语义或 STC 行为。
- 提升 checkpoint schema 和 solver protocol identity；不得静默加载 schema 4 或完整 remaining-ranking checkpoint。
- 审查仅限本次改动涉及的接口、动作概率、候选资格和回滚不变量。不得做无关重构，也不运行完整 validation benchmark 套件。

---

### 任务 1：引入原生分阶段协议和固定 solver identity

**涉及文件：**

- 修改：`PODEM/src/atpg.h`
- 修改：`PODEM/src/atpg.cpp`
- 修改：`PODEM/src/python_bindings.cpp`
- 测试：`PODEM/tests/test_fault_mapping.py`

**接口：**

- 用以下接口替换公开的“双参数完整排序”入口：

```cpp
StuckAtPhaseResult begin_stuck_at_step(const string &fault_id);
StuckAtPhaseResult rank_stuck_at_dtc_candidates(
    const vector<string> &ranked_candidate_fault_ids);
AtpgStepResult step_stuck_at(const string &fault_id);
```

- 新增 `StuckAtPhaseResult`。当 `phase == "complete"` 时包含完成的 `AtpgStepResult`；当 `phase == "dtc"` 时包含 `selected_fault_id`、`unknown_po_id`、`dtc_candidate_fault_ids`、`dtc_batch_index`、`select_fault_try` 和 `visited_wire_count`。
- 新增明确的原生阶段状态：`idle` 或 `awaiting_dtc_order`。阶段错误的 begin、finalize 或 continuation 调用必须在修改 solver 状态前被拒绝。
- 扩展 `StuckAtProtocolConfig` 和 `protocol_to_dict()`，加入以下名称、类型和值均固定的字段：

```text
dtc_bfs_small_input_threshold = 32
dtc_bfs_small_select_fault_try = 15
dtc_bfs_default_select_fault_try = 100
dtc_rollback_algorithm = "accepted_pi_cube_resim_v1"
```

**实施步骤：**

- [ ] 在 `PODEM/tests/test_fault_mapping.py` 增加小型会话测试：调用 `begin_step(primary)`，断言结果阶段只能为 `dtc` 或 `complete`；再次调用 `begin_step`、过早调用 `result()` 或在错误阶段 continuation 时必须抛错，且 `remaining_fault_ids()` 和累计计数器不得改变。
- [ ] 增加协议配置断言，检查上述四个 BFS/回滚字段的精确值和 Python 类型。
- [ ] 运行 `python PODEM/setup.py build_ext --inplace` 和 `python -m pytest PODEM/tests/test_fault_mapping.py -q`，确认新增阶段/配置测试因接口尚不存在而失败。
- [ ] 在 `atpg.h` 定义阶段和结果结构，并保存会话状态。未完成 step 的数据保持私有，不通过 pybind 暴露裸指针。
- [ ] 在 `atpg.cpp` 拆分 `ATPG::step_stuck_at`：`begin_stuck_at_step` 只执行并统计一个 Primary。FALSE/MAYBE 和关闭 DTC 的情况立即完成；Primary 成功且开启 DTC 时交给后续候选发现流程。
- [ ] 将单参数 `step_stuck_at` 实现为 heuristic convenience loop：先调用 `begin_stuck_at_step`，只要阶段为 `dtc`，就把返回的 BFS 列表原样提交给 `rank_stuck_at_dtc_candidates`。
- [ ] 在 `python_bindings.cpp` 暴露 `StuckAtSession.begin_step(id)`、`rank_dtc_candidates(ids)` 和单参数 `step(id)`，删除接收全部 remaining secondary IDs 的生产绑定。
- [ ] 使用一个统一 helper 序列化阶段结果：`complete` 返回现有 step 字段；`dtc` 只返回不可变 batch metadata 和日志所需的累计进度字段。
- [ ] 重新编译并运行原生测试模块，确认阶段和配置测试通过后再继续。
- [ ] 提交：`git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/python_bindings.cpp PODEM/tests/test_fault_mapping.py && git commit -m "refactor: add phased stuck-at DTC session"`

---

### 任务 2：实现 unknown-PO 反向 BFS 和 `select_fault_try`

**涉及文件：**

- 修改：`PODEM/src/atpg.h`
- 修改：`PODEM/src/saf_compaction.cpp`
- 修改：`PODEM/src/atpg.cpp`
- 测试：`PODEM/tests/test_fault_mapping.py`

**接口：**

- 新增职责单一的私有 batch finder：

```cpp
bool find_next_stuck_at_dtc_batch(DtcBatchState &batch);
```

- `DtcBatchState` 在内部保存目标 PO 指针，对外只发布稳定 ID 和计数器。
- 候选顺序固定为：`cktout` 顺序 → FIFO 反向 BFS → gate fan-in 顺序 → wire `udflist` 顺序。去重时保留第一次出现的位置。

**实施步骤：**

- [ ] 在 `PODEM/tests/test_fault_mapping.py` 增加聚焦网表 fixture：包含两个 unknown PO、重汇聚的全 `U` cone、已知值 fan-in 分支、cone 外 fault，以及足以越过 15-wire 预算的 buffer 链。
- [ ] 断言只有通过 `U` fan-in 可达的 fault 才会出现；cone 外或已知分支 fault 永不出现；重汇聚不会产生重复 ID；重复运行得到相同 BFS 顺序。
- [ ] 增加小输入 fixture，证明 `visited_wire_count <= 15`、截断点以下的候选不存在，并且下一个 PO 获得新的预算。再增加自动生成的 33-input fixture，证明报告预算为 `100`，但不对运行时间做断言。
- [ ] 运行新增测试，确认它们在阶段接口骨架上先失败。
- [ ] 使用 `std::queue<wptr>` 实现 FIFO 遍历，并维护每个 PO 的 visited-wire 集合、每个 batch 的 candidate-ID 去重集合和每个 Primary 的 attempted-ID 排除集合。
- [ ] 发现候选时与当前 selectable fault catalog 取交集，并排除 Primary、`test_tried`、`REDUNDANT` 以及本 Primary 内已尝试的 fault。
- [ ] 只统计从队列弹出并实际展开的 wire。达到预算后停止更深扩展，但保留处理最后一个允许 wire 时已经发现的候选。
- [ ] 如果某个 PO 没有候选，继续检查下一个稳定 `cktout`。如果所有 PO 都没有候选，只完成一次向量填充和 fault simulation。
- [ ] 确保 `run_stuck_at_ordered()` 和单参数 heuristic session 原样使用 batch 列表；输入的 Primary 顺序绝不能被解释为 secondary 顺序。
- [ ] 重新编译并运行 `python -m pytest PODEM/tests/test_fault_mapping.py -q`，确认候选资格、顺序、去重以及 15/100 预算测试通过。
- [ ] 提交：`git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/saf_compaction.cpp PODEM/tests/test_fault_mapping.py && git commit -m "feat: filter stuck-at DTC with bounded BFS"`

---

### 任务 3：按排序执行 batch、逐 fault 检查 PO，并优化回滚

**涉及文件：**

- 修改：`PODEM/src/atpg.h`
- 修改：`PODEM/src/saf_compaction.cpp`
- 修改：`PODEM/src/atpg.cpp`
- 测试：`PODEM/tests/test_fault_mapping.py`

**接口：**

- 新增职责单一的私有 helper：

```cpp
void restore_stuck_at_good_cube(const vector<int> &accepted_pi_cube);
bool stuck_at_cube_detects(fptr fault);
StuckAtPhaseResult try_stuck_at_dtc_batch(
    const vector<string> &ranked_candidate_fault_ids);
```

- 活跃 step 保存 `accepted_pi_cube`、有序 `preserved_faults`、当前 Primary 的 `attempted_fault_ids`、`embedded_fault_ids`、累计 DTC 计数、当前目标 PO 和当前候选 IDs。

**实施步骤：**

- [ ] 用 batch 局部测试替换旧的“ranked DTC follows supplied prefix”测试：取得原生候选后反向提交，断言 attempted IDs 是反向 batch 的连续前缀，且不包含任何非候选。
- [ ] 增加 fixture，使第一个成功 secondary 将目标 PO 变为已知。即使提交的 permutation 还有尾部，也只能尝试一个 secondary；如果还有下一个 batch，必须基于修改后的 cube 重新 BFS。
- [ ] 增加失败/MAYBE fixture，记录尝试前可观察的 canonical 状态，并验证尝试后 accepted PI、内部 good-circuit 值一致，PO 检查看不到残留 `D`/`D_bar`，fault 仍可选择，而且除了 DTC calls/backtracks 外没有其他累计状态变化。
- [ ] 增加 preserved-fault fixture，证明临时 fault injection 检查结束后，会先恢复 accepted good-circuit cube，再执行 PO 检查和下一次 BFS。
- [ ] 增加非法 permutation 测试：遗漏、重复、额外、未知和过期 ID。所有情况都必须在 attempted IDs、计数器、阶段或 cube 改变前失败。
- [ ] 运行原生聚焦测试，确认新增测试先失败。
- [ ] 实现 `restore_stuck_at_good_cube`：清除 transient scheduled/changed/assignment 状态，从 `accepted_pi_cube` 恢复 good-value PI，把内部 wire 设为 `U`，再执行正常 implication，得到确定的 fault-free 状态。
- [ ] 重构 `stuck_at_podemx_secondary`，使 decision stack 只包含 accepted cube 中原本为 `U` 的 PI；每条退出路径都必须清除 decision/backtrack 状态和 propagate-tree mark。
- [ ] 删除 `run_stuck_at_dtc` 中的 `struct Snapshot` 和 `vector<Snapshot>`，不得用另一份全 wire 保存副本替代。
- [ ] 尝试任何 fault 之前，先验证输入 ID 恰好是当前公开 batch 的一个 permutation。随后按输入顺序执行：搜索前追加 attempted ID，每次只增加一次 calls/backtracks。
- [ ] FALSE/MAYBE 时恢复 accepted PI cube 并重新 implication。TRUE 时验证 Primary 和所有已 embedded fault；每次验证注入结束后恢复 accepted good cube。只有 preserved 检查全部成功后，才提交新的 accepted PI cube。
- [ ] 每个候选尝试完成后，无论 TRUE、FALSE 还是 MAYBE，都恢复 canonical good-circuit 状态并读取当前目标 PO；只要 PO 不再为 `U`，立即停止当前 ranking。
- [ ] Attempted 信息只写入当前 Primary step 集合；secondary 尝试不得设置全局 `test_tried`、`REDUNDANT` 或 `MAYBE`。
- [ ] 重新编译并运行 `python -m pytest PODEM/tests/test_fault_mapping.py -q`，确认回滚、preserved-fault、PO-stop、连续前缀和非法排序测试通过。
- [ ] 运行 `rg -n "Snapshot|vector<Snapshot>" PODEM/src/saf_compaction.cpp`，要求没有匹配项。
- [ ] 提交：`git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/saf_compaction.cpp PODEM/tests/test_fault_mapping.py && git commit -m "feat: stop ranked DTC by PO and restore PI cubes"`

---

### 任务 4：在 Python 中校验并编排 batch 阶段

**涉及文件：**

- 修改：`fault_order_rl/environment.py`
- 测试：`tests/test_fault_order_rl.py`
- 测试：`tests/test_runtime_progress.py`

**接口：**

- 将 Python wrapper 改为：

```python
PodemSession.step(fault_id, rank_dtc_candidates=None) -> dict
```

- 调用回调的方式为 `ranker(candidate_fault_ids, batch_metadata)`，返回该 batch 的完整 permutation。`None` 表示保持 identity/BFS 顺序。
- 最终结果新增 `dtc_batches`；每个元素包含 `unknown_po_id`、`bfs_candidate_fault_ids`、`requested_fault_ids`、`executed_prefix_fault_ids`、`embedded_fault_ids`、`select_fault_try` 和 `visited_wire_count`。

**实施步骤：**

- [ ] 将 `tests/test_fault_order_rl.py` 中的 fake native session 改成确定性的两批次阶段机，分别测试默认 identity ranker 和 reverse ranker，不引入模型。
- [ ] 测试 Python 拒绝返回遗漏、重复、额外、未知或跨 batch ID 的 ranker；验证校验失败后 wrapper 不会调用原生 continuation。
- [ ] 验证原生 `dtc_attempted_fault_ids` 只能按各 batch requested 连续前缀增长；embedded IDs 是有序子序列；不同 batch 不重复 secondary；所有 attempted ID 都属于对应的 BFS candidate set。
- [ ] 将 `tests/test_runtime_progress.py` 中的 fake session 更新为新的单参数/default-ranker 接口，同时保持已有的采样进度断言。
- [ ] 运行 `python -m pytest tests/test_fault_order_rl.py tests/test_runtime_progress.py -q`，确认阶段测试先失败。
- [ ] 在 `environment.py` 增加 `PHASE_FIELDS` 和 batch validator，校验字符串 ID、精确 permutation、阶段转换、稳定 Primary ID、单调累计的原生计数器，以及只在完成后严格缩小的 remaining set。
- [ ] 将 `PodemSession.step` 实现为 begin/continue 循环。只有 `phase == "complete"` 时才提交 session accounting；如果 ranker 抛错，原生会话保持等待同一个 batch，以便调用方修正或显式终止。
- [ ] ID 和 batch sequence 使用不可变 tuple 返回，避免 trainer 在 step 结束后修改协议审计证据。
- [ ] 将任务 1 的四个原生字段及其精确值/类型加入 `PROTOCOL_CONFIG`。
- [ ] 运行两个聚焦 Python 测试模块并确认通过。
- [ ] 提交：`git add fault_order_rl/environment.py tests/test_fault_order_rl.py tests/test_runtime_progress.py && git commit -m "refactor: orchestrate BFS DTC batches in Python"`

---

### 任务 5：拆分 Primary 选择和基于缓存分数的候选排序

**涉及文件：**

- 修改：`fault_order_rl/policy.py`
- 修改：`fault_order_rl/__init__.py`
- 测试：`tests/test_fault_order_rl.py`

**接口：**

- 用以下函数替换完整 remaining-ranking helper：

```python
sample_primary(scores, remaining_rows, temperature, stochastic, generator=None)
sample_candidate_ranking(scores, remaining_rows, candidate_rows,
                         temperature, stochastic, generator=None)
joint_action_stats(model, embeddings, remaining_rows, primary_row,
                   dtc_batches, temperature)
```

- `scores[i]` 对应 `remaining_rows[i]`。候选排序只能索引这个 tensor，不能再次调用模型。
- 每个 replay batch 提供 `bfs_candidate_rows` 和 `executed_prefix_rows`；requested 但未执行的尾部不计入 log-probability 或 entropy。

**实施步骤：**

- [ ] 用手算概率测试替换现有 full-ranking policy 测试：三个 fault 的 Primary categorical 概率，加上两个 DTC executed-prefix 的 Plackett–Luce 项。断言 `joint_action_stats` 等于手算总和，只返回一个 critic value，并对全部真实执行的条件选择取 entropy 均值。
- [ ] 增加计数 Actor-Critic 测试，证明即使存在多个 DTC batch，`joint_action_stats` 也只调用一次 `forward`。
- [ ] 增加候选排序测试：非候选的高分必须被忽略；随机输出恰好是 candidate permutation；确定性相同分数按 catalog row 升序打破平局。
- [ ] 增加畸形 replay 测试：候选 rows 重复/遗漏、executed row 不在 batch、非前缀执行证据、不同 batch 重复 secondary row、Primary 不属于 remaining rows，以及非有限 score/value/statistics。
- [ ] 运行 `python -m pytest tests/test_fault_order_rl.py -q`，确认新增 policy 测试先失败。
- [ ] 实现基于显式 local indices 的共享 conditional prefix-stat helper。Primary categorical 项和每个 batch prefix 都复用它，且不重建 features。
- [ ] `sample_primary` 只执行一次 categorical sample/argmax。`sample_candidate_ranking` 先把 candidate catalog rows 映射到 `remaining_rows` 中的唯一位置，再只在该子集内执行 Plackett–Luce sampling 或稳定 score sort。
- [ ] `joint_action_stats` 只调用一次 `build_dynamic_features` 和一次模型 forward；log probability 为 Primary 与各 batch项之和，entropy 对 Primary 选择和每个实际执行的 secondary 选择取均值。
- [ ] 所有调用方迁移后，从生产导出中删除 `sample_ranking` 和 `executed_prefix_stats`；不得保留可对全部 remaining secondary 排序的兼容路径。
- [ ] 运行聚焦 policy 测试，并确认 Actor 和 Critic 参数梯度均为有限值。
- [ ] 提交：`git add fault_order_rl/policy.py fault_order_rl/__init__.py tests/test_fault_order_rl.py && git commit -m "feat: score Primary and BFS DTC actions jointly"`

---

### 任务 6：Rollout 与 PPO replay 各自只使用一次缓存前向计算

**涉及文件：**

- 修改：`fault_order_rl/trainer.py`
- 测试：`tests/test_fault_order_rl.py`
- 测试：`tests/test_runtime_progress.py`

**接口：**

- 每个 rollout decision 保存：

```text
remaining_rows
primary_row
old_joint_log_probability
old_value
mean_entropy
dtc_batches[]:
  unknown_po_id
  bfs_candidate_rows
  requested_rows
  executed_prefix_rows
  embedded_rows
  select_fault_try
  visited_wire_count
```

- Native baseline 调用 `session.step(primary_id)`，不得构造模型分数。
- RL rollout 只计算一次 `scores, value`，采样/选择 Primary，然后把捕获该 `scores` 的 batch ranker 传给 `session.step`。

**实施步骤：**

- [ ] 增加 counting-model rollout 测试：fake native 提供两个 DTC batch，整个 Primary step 只能有一次 forward；两个 requested batch 顺序都必须来自同一个捕获的 score tensor。
- [ ] 增加 PPO replay 测试：每个存储 transition 只能 forward 一次，而不是每个 DTC batch 一次；optimizer 更新前，rollout 和 replay joint log probability 必须相等。
- [ ] 更新 trace 断言：每个 requested row 恰好覆盖对应 BFS candidate rows；每个 executed list 是 requested prefix；embedded rows 是有序 executed 子序列；同一 Primary step 不重复 secondary row。
- [ ] 增加确定性评估测试，覆盖 Primary/candidate 同分时按 catalog row 打破平局。
- [ ] 运行 `python -m pytest tests/test_fault_order_rl.py tests/test_runtime_progress.py -q`，确认 trainer 测试先失败。
- [ ] 在 `_run_policy` 中构造动态特征并只调用模型一次；只选择 `primary_row`；建立 ID-to-row 映射，并创建 ranker closure，使用缓存 `scores` 对每个原生 batch 调用 `sample_candidate_ranking`。
- [ ] 将返回的 batch IDs 转回 rows，验证它们都属于 step 开始时的 `remaining_rows - {primary_row}`，并直接使用已经存在的 `scores`/`value` 计算 rollout statistics，不得再次 forward。
- [ ] 用 `primary_row` 和嵌套 `dtc_batches` 替换 step-level 的 `requested_rows`/`executed_rows`。同步更新 `_evaluation_export`、日志和 best-result metadata。
- [ ] 在 `_ppo_update` 中每个 transition 只调用一次 `joint_action_stats`，保留现有 clipping、value loss、entropy coefficient、gradient clipping、GAE 和 minibatch 行为。
- [ ] `_run_native` 使用单参数 session 路径，使 heuristic DTC 原样提交每个 BFS batch，且不实例化或调用模型。
- [ ] 运行两个聚焦测试模块，确认 rollout、replay、确定性评估和进度测试全部通过。
- [ ] 提交：`git add fault_order_rl/trainer.py tests/test_fault_order_rl.py tests/test_runtime_progress.py && git commit -m "feat: reuse Primary scores for ranked DTC"`

---

### 任务 7：升级 checkpoint 并序列化嵌套 DTC 审计数据

**涉及文件：**

- 修改：`fault_order_rl/checkpoint.py`
- 修改：`fault_order_rl/trainer.py`
- 修改：`fault_order_rl/__main__.py`
- 测试：`tests/test_fault_order_rl.py`

**接口：**

- Checkpoint schema 改为 `5`。
- `POLICY_IDENTITY = "dynamic_bfs_ranked_dtc_actor_critic_ppo_v2"`。
- `SOLVER_PROTOCOL["compression_algorithm_version"] = "stuck_at_podemx_bfs_ranked_dtc_v3"`，并包含全部固定 BFS/回滚配置字段。
- 使用两层 offset，把嵌套 trajectory 展平成非 object NumPy arrays：

```text
primary_rows
dtc_batch_step_offsets
dtc_candidate_rows / dtc_candidate_offsets
dtc_requested_rows / dtc_requested_offsets
dtc_executed_rows / dtc_executed_offsets
dtc_embedded_rows / dtc_embedded_offsets
dtc_select_fault_try
dtc_visited_wire_count
```

**实施步骤：**

- [ ] 更新 checkpoint 测试：schema 1–4 都必须给出明确的“需要重新训练”错误；schema 4 的错误需要指出旧的完整 remaining Ranked-DTC 动作空间已失效。
- [ ] 增加 NPZ round-trip 测试：连续 Primary steps 分别包含零个、一个和多个 DTC batch。使用 `allow_pickle=False` 加载，并通过 step/batch offsets 精确重建嵌套 rows。
- [ ] 增加 resume validation 测试：schema 5、policy identity、solver version、BFS 阈值/预算和 rollback identity 必须全部匹配，之后才能恢复 optimizer/RNG 状态。
- [ ] 运行 `python -m pytest tests/test_fault_order_rl.py -q`，确认新增 schema/serialization 测试先失败。
- [ ] 将 checkpoint 创建和加载升级为 schema 5，增加明确的 schema 4 拒绝信息，并更新 CLI/help 中提到的 checkpoint schema。
- [ ] 更新 `_write_circuit`，输出有类型的 `int64` flat arrays 和单调 offsets。不得使用 object arrays、pickle 嵌套数据，也不得遗漏空 batch。
- [ ] 更新 policy/solver identity，确保 resume 和 final-model load 时对保存的 `protocol` 做精确 mapping 比较。
- [ ] 运行聚焦测试，并使用 `np.load(path, allow_pickle=False)` 检查一个实际生成的 NPZ。
- [ ] 提交：`git add fault_order_rl/checkpoint.py fault_order_rl/trainer.py fault_order_rl/__main__.py tests/test_fault_order_rl.py && git commit -m "feat: version BFS-ranked DTC trajectories"`

---

### 任务 8：更新说明并只执行必要的集成检查

**涉及文件：**

- 修改：`docs/fault-order-rl.md`
- 修改：`docs/fault-reorder-algorithm.md`
- 仅当聚焦测试发现缺陷时修改：任务 1–7 中已涉及的文件

**接口要求：**

- 文档必须明确区分：Primary 从全部 selectable faults 中选择；secondary 只在每个 BFS-filtered batch 内排序。
- 文档必须说明：每次 secondary 尝试后恢复 canonical good-circuit 状态，并检查目标 PO 是否仍为 `U`。

**实施步骤：**

- [ ] 更新两份文档，写明分阶段 API、15/100 `select_fault_try` 规则、逐候选 PO-stop 规则、缓存分数规则、joint probability、schema 5 和 accepted-PI-cube 回滚。删除“DTC 接收全部 remaining faults”的描述。
- [ ] 重新编译扩展：`python PODEM/setup.py build_ext --inplace`。
- [ ] 运行原生/binding 回归：`python -m pytest PODEM/tests/test_fault_mapping.py -q`。
- [ ] 运行 policy/trainer/progress 回归：`python -m pytest tests/test_fault_order_rl.py tests/test_runtime_progress.py -q`。
- [ ] 使用 `validation/bench/b12.ckt` 做一次小型真实集成测试，开启 DTC、关闭 STC。记录 candidate count、attempted prefix count、DTC calls、patterns、coverage 和 backtracks；验证每个 attempted ID 都属于日志中的 BFS batch，且 batch 中没有 cone 外 fault。不对 wall-clock 做断言，也不运行完整 validation 套件。
- [ ] 仅为核对统计一致性，将该电路与关闭 DTC 的执行比较：两个会话都能结束、计数器内部一致、DTC 不会伪造 detected IDs。不要求 pattern count 或 coverage 轨迹完全相同。
- [ ] 运行 `git diff --check` 和 `git status --short`。只检查本功能修改的文件；如果构建产物未被跟踪，不得把它们提交。
- [ ] 确认生产路径没有旧接口残留：`rg -n "sample_ranking|executed_prefix_stats|ranked_secondary_fault_ids|vector<Snapshot>" PODEM/src fault_order_rl tests` 只能匹配刻意保留的迁移错误测试，不得匹配运行时代码。
- [ ] 提交：`git add docs/fault-order-rl.md docs/fault-reorder-algorithm.md && git commit -m "docs: describe BFS-filtered ranked DTC"`

## 完成标准

- Unknown-PO 反向 BFS 是 secondary 候选的唯一来源；顺序稳定；遵守每个 PO 的 15/100 wire 预算；当前全 `U` cone 外的 fault 不会进入候选集。
- Heuristic 保持 BFS 顺序；RL 只使用 Primary 单次 forward 缓存的分数改变各 batch 内部顺序。
- 每次尝试 secondary 后都恢复 canonical good-circuit 状态并检查目标 PO；PO 已知时，requested ranking 被截断为已记录的 executed prefix。
- 失败或拒绝的候选除规定的 call/backtrack 统计外不留下 solver 状态；代码中不再存在逐候选的全 wire snapshot。
- PPO rollout/replay 概率只覆盖 Primary 和真实执行的 DTC prefixes，每个 transition 只执行一次 Actor-Critic forward。
- Schema 4 checkpoint 被拒绝；schema 5 的嵌套 trajectory 数据可以在禁用 pickle 时完整 round-trip；solver identity 固定 BFS 与 rollback 协议。
- 三个聚焦测试模块和一次 b12（DTC 开、STC 关）集成测试通过；不需要进行无关的全套测试或宽泛审查。
