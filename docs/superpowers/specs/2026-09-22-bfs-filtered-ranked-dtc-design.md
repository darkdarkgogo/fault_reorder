# BFS 筛选与缓存分数 Ranked-DTC 设计规范

## 状态与效力

本规范记录 2026-09-22 确认的 stuck-at DTC 候选筛选与 RL 排序语义。

本规范只替换以下旧行为：

- `2026-09-15-dynamic-fault-selection-design.md` 中“扫描其他全部仍可选 secondary
  fault”的定义；
- `2026-09-20-dynamic-ppo-strict-design.md` 中“Primary 与其余全部 selectable faults
  组成完整 permutation”的定义；
- 当前实现把全部 non-primary remaining faults 传给 DTC 并逐个运行 PODEMX 的行为。

Primary PODEM、PODEMX 回溯上限、fault simulation、STC、coverage guard、reward、PPO
超参数和 checkpoint 提交边界等未在本文中明确修改的协议继续保持不变。

## 目标

恢复原始 DTC 的结构性候选筛选：只从当前值为 `U`（即 X）的 primary output 向前
反向 BFS，在未知逻辑锥中发现可能约束该输出的 secondary faults。候选资格完全由
C++ ATPG 当前 test cube 和 BFS 决定，RL 不得扩大候选集合。

两种执行模式共享同一个候选发现过程：

- heuristic/native baseline 按 BFS 返回顺序尝试；
- RL 复用本次 Primary 选择时已经计算的 fault scores，只对 BFS 返回的候选 ID
  排序，不再次运行模型，也不为全部 remaining faults 构造 DTC permutation。

该修改的主要结果是把 DTC secondary 搜索从“全部 remaining faults”收缩到当前未知
PO 逻辑锥内的候选，避免大电路中无意义的二次 PODEM 调用。

## 已确认的核心语义

每个 primary step 只执行一次模型前向计算：

```text
remaining fault set F_t
  -> build_dynamic_features(F_t)
  -> Actor-Critic forward once
  -> cached score map {fault_id: score}
  -> select Primary
  -> Primary PODEM
  -> BFS candidate set C_t,1
  -> use cached scores to order C_t,1
  -> possibly BFS candidate set C_t,2
  -> use the same cached scores to order C_t,2
  -> ...
  -> finalize vector and fault simulation
```

“复用分数”具有以下严格含义：

- 所有 DTC batch 使用 Primary 选择时生成的同一份 score tensor；
- DTC 期间 test cube、未知 PO 或候选集合改变时，不重新构造特征、不重新调用模型；
- score 的动态上下文仍是 step 开始时的完整 `F_t`，不是某个 BFS 子集；
- 候选子集只充当 action mask 和排序范围；
- Critic value 同样只在 step 开始时计算一次。

## BFS 候选资格

### Unknown PO 选择

Primary PODEM 返回 TRUE 后，不随机填充仍为 `U` 的 PI。C++ 按 `cktout` 的稳定网表
顺序查找当前值为 `U` 的 PO。每次只为一个 unknown PO 建立候选 batch。

如果没有 unknown PO，DTC 立即结束并进入向量填充与 fault simulation。

### 反向遍历

候选发现从当前 unknown PO 开始，使用 FIFO 队列执行反向 BFS：

1. 出队一个值为 `U` 的 wire；
2. 按该 gate 的输入顺序，把值仍为 `U` 的 fan-in wires 入队；
3. 按 wire `udflist` 的稳定顺序检查 fault；
4. 对 reconvergent 路径使用 visited-wire 集合，避免重复遍历；
5. 对 fault ID 去重，第一次出现的位置决定 BFS 顺序。

一个 fault 只有同时满足下列条件时才进入当前 batch：

- 属于当前会话的 undetected/selectable fault catalog；
- 不是本步 Primary；
- 尚未被标记为 `test_tried`；
- 当前状态不是 `REDUNDANT`；
- 本次 Primary 的 DTC 尚未尝试过该 fault；
- 该 fault 从当前 unknown PO 的全 `U` 反向逻辑锥可达。

这里的“候选”表示结构上允许尝试，不表示 PODEMX 一定成功。可检测性仍由真实
`stuck_at_podemx_secondary()` 搜索决定。

### Batch 生命周期

C++ 把当前 unknown PO、BFS 顺序和完整候选 ID 列表暴露给调用方。调用方必须返回
该列表的一个完整 permutation，不能增加、遗漏或重复 ID。

C++ 按返回顺序尝试 secondary fault：

- 成功时保留新增 PI 约束，并继续保证 Primary 和先前已接受 secondary 可检测；
- 失败或达到回溯上限时完整回滚本次尝试；
- 无论成功或失败，已执行的 fault 都加入本 Primary 的 attempted set；
- 当前 unknown PO 变为已知后，立即停止当前 ranking，未执行尾部不计入动作；
- ranking 耗尽而 PO 仍未知时，跳过该 PO 并继续查找下一个 unknown PO；
- test cube 改变后，下一批候选必须重新执行 BFS，不得复用旧候选集合。

同一 fault 在一个 Primary DTC 内最多尝试一次。DTC 在没有 unknown PO，或所有仍未知
PO 都没有未尝试候选时结束。该终止规则取代依赖“遍历全部 remaining faults”才能结束
的旧逻辑。

## Heuristic/native baseline

Native baseline 的 Primary 选择保持为当前 catalog 中最早的 selectable fault。Primary
成功后的每个 DTC batch 直接按 C++ 返回的 BFS 顺序提交，不做额外排序。

因此 heuristic 的可复现顺序由以下稳定顺序共同决定：

```text
cktout order
-> FIFO reverse BFS
-> gate fan-in order
-> wire udflist order
-> first occurrence after de-duplication
```

Native baseline 和 RL 必须共享候选发现、PODEMX、回滚、preserved-fault 检查、随机填充
和 fault simulation 实现；两者唯一差别是候选 batch 的顺序来源。

## RL Primary 选择与 DTC 排序

### 一次前向计算

在 step 开始时，对完整 selectable set `F_t` 构造现有 515 维动态特征，并调用模型一次：

```text
scores_t, value_t = model(build_dynamic_features(embeddings, F_t))
```

`scores_t` 与 `F_t` 的 catalog rows 建立稳定映射，并在整个 Primary/DTC step 中缓存。
如果 C++ 返回不属于 `F_t - {primary}` 的候选 ID，Python 必须以 protocol error 终止。

### Primary action

训练时，Primary 从 `F_t` 的 temperature-scaled categorical 分布采样。确定性训练评估和
验证按 score 降序选择；score 相同时按 catalog row 升序。

不再为了取得 Primary 而预先生成全部 remaining faults 的完整 permutation。

### Secondary batch action

对于 C++ 返回的 BFS candidate batch `C_t,b`：

- Python 从缓存的 `scores_t` 中索引候选分数；
- 训练 rollout 只在 `C_t,b` 内按 Plackett-Luce 分布采样 permutation；
- 确定性评估只在 `C_t,b` 内按缓存 score 降序排列，平分时按 catalog row 升序；
- 未进入 `C_t,b` 的 remaining faults 不参与该 batch 的归一化、采样或排序；
- 后续 batch 继续复用同一 `scores_t`，不再次调用模型。

一个 batch 的真实动作只包含 C++ 实际尝试的 requested ranking 前缀。当前 PO 提前被
填充时，未执行尾部既不计入 log-probability，也不计入 entropy。

### 联合动作概率

一个 primary step 的 joint log-probability 为：

```text
log P(step action)
  = log P(primary | F_t)
  + sum_b log P(executed DTC prefix_b | C_t,b, cached scores_t)
```

Primary categorical 和各 DTC batch 的每次条件选择 entropy 合并后取均值，避免候选 batch
数量或执行前缀长度改变 entropy bonus 的天然尺度。

PPO replay 必须：

1. 使用保存的 `F_t` rows 重建特征；
2. 对模型只执行一次 forward；
3. 使用保存的 Primary row、每个 BFS candidate rows 和 executed prefix rows，从同一
   score tensor 重建 joint log-probability 与 mean entropy；
4. 使用该次 forward 得到的唯一 value 计算 critic loss。

Step reward、terminal correction、GAE 和 PPO ratio 仍以一个完整 primary step 为一个
transition。DTC batch 不拆成独立 reward transition。

## Stateful native 接口

现有一步完成 Primary 与全量 DTC 的接口：

```python
session.step(primary_fault_id, ranked_all_remaining_fault_ids)
```

不能在 Primary PODEM 后暴露由 test cube 决定的 BFS 候选，因此替换为显式状态机：

```python
state = session.begin_step(primary_fault_id)

while state["phase"] == "dtc":
    candidate_ids = state["dtc_candidate_fault_ids"]
    ranked_ids = rank_only_this_batch(candidate_ids)
    state = session.rank_dtc_candidates(ranked_ids)

step_result = state  # phase == "complete"
```

### `begin_step(primary_fault_id)`

- 只允许在 session idle 状态调用；
- 验证 Primary 当前 selectable；
- 执行一次 Primary PODEM；
- FALSE/MAYBE 时完成 step 并返回 `phase="complete"`；
- TRUE 且 DTC disabled 时直接填充、fault-sim 并完成 step；
- TRUE 且存在 BFS 候选时保存未完成的 test cube，返回 `phase="dtc"`；
- TRUE 但不存在 BFS 候选时直接完成向量生成和 fault simulation。

`phase="dtc"` 至少返回：

```text
selected_fault_id
unknown_po_id
dtc_candidate_fault_ids   # 去重后的稳定 BFS 顺序
dtc_batch_index
```

### `rank_dtc_candidates(ranked_candidate_fault_ids)`

- 只允许在等待当前 DTC batch 排名时调用；
- 输入必须恰好是当前暴露候选列表的 permutation；
- 验证失败不得改变 test cube、attempted set、metrics 或 phase，调用方可以修正后重试；
- 按输入顺序执行，直到当前 PO 已知或 ranking 耗尽；
- 自动寻找下一个可用 BFS batch；有新 batch 时返回 `phase="dtc"`；
- 没有后续 batch 时填充未知 PI、执行一次 fault simulation 并返回
  `phase="complete"` 的完整 step result。

Session 在 `phase="dtc"` 时拒绝新的 `begin_step()`、episode finalize 和 STC。这样不能把
半完成 test cube 当作已提交状态。

### 兼容入口

单参数 heuristic `step(primary_fault_id)` 可以保留为 convenience API，但内部必须通过
同一状态机，并对每个 batch 原样提交 BFS 顺序。旧的“两参数 + 全部 remaining secondary
permutation”生产接口删除，防止再次绕过 BFS eligibility。

`run_stuck_at_ordered()` 继续把输入顺序解释为 Primary 顺序；其内部 DTC 使用 heuristic
BFS 顺序，不再把 Primary 排列的尾部当作 secondary 候选。

## 返回值与轨迹

完成后的 step 继续返回现有累计字段，并保留：

```text
dtc_attempted_fault_ids
dtc_embedded_fault_ids
current_dtc_secondary_calls
current_primary_backtracks
current_dtc_backtracks
current_total_backtracks
generated_test_vector
newly_detected_fault_ids
remaining_fault_ids
```

`dtc_attempted_fault_ids` 是所有 batch 实际执行前缀按时间拼接的结果；
`dtc_embedded_fault_ids` 是其保序子序列。

训练 trajectory 中每个 primary step 新增：

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
```

审计必须能够证明：

- 每个 requested ranking 恰好覆盖对应 BFS candidate set；
- executed rows 是 requested rows 的连续前缀；
- batch 之外的 remaining fault 从未作为本 batch secondary 执行；
- 同一 Primary 内没有 secondary 被重复尝试；
- 一个 transition 的所有 batch 都来自同一次缓存 score forward。

## Fault 状态和回滚

候选资格、PODEMX 成功与最终 fault dropping 是三个不同概念：

- BFS candidate 只表示结构上允许尝试；
- embedded 表示 PODEMX 成功增加约束且 preserved faults 仍可检测；
- detected/drop 只由最终完整向量的 fault simulation 决定。

Secondary 尝试不能设置全局 `test_tried`、`REDUNDANT` 或 `MAYBE`。本 Primary 内的
attempted 去重使用独立 session-step 集合；step 完成后销毁。失败尝试必须恢复全部 wire
value、assigned、changed 和 scheduled 状态。成功尝试保留 cube 约束，但不能直接从
remaining set 删除 fault。

## Checkpoint 与协议版本

新 action mask、joint probability 和 trajectory schema 与现有“完整 remaining ranking”
checkpoint 不兼容。实现必须：

- 提升 checkpoint schema；
- 把 solver protocol identity 改为新的 BFS-filtered ranked-DTC 版本；
- 拒绝恢复旧 schema 或旧 solver protocol 的 optimizer、baseline、EMA 与 RNG 状态；
- 要求从头训练，不能静默迁移旧 PPO 轨迹或 checkpoint。

确定性 tie-break、PyTorch RNG、Python RNG、NumPy RNG 和 optimizer 状态继续进入 checkpoint
可复现性边界。

## 日志与指标

运行日志按 batch 输出采样后的进度，不逐 fault 淹没 stderr。至少记录：

```text
Primary ID
unknown PO ID
BFS candidate count
attempted prefix count
embedded count
DTC backtracks
batch elapsed seconds
```

现有累计指标语义保持：

```text
primary_podem_calls
dtc_secondary_calls          # 只统计 BFS 合法且实际尝试的 secondary
primary_backtracks
dtc_backtracks
total_backtracks = primary_backtracks + dtc_backtracks
```

不得以 wall-clock 时间作为单元测试断言。性能回归通过候选和调用次数证明：构造一个包含
大量 remaining faults、但只有少量 fault 位于 unknown-PO `U` cone 的 fixture，确认
`dtc_secondary_calls` 只等于实际执行的 BFS 前缀长度。

## 错误处理

以下情况属于 protocol error，立即失败，不允许回退到全量候选或 catalog order：

- unknown、重复、遗漏或额外的 candidate ID；
- candidate 不属于 step 开始时的 remaining set；
- 在错误 session phase 调用接口；
- C++ 返回的 attempted IDs 不是 requested ranking 的连续前缀；
- 同一 Primary 内重复尝试 secondary；
- Primary/DTC 完成前调用 finalize；
- replay 保存的 candidate mask、requested order 或 executed prefix 不一致；
- 非有限缓存 score、value、log-probability 或 entropy。

## 测试要求

### C++ 与 binding

- unknown PO 反向 BFS 只遍历 `U` fan-in cone；cone 外 fault 不返回、不尝试；
- BFS 顺序对固定网表稳定，并正确处理 reconvergence 与重复 `udflist` fault；
- heuristic 按 BFS 顺序执行；反向 RL ranking 只改变候选内部顺序；
- 当前 PO 被填充后停止当前 ranking，attempted IDs 是连续前缀；
- 下一 unknown PO 使用修改后的 cube 重新 BFS；
- 同一 fault 在一个 Primary 内不重复尝试；
- candidate 失败完整回滚，成功继续保持 Primary 与已接受 secondary；
- 无候选、DTC disabled、Primary FALSE 和 Primary MAYBE 均正确完成 step；
- 非法 permutation 和错误 phase 在任何状态修改前失败；
- legacy heuristic convenience path 与显式 BFS-order state machine 结果一致。

### Python、策略与 PPO

- 一个包含多个 DTC batch 的 primary step 只调用一次 model forward；
- 后续 batch 从同一 score tensor 索引候选分数；
- non-candidate remaining faults 不进入 batch softmax 或排序；
- stochastic batch ranking 只包含候选，deterministic tie-break 使用 catalog row；
- 手算 Primary categorical 加多个 executed-prefix Plackett-Luce 的 joint log-probability；
- PPO replay 每个 transition 只 forward 一次，并精确复现 rollout 概率与 value；
- trajectory 完整保存 BFS masks、requested orders 和 executed prefixes；
- heuristic baseline 不调用模型且保持 BFS order；
- 旧 checkpoint schema 和旧 solver protocol 被明确拒绝。

### 真实集成

至少一个小型真实 BENCH 电路完成以下两条路径：

1. native Primary order + BFS-order DTC；
2. RL Primary selection + cached-score candidate-only DTC ordering。

两条路径都必须完成 fault dropping 与可选 STC，验证 coverage/accounting 不变量。集成测试
另外断言 DTC attempted IDs 全部来自记录的 BFS batch，不能只检查最终 pattern count。

## 验收标准

1. DTC 不再遍历全部 non-primary remaining faults。
2. 每个 attempted secondary 都来自当时 unknown PO 的反向 `U`-cone BFS 候选集合。
3. Heuristic 严格使用稳定 BFS 顺序。
4. RL 每个 primary step 只执行一次模型 forward。
5. RL 的每个 DTC batch 只复用该次 Primary scores 进行候选内排序。
6. test cube 改变或进入下一 unknown PO 后重新 BFS，但不重新打分。
7. PPO joint probability 只包含 Primary 和真实执行的各 batch 前缀。
8. 旧 full-remaining-ranking checkpoint 不能恢复。
9. coverage、pattern、fault status、calls 和 backtracks 指标保持可审计且一致。
10. 单元测试、Python 测试和至少一个真实 binding 集成测试全部通过。
