# BFS 筛选动态故障重排算法

当前实现是 coverage-first 的 Actor-Critic PPO。完整约束见
[`2026-09-22-bfs-filtered-ranked-dtc-design.md`](superpowers/specs/2026-09-22-bfs-filtered-ranked-dtc-design.md)，
其 heuristic lazy 修订见
[`2026-09-23-tdf-lazy-baseline-design.md`](superpowers/specs/2026-09-23-tdf-lazy-baseline-design.md)，
运行命令见 [`fault-order-rl.md`](fault-order-rl.md)。

## 1. 动态状态与 Primary 选择

每个 fault 有固定的 257 维 embedding。对当前 remaining set `F_t` 中的候选 `i`，模型
输入为：

```text
[fault_embedding_i, mean_embedding(F_t), |F_t| / |F_0|]
```

总维度为 515。共享 encoder 输出每个 remaining fault 的 actor score，以及当前集合唯一
的 critic value。每个 Primary step 只执行一次模型 forward。训练时从完整 `F_t` 的
categorical 分布选择 Primary；确定性评估按 score 降序选择，同分按 catalog row 升序。

## 2. RL Ranked-DTC 与 heuristic lazy DTC

Primary 成功后，RL 路径由 C++ 在 canonical good-circuit cube 上按稳定 `cktout` 顺序
查找 unknown PO，并从该 PO 沿值为 `U` 的 fan-in 执行 FIFO 反向 BFS：

```text
cktout order
-> FIFO reverse BFS
-> gate fan-in order
-> wire udflist order
-> first occurrence after de-duplication
```

只有当前 selectable、非 Primary、非 redundant、未在本 Primary 中尝试，且位于该全
`U` cone 的 fault 才能进入 batch。每个 PO 的 BFS wire 展开预算为：

```text
ncktin <= 32 : select_fault_try = 15
ncktin > 32  : select_fault_try = 100
```

RL 在预算内先收集完整 batch，再从 Primary 的缓存 score tensor 中索引当前 candidate
rows 并排序；不能增加、遗漏或重复 ID，也不能重新调用模型。

Heuristic/native baseline 不建立完整 batch，而是严格使用原 TDF lazy 调度：维护
`q_wire` 和 `q_fault`，仅当 `q_fault` 为空时才从 `q_wire` 展开下一个 `U` wire；一旦
该 wire 的 `udflist` 产生 eligible faults，就立即逐个 DTC。只有这些 faults 全部耗尽且
目标 PO 仍为 `U` 时，才继续反向 BFS。`StuckAtSession.step()` 与
`run_stuck_at_ordered()` 共用这条 lazy 路径。

## 3. 执行前缀、PO 检查与回滚

C++ 的两条路径共用单个 secondary 的执行操作。每完成一个 secondary fault，无论
PODEMX 返回 TRUE、FALSE 或 MAYBE，都恢复 accepted fault-free PI cube、重新
implication，再检查目标 PO。如果 PO 已知，baseline 立即丢弃 lazy 队列尾部，RL 则
立即丢弃 ranking 尾部；因此 RL 实际动作只包含 requested order 的连续前缀。

成功 secondary 只有在 proposed PI cube 通过单调性 invariant 后才提交：旧 cube 中已经
确定的 `0/1` PI 必须保持不变，仅允许原来的 `U` 保持 `U` 或细化为 `0/1`。在五值仿真中，
已经到达 PO 的确定 `D`/`D_bar` 不会被这种细化破坏，因此生产路径不再重放 Primary 和历史
accepted secondary。失败或达到回溯上限时不提交 proposed cube，并恢复旧 accepted
good-circuit cube、重新 implication；invariant 违反则立即报错，因为它表示 PODEMX 实现
破坏了固定 PI。这样也能避免临时 `D`/`D_bar` 影响 PO 判断或下一次 BFS。

测试 cube 改变后重新执行 BFS，但仍复用同一个 Primary score tensor。同一 Primary 内
一个 secondary 最多实际尝试一次。

## 4. 联合动作概率

一个 Primary step 的概率为：

```text
log P(action_t)
  = log P(primary_t | F_t)
  + sum_b log P(executed_prefix_t,b | C_t,b, cached_scores_t)
```

未进入 `C_t,b` 的 remaining faults 不参与该 batch softmax；PO 已知后未执行的 requested
尾部也不产生梯度。Primary 和所有 batch 的实际条件选择 entropy 合并后取均值。PPO
replay 使用保存的 `remaining_rows`、`primary_row` 和嵌套 batch rows，一次 forward 重建
joint log probability、mean entropy 和 value。

## 5. 固定求解协议

```text
Primary backtrack limit = 100
DTC secondary limit     = 50
PODEM seed              = 14
attempts per Primary    = 1
DTC                     = enabled
STC                     = enabled
STC shuffle seed        = 7
STC no-improvement      = 5
rollback                = accepted_pi_cube_resim_v1
```

DTC 完成后进行 fault simulation。所有 Primary 结束后执行 reverse order 与固定 seed
shuffle STC。最终优化指标是 `patterns_after_stc`。

## 6. 奖励、GAE 与 PPO

设 `InitialEqv` 为电路初始 equivalent fault 总数。每步 shaping：

```text
r_t = (-pattern_increment + 0.1 * newly_detected_eqv) / InitialEqv
```

Episode 目标回报：

```text
shortfall == 0: 1 - patterns_after_stc / InitialEqv
shortfall > 0 : -10 * shortfall / InitialEqv
```

最后一步加入 terminal correction，使 `sum(r_t)` 严格等于目标回报。随后以
`gamma=1.0`、`lambda=0.95` 计算 GAE，并标准化 advantage。每个 circuit rollout 后执行
4 次 PPO epoch：clip `0.2`，value 系数 `0.5`，entropy 系数 `0.01`，gradient clip `1.0`。

## 7. Training、validation 与恢复

训练 manifest 由调用方指定；validation 默认 `configs/anchor_validation_6.json`。两个 split
分别建立 native baseline，artifact provenance 不得重叠。Validation 不更新参数。每轮按
以下字典序 key 选择 best：

```text
(sum(validation coverage shortfall), sum(validation patterns_after_stc))
```

固定执行 5 轮。每个 training circuit 完成 rollout 和 4 次 PPO epoch 后，写入 circuit
artifacts，并原子提交 schema 5 `latest.pt`。Checkpoint 保存模型、optimizer、RNG、两个
manifest、provenance、两个 split 的 baseline、completed round 和 next circuit index。

每轮完成后只运行独立 validation 并更新 `best.pt`；第 5 轮结束发布 `final.pt`。Schema
1–4 与新的 BFS action mask、joint probability 和 rollback protocol 不兼容，不能恢复。
