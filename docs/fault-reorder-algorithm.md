# 动态故障重排算法

当前实现是 coverage-first 的 Actor-Critic PPO。完整约束见
[`2026-09-20-dynamic-ppo-strict-design.md`](superpowers/specs/2026-09-20-dynamic-ppo-strict-design.md)，
运行命令见 [`fault-order-rl.md`](fault-order-rl.md)。

## 1. 动态状态与排名

每个 fault 有固定的 257 维 embedding。对当前 remaining set `F_t` 中的候选 `i`，模型输入为：

```text
[fault_embedding_i, mean_embedding(F_t), |F_t| / |F_0|]
```

总维度为 515。共享 encoder 输出：

- actor score：每个 remaining fault 一个标量；
- critic value：对 remaining set 编码取均值后输出一个状态值。

训练时用 Gumbel 排序采样完整 permutation；验证和评估按 score 降序，同分按 catalog
row 升序。

## 2. Primary 与 DTC 的统一动作

每个 step 的完整排名 `r_t` 直接决定求解行为：

```text
Primary = r_t[0]
DTC secondary priority = r_t[1:]
```

C++ 必须收到所有非 Primary 的 selectable fault，拒绝缺失、重复和未知 ID。DTC 按传入
顺序尝试，Python 再校验：

- `dtc_attempted_fault_ids` 是请求 secondary ranking 的连续前缀；
- `dtc_embedded_fault_ids` 是 attempted 的保序子序列。

PPO 只对实际执行序列 `Primary + attempted secondaries` 计算联合 log probability，未执行
后缀不产生梯度。

## 3. 固定求解协议

```text
Primary backtrack limit = 100
DTC secondary limit     = 50
PODEM seed              = 14
attempts per Primary    = 1
DTC                     = enabled
STC                     = enabled
STC shuffle seed        = 7
STC no-improvement      = 5
```

Primary 成功后先做 ranked DTC，再进行 fault simulation。所有 Primary 结束后执行 reverse
order 与固定 seed shuffle STC。最终优化指标是 `patterns_after_stc`。

## 4. 奖励、GAE 与 PPO

设 `InitialEqv` 为电路初始 equivalent fault 总数。每步 shaping：

```text
r_t = (-pattern_increment + 0.1 * newly_detected_eqv) / InitialEqv
```

episode 目标回报：

```text
shortfall == 0: 1 - patterns_after_stc / InitialEqv
shortfall > 0 : -10 * shortfall / InitialEqv
```

最后一步加入 terminal correction，使 `sum(r_t)` 严格等于目标回报。随后以
`gamma=1.0`、`lambda=0.95` 计算 GAE，并标准化 advantage。每个 circuit rollout 后立即
做 4 次 PPO epoch：clip `0.2`，value 系数 `0.5`，entropy 系数 `0.01`，gradient clip `1.0`。

## 5. Training / validation 隔离

训练 manifest 由调用方指定；validation 默认 `configs/anchor_validation_6.json`。两个 split
分别建立 native baseline，artifact provenance 不得重叠。Validation 不更新参数。

每轮结束根据以下字典序 key 选择 best：

```text
(sum(validation coverage shortfall), sum(validation patterns_after_stc))
```

因此覆盖优先级严格高于向量数。

## 6. 五轮流程与恢复

固定执行 5 轮。每个训练 circuit 完成 rollout 和 4 次 PPO epoch 后，写入 circuit artifacts，
并原子提交 schema 4 `latest.pt`。Checkpoint 保存模型、optimizer、RNG、两个 manifest 及
provenance、两个 split 的 baseline、completed round 和 next circuit index。

每轮完成后只运行独立 validation 并更新 `best.pt`。第 5 轮结束时无条件发布 `final.pt`。
`best.pt` 和 `final.pt` 不含 optimizer/RNG，且可能来自不同轮。Schema 1–3 不兼容，不能
恢复或用新评估器读取。
