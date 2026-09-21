# 动态 Fault Ranking Actor-Critic PPO 严格设计规范

## 状态与效力

本规范记录 2026-09-20 确认的设计。实现以用户提供的
`Fault_Reorder_Dynamic_PPO_Training_Design_v2.3.pdf` 为主体，并应用
`Fault_Reorder_PPO_v2.3_Strict_Revision_Implementation_Guide.pdf` 的严格修正。
两份文档冲突时，以 Strict Revision 为准。

结合现有仓库实现，额外加入两项已经确认的约束：

- DTC 实际尝试顺序必须是输入 secondary ranking 的连续前缀。
- `latest.pt` 必须在每个 training circuit 完成后提交进度，不能只在整轮结束时保存。

现有固定 solver protocol 保持不变，唯一行为变化是 DTC secondary 顺序由策略提供，
C++ 内部不再强制恢复为 catalog order。

## 目标

把当前“动态选择 Primary 的 REINFORCE”训练器改成 Actor-Critic PPO。新策略同时控制：

- 当前 step 的 Primary fault；
- 当前 step 内真实执行的 DTC secondary priority。

在每个 circuit 的 covered-equivalent-fault coverage 不低于 heuristic baseline 的硬约束下，
优化最终 `patterns_after_stc`。使用独立 validation manifest 选择 checkpoint，并保证训练在
circuit 边界中断后可以精确恢复。

## 采用的方案

采用一次完成 Python、Pybind、C++ ATPG/DTC、训练器与 checkpoint 的全链路修改。

不采用以下替代方案：

1. 保留 catalog-ordered DTC，仅对 Primary 使用 PPO。这样策略声明的 ranking 与 solver
   实际执行动作不一致，不能称为学习 DC ordering。
2. 原样保留 v2.3 的旧 terminal reward。detected/redundant 构成变化时，dense shaping
   会改变最终目标，甚至可能抵消 coverage failure。
3. 只在 round 结束时保存。训练集有 1024 个 circuit，中断可能丢失一整轮的昂贵 rollout
   与 PPO 更新。

## 固定实验配置

第一版使用以下默认值：

```text
Training rounds                 = 5
每个 circuit rollout 的 PPO epochs = 4
gamma                           = 1.0
GAE lambda                      = 0.95
PPO clip epsilon                = 0.2
Value loss coefficient          = 0.5
Entropy coefficient             = 0.01
Actor learning rate             = 1e-4
Critic learning rate            = 1e-4
Gradient norm clip              = 1.0
Step shaping alpha              = 0.1
Coverage penalty beta           = 10.0
Training manifest               = 调用方显式传入
Validation manifest             = 默认 configs/anchor_validation_6.json
```

现有 PODEM protocol 保持：

```text
Primary backtrack limit         = 100
Primary seed                    = 14
每个 primary fault 尝试次数       = 1
Stuck-at DTC                    = enabled
DTC secondary backtrack limit   = 50
Stuck-at STC                    = enabled
STC reverse-order compaction    = enabled
STC shuffle seed                = 7
STC no-improvement limit        = 5
SCOAP ordering                  = disabled
Transition-delay mode           = disabled
```

所有数值配置必须为有限值并满足各自范围。正式训练固定为 5 rounds；resume 只能继续完成同一
次 5-round 训练的剩余部分，不能改变目标轮数。checkpoint 保存完整配置，配置不一致时拒绝恢复。

## 动态 State 与 Actor-Critic 模型

初始 fault embedding `e_i` 为 257 维。第 `t` 个 step 的 selectable fault set 为 `F_t`，
每个候选的输入继续使用：

```text
x_i,t = [e_i, MeanPool({e_j | j in F_t}), |F_t| / |F_0|]
```

输入维度保持 515。以下情况直接报错：空集合、重复 row、越界 row、非有限 embedding、
非有限 model output。

当前 scorer 改为共享 encoder 的 Actor-Critic：

```text
Candidate features [K, 515]
  -> LayerNorm(515)
  -> Linear(515, 256) + ReLU
  -> Linear(256, 128) + ReLU
  -> Candidate encodings [K, 128]

Actor head:
  Linear(128, 1) -> 每个 candidate 的 score

Critic head:
  MeanPool(candidate encodings)
  -> Linear(128, 64) + ReLU
  -> Linear(64, 1) -> V(s_t)
```

模型不使用 dropout。参数不变时，重放同一 state 必须精确复现 scores 与 value。共享 encoder、
Actor head 和 Critic head 由同一个事务性 optimizer 管理。

## Ranking Action 与概率

每个环境 step 都重新对全部 selectable faults 打分，并从 Plackett-Luce 分布采样一个无放回
完整 ranking。可以使用与该分布严格等价的 Gumbel top-k，但其 RNG state 必须进入 checkpoint。

- Rank 1 是 Primary。
- Rank 2...N 作为有序 DTC candidate list 原样传入 C++。
- deterministic validation 按 score 降序；score 完全相同时按原始 catalog row 从小到大。

环境实际执行的动作定义为：

```text
executed_sequence = [primary] + dtc_attempted_fault_ids
```

长度为 `K_t` 的 executed sequence，其 log-probability 是这 `K_t` 次 sequential masked
categorical 条件 log-probability 的和。未访问的 ranking tail 不进入 log-probability；
attempted-but-failed secondary 必须进入。

每次条件选择都有 categorical entropy。本 ATPG step 用于 PPO loss 的 entropy 定义为
executed decisions 上的均值：

```text
H_t = mean(H_t,1, ..., H_t,K_t)
```

使用均值避免 prefix 越长，entropy bonus 天然越大的长度偏置。

每个 ATPG step 的 rollout 至少保存：

```text
Step 前的 remaining catalog rows
Requested full ranking rows
Executed sequence rows
Old executed-sequence log-probability
Old value estimate
Step shaping reward
Done flag
Pattern increment
Newly detected equivalent-fault increment
DTC attempted / embedded IDs
Solver trace 与累计 metrics
```

虽然 PPO ratio 只使用 executed sequence，仍保存 requested full ranking，用于审计和连续前缀校验。

## Native Ranked-DTC 接口契约

Python-facing session API 从：

```python
step(primary_fault_id)
```

改为逻辑等价的：

```python
step(primary_fault_id, ranked_secondary_fault_ids)
```

调用时必须满足：

- `primary_fault_id` 当前可选；
- `ranked_secondary_fault_ids` 恰好包含其余所有 selectable faults；
- 不允许重复、未知 ID、Primary ID、遗漏 candidate 或额外 candidate；
- Primary 与 secondary 合并后是 pre-step selectable set 的完整 permutation。

Binding 将 ordered vector 原样传给 ATPG step，再传给 stuck-at DTC。
`saf_compaction.cpp` 不得按 `fault_no` 排序，也不得静默回退到 catalog order。

C++ 必须按真实执行顺序返回 `dtc_attempted_fault_ids`。返回的 attempted IDs 必须严格等于
`ranked_secondary_fault_ids` 的某个连续前缀。如果 DTC 遇到停止条件，应立即停止，不能跳过
中间 candidate 后继续尝试后面的 candidate。

`dtc_embedded_fault_ids` 必须是 attempted IDs 的保序子序列。Python 在 trajectory 进入训练前
检查所有返回值；出现换序、缺口、重复、未知 ID 或非前缀结果时，视为 protocol error。

现有结果字段继续保留，包括 attempted/embedded IDs、newly detected IDs、current pattern
count、primary/DTC backtracks、final STC counts 和 covered equivalent-fault metrics。
Fault dropping 继续以 fault simulation 的 newly detected 结果为准，不能用 DTC embedded 代替。

旧 ordered-run API 可以保留用于兼容和回归测试；training、validation 和 dynamic evaluation
统一使用 ranked session API。

## Step Reward 与 Terminal Correction

`InitialEqv` 是经过验证的 equivalent-fault multiplicity 总和。每个 solver step 后计算：

```text
I_pattern,t = current_pattern_count_t - current_pattern_count_t-1
DeltaDetectedEqv_t = 本 step newly detected fault IDs 的 eqv multiplicity 总和
r_t = (-I_pattern,t + 0.1 * DeltaDetectedEqv_t) / InitialEqv
```

`I_pattern,t` 必须由真实累计 pattern count 的差得到，不能根据 target status 推测。在当前
one-primary-call protocol 下只能是 0 或 1。未生成 pattern 且没有 newly detected fault 的
redundant/aborted step，其 shaping reward 为 0。

所有 Primary 完成并执行 STC 后，定义：

```text
shortfall = max(heuristic_covered_eqv - rl_covered_eqv, 0)

G_target = 1 - patterns_after_stc / InitialEqv       if shortfall == 0
G_target = -10 * shortfall / InitialEqv              otherwise

G_step = sum(r_t)
R_terminal = G_target - G_step
```

在计算 returns 与 GAE 前，把 `R_terminal` 加入最后一个 transition。因为 gamma 固定为 1.0：

```text
G_episode = G_step + R_terminal = G_target
```

因此 dense shaping 不能抵消 coverage failure，也不会改变 coverage 合格轨迹之间按
`patterns_after_stc` 的排序。

环境和训练器必须检查：

```text
0 <= patterns_after_stc <= patterns_before_stc <= InitialEqv
```

最后一个不等式来自：每个 selected collapsed fault 最多生成一个 pattern，且
`collapsed_fault_count <= InitialEqv`。实现中仍需显式验证。因此 coverage 合格 target 非负，
coverage 不合格 target 为负，实现严格 coverage-first 分离。

## GAE 与 PPO Update

Terminal state 的 value 为 0。把 terminal correction 加入最后一个 transition 后，使用
gamma 1.0、lambda 0.95 计算 returns 与 GAE。

只在当前 circuit trajectory 至少有两个有效 advantage，且 population standard deviation
（`unbiased=False`）大于 epsilon 时做 advantage normalization。否则保留 raw advantages，
避免单样本或零方差信号被归零。

每个 transition 使用：

```text
ratio = exp(new_executed_log_prob - old_executed_log_prob)
actor_loss = -mean(min(ratio * A, clip(ratio, 0.8, 1.2) * A))
critic_loss = mean((V(s_t) - return_t)^2)
entropy_bonus = mean(H_t)
total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
```

一个 circuit 必须使用固定参数完成完整 rollout。之后对同一 trajectory 做 4 次 PPO epoch，
不重新执行 ATPG。每个 epoch 都从保存的 pre-step state 和 executed sequence 重算 new
log-probability、value 与 entropy。rollout 期间禁止更新参数。

提交 optimizer update 前，必须检查 advantages、log ratios、ratios、各项 loss、gradient norm
以及所有模型参数均为 finite。梯度范数裁剪为 1.0。日志记录 approximate KL 和 clip fraction。
失败时中止当前尚未提交的 circuit，最后一个 checkpoint 与其中的 RNG state 保持不变。

## Training 与 Validation 隔离

Trainer 分别持有：

- `train_circuits`：调用方传入的 training manifest；
- `validation_circuits`：显式 validation manifest，默认
  `configs/anchor_validation_6.json`。

两个 manifest 必须解析为不同文件。Circuit name 与 artifact identity 必须互不重叠；重叠时
立即报配置错误。两个 split 分别使用相同固定 PODEM/DTC/STC protocol 运行并缓存自己的
catalog-first heuristic/native baseline。

Training circuits 仅用于 stochastic rollout、GAE/PPO update 与 training diagnostics。
Validation circuits 仅在每个完整 round 后做 deterministic inference，不执行 backward 或
`optimizer.step()`。

`best.pt` 只使用 validation 结果比较：

```text
(
  sum(max(validation_heuristic_covered_eqv - validation_rl_covered_eqv, 0)),
  sum(validation_patterns_after_stc),
)
```

越小越好。在出现 zero-shortfall checkpoint 前，可以保存 shortfall 最小的模型；出现
zero-shortfall 模型后，positive-shortfall 模型不能替换它。

本次修改不创建 test dataset。Evaluation command 继续支持调用方传入 external manifest，
并支持评估 `best.pt` 与 `final.pt`，全程不更新参数。

## 五轮训练流程

训练 circuit 顺序固定并写入 checkpoint。每个 round 从 1 到 5：

1. 从 checkpoint 的 next circuit index 开始，为当前 training circuit 收集完整 stochastic trajectory。
2. 对该 trajectory 执行 4 次 PPO epoch。
3. 原子写入该 circuit 的训练 artifacts，并发布 `latest.pt`，记录下一个 circuit index 与
   update 后的 RNG state。
4. 继续直到本轮所有 training circuits 完成。
5. 只在独立 validation set 上执行 deterministic validation。
6. 根据 validation key 更新 best；原子提交 round-complete `latest.pt`，然后派生并发布 `best.pt`。
7. 进入下一 round 的第一个 circuit。

Round 5 完成后，无条件原子保存当前模型为 `final.pt`，并保留 `best.pt`。两者可以来自不同 round。

默认语义固定为“一个 circuit rollout，紧接着做它的 4 次 PPO epoch”。删除旧的多 circuit
REINFORCE minibatch 选项，避免含糊的 PPO 语义。

## Checkpoint Schema 与精确恢复

Checkpoint schema 从 version 3 升级为 version 4，因为 model、optimizer、policy probability、
reward、validation state 与进度语义都不兼容。Version 1-3 不能用于 resume 或新评估，必须给出
明确的“需要重新训练”错误。

`latest.pt` 至少包含：

```text
Actor-Critic state
Optimizer state
Completed round 与 active round
Next training circuit index
固定的 training circuit traversal order
Python / NumPy / Torch RNG states
Training / validation manifest paths 与 digests
Training / validation artifact provenance
两个 split 的 native baselines
完整配置与 fixed solver protocol identity
Solver binary digest 与 Torch version
当前 best model / report / key
已提交的 per-round / per-circuit training progress
```

一个 circuit checkpoint 提交后，如果下一个 circuit 运行中崩溃，resume 必须恢复到该 rollout
之前的精确 model、optimizer、progress 与 RNG state。从该点重跑必须复现 requested rankings、
executed sequences、模型参数、optimizer state、日志和 checkpoint selection。

`best.pt` 与 `final.pt` 不包含 optimizer 和 RNG，不可用于 resume；但保留 model、config、
provenance 与 checkpoint kind。`best.pt` 记录 validation key、report 和来源 round；`final.pt`
记录 Round 5 的最终 policy identity。

## 日志与 Artifacts

Training records 至少包含：

- circuit、round、circuit index、episode steps、`InitialEqv`；
- patterns before/after STC 与各 step pattern increment；
- training heuristic coverage、RL coverage、shortfall、coverage valid；
- sum step reward、terminal correction、target return、episode return；
- advantage mean/std、actor loss、critic loss、entropy、approximate KL、clip fraction、
  gradient norm；
- requested ranking audit、executed sequence、DTC attempted/embedded、backtracks、runtime。

Compact NPZ trajectory 保存 remaining/requested/executed row arrays 与 offsets。面向人工检查的
deterministic validation/evaluation trace 继续使用 JSONL。Validation report 与 comparison CSV
必须标明 validation manifest，且不能混入 aggregate training metrics。

## 失败处理

当前 circuit update 是事务边界。以下情况中止当前 circuit，不发布 model、optimizer、progress、
已提交日志或新 RNG state：

- features、values、rewards、advantages、ratios、loss、gradients 或 parameters 非有限；
- native ranking input 不是 selectable set 的 permutation；
- DTC attempted IDs 不是 requested secondary order 的连续前缀；
- embedded IDs 不是 attempted IDs 的保序子序列；
- native trace 含未知、重复、换序或静默跳过的 ID；
- pattern/coverage counter 回退或违反 final STC invariants；
- episode return 与 `G_target` 的差超过浮点容差；
- training/validation manifest 重叠或 provenance 不匹配；
- checkpoint config、solver digest 或 runtime 不兼容。

Coverage shortfall 是合法训练结果，按严格负 target 处理，不属于程序错误。

## 必须通过的测试

### Model 与 Policy

- Actor scores 和 Critic value shape 正确，梯度有限。
- update 前重放相同 state 得到相同 scores 与 value。
- stochastic ranking 是合法 permutation；deterministic ranking 使用稳定 catalog-row tie break。
- Executed-prefix log-probability 与直接逐次 categorical 计算一致。
- 未访问 tail 不影响 executed log-probability 与 entropy。
- Attempted-but-failed rows 必须参与两者。
- `K_t=1` 时 entropy 有限；长 prefix 使用 entropy mean 而不是 sum。

### Reward、GAE 与 PPO

- 无 pattern、无新增检测的 redundant/aborted step reward 为 0。
- Detection increment 使用 equivalent multiplicity，不是 collapsed count。
- Coverage 合格时总回报精确等于 `1 - P_after/InitialEqv`。
- Coverage 不合格时，无论 shaping 累计多少，总回报精确等于
  `-10 * shortfall/InitialEqv`。
- Valid target 非负，invalid target 为负。
- Terminal correction 在 GAE 前加入最后一个 transition。
- 单样本/零方差 advantage 不标准化；正常 batch 执行标准化。
- 4 次 PPO epoch 复用同一 rollout，所有 metrics 有限，且不会再次调用 environment。
- PPO clipping、critic loss、entropy coefficient、gradient clipping 与配置一致。

### Native Ranked DTC

- Binding 与 C++ 接收 Primary 和完整有序 secondary permutation。
- Ranking 从 Python 到 ATPG/compaction 全链路保持顺序。
- Attempted IDs 是连续前缀，包含失败尝试。
- 换序、中间跳过、重复、未知、遗漏、额外 candidate 均被拒绝。
- Embedded IDs 是 attempted IDs 的保序子序列。
- 现有 cube rollback、primary preservation、fault simulation、backtrack、STC、并发和
  deterministic-session 回归测试继续通过。

### Training、Validation 与 Checkpoint

- Training/validation manifest 与 baseline 分离；重叠会被拒绝。
- Training circuits 不进入 validation 或 best selection。
- Validation 不改变模型或 optimizer。
- Best key 使用 `(shortfall, patterns_after_stc)`；在 zero-shortfall 模型出现前可以选择
  positive-shortfall 模型。
- `latest.pt` 在每个 circuit 完成后推进，并精确从下一个 circuit 恢复。
- Rollout 或任意 PPO epoch 失败时，前一个 circuit checkpoint 与 RNG state 不变。
- 连续训练与中断恢复产生相同 rankings、traces、models、optimizer states、best selection
  和 final checkpoint。
- 五轮结束后 `latest.pt`、`best.pt`、`final.pt` 同时存在，best 与 final 可以不同。
- Schema version 1-3 给出明确的 retraining-required 错误。
- External deterministic evaluation 支持 best 与 final checkpoint。

## 文档与兼容性

更新 CLI help、README 和 `docs/fault-order-rl.md`，说明 validation manifest、固定五轮 PPO
默认值、ranked DTC、terminal correction、checkpoint kinds 与重新训练要求。

旧 static/listwise helper 只有在兼容测试仍使用时才保留；删除无调用的 REINFORCE/EMA 训练路径
和误导性文档。

本次修改不改变 frozen fault embeddings、fault-map 语义、fault collapsing、BENCH/AIG anchor
或固定 PODEM/STC 算法；唯一 solver 行为变化是显式传入的 DTC priority order。
