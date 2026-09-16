# 基于剩余 fault 上下文的动态 fault 选择

## 目标

把当前“一次生成完整 fault permutation，再由 PODEM 沿固定顺序执行”的静态策略，
改成“每次 primary-fault 尝试后，根据 fault simulation 更新后的候选集合重新打分并
选择下一个 fault”的动态策略。唯一主要优化目标仍是减少最终 test pattern 数量；
模型不得通过降低当前定义的 resolved coverage 获益。

现有 257 维 DeepGate2 fault embedding 保持冻结，也不改变 fault catalog、fault
collapsing 或原始 BENCH/AIG anchor。本功能改变 scorer 输入、策略采样方式、Python
与 PODEM 之间的执行粒度，并为 stuck-at 路径新增 PODEMX 动态测试压缩和生成后的
静态测试压缩。

## 方案选择

考虑过以下三种方案：

1. 将 fault embedding 与整个初始电路的 mean pooling 拼接。改动最小，能够提供
   circuit context，但 pooling 在 episode 内固定，不能根据 fault dropping 改变排序。
2. 将 fault embedding 与当前可选 fault 的 mean pooling、remaining ratio 拼接，
   并在每次 PODEM 尝试后重新打分。本次采用该方案；它能提供动态上下文，同时保留
   简单的 MLP scorer 和现有 REINFORCE 训练框架。
3. 使用 attention、Set Transformer 或 Pointer Network 对剩余集合建模。表达能力
   更强，但第一版会显著增加模型、训练和验证复杂度，暂不采用。

第一版不加入手工点积、逐维乘积或绝对差等交互特征。非线性 MLP 直接从拼接后的
输入学习 fault 与剩余集合之间的条件关系。后续只有在消融实验表明简单拼接不足时，
才考虑增加显式交互项。

## 固定求解与压缩配置

native baseline、训练 episode 和确定性评估必须使用完全相同的求解与压缩配置：

```text
primary PODEM backtrack limit       = 200
primary PODEM seed                  = 14
attempts per primary fault          = 1
stuck-at DTC                        = enabled
DTC secondary backtrack limit       = 50
stuck-at STC                        = enabled
STC reverse-order compaction        = enabled
STC shuffle seed                    = 7
STC consecutive no-improvement limit= 5
SCOAP fault ordering                = disabled
transition-delay mode               = disabled
```

主 PODEM 的回溯上限 200 作用于每一次 primary-fault 调用，不是整个电路的总上限。
达到上限仍未找到测试向量时，该 primary fault 返回 MAYBE/aborted。DTC 对每个
secondary fault 使用独立的 50 次回溯上限。

SCOAP 在本项目中只用于重排 fault list，因此必须关闭。primary fault 的选择顺序
只能由 native baseline 的 catalog-first 规则或动态策略决定，不能再被启发式排序覆盖。

## 动态状态与候选集合

对于一个初始包含 \(N\) 个 collapsed fault 的电路，fault \(i\) 的冻结 embedding 为：

\[
e_i\in\mathbb R^{257}.
\]

第 \(t\) 步的候选集合 \(R_t\) 只包含仍允许作为 primary target 的 fault。以下 fault
不属于候选集合，也不参与 remaining mean：

- 已被 fault simulation 检测并 drop 的 fault；
- 已被 PODEM 判定为 redundant 的 fault；
- 已尝试但因 backtrack limit 返回 aborted/MAYBE 的 fault。

PODEM 保持当前 one-attempt 协议，因此一个 fault 在同一 episode 中最多被选中一次。
如果一次成功的 pattern 同时检测多个 fault，这些 fault 在本步结束时全部从
\(R_t\) 移除。FALSE 或 MAYBE 不产生 pattern，但目标 fault 仍从后续候选集合移除，
然后策略基于新集合重新决策。

当前剩余集合的动态 mean pooling 为：

\[
c_t=\frac{1}{|R_t|}\sum_{j\in R_t}e_j.
\]

remaining ratio 为：

\[
r_t=\frac{|R_t|}{N}.
\]

初始时 \(R_0\) 是完整 catalog，因此 \(c_0\) 同时也是整个初始电路的 fault mean，
且 \(r_0=1\)。当 \(R_t\) 为空时 episode 结束，不再计算 mean，避免空集合产生
非有限数值。

## 动态 scorer

对于当前候选 fault \(i\in R_t\)，构造：

\[
x_{i,t}=[e_i\Vert c_t\Vert r_t]\in\mathbb R^{515}.
\]

其中前 257 维描述候选 fault，接下来的 257 维描述当前剩余 fault 集合，最后一维
描述 episode 进度和集合规模。对同一步的所有候选，\(c_t\) 和 \(r_t\) 相同，
\(e_i\) 不同。

共享 scorer 使用以下结构：

```text
输入 x_i,t ∈ R^515
  → LayerNorm(515)
  → Linear(515, 256) + ReLU
  → Linear(256, 128) + ReLU
  → Linear(128, 1)
  → 当前候选 score s_i,t
```

必须保留至少一层非线性激活。若拼接后只使用线性打分，同一步所有候选共享的
context 只会形成公共偏置，无法改变候选之间的相对次序。模型不使用 dropout，
以便在相同状态下重算出完全相同的 logits。

## 动态 masked categorical policy

当前静态 Plackett–Luce 策略在 episode 开始时计算一次 logits 并采样完整
permutation。新策略在每一步重新计算 logits，只在 \(R_t\) 内选择一个动作。

训练时先对本步候选分数中心化、除以温度 \(\tau\)，再计算：

\[
P_\theta(a_t=i\mid R_t)
=\frac{\exp(z_{i,t})}{\sum_{j\in R_t}\exp(z_{j,t})}.
\]

mask 的语义是：不在 \(R_t\) 中的 fault 不进入 softmax，选择概率严格为零。
固定训练 seed、checkpoint RNG 状态、数据和 PODEM 二进制时，采样轨迹必须能够
精确恢复。

评估时不采样，选择本步 score 最大的 fault；score 完全相同时使用原始 catalog
行号较小者，保证动态轨迹确定且可复现。

一次 episode 实际选择 \(T\) 个 primary fault，完整轨迹为
\(\tau=(a_0,\ldots,a_{T-1})\)。由于每一步的 remaining mean、ratio 和 mask 都可能
不同，轨迹 log probability 为：

\[
\log P_\theta(\tau)
=\sum_{t=0}^{T-1}\log P_\theta(a_t\mid R_t).
\]

这不是固定 logits 下的完整 permutation：每次 fault simulation 后都会根据新的
\(R_t\) 重新构造 515 维输入并重新计算全部候选分数。

## Stuck-at PODEMX 动态测试压缩

当主 PODEM 对 primary fault 返回 TRUE 后，先保留当前尚未确定的输入位，不立即
随机填充。stuck-at PODEMX 随后排除本步 primary fault，按当前 catalog 行号从小到大
的固定顺序扫描其他仍可选的 secondary fault，尝试利用这些未知输入位让同一个测试
cube 同时检测更多 fault。扫描在全部 secondary 候选都已尝试或不再存在未知输入位
时结束。
该固定顺序只决定 DTC 内部的 secondary 尝试顺序，不构成新的 primary 选择，也不
使用 SCOAP。

每次 secondary 尝试必须满足以下约束：

- 只能继续约束当前仍为未知值的输入位，不能修改已经固定的输入位；
- 必须保持 primary fault 仍然可检测；
- 必须保持之前已经成功加入的 secondary fault 仍然可检测；
- 每个 secondary fault 最多允许 50 次 PODEMX 回溯；
- 成功时保留新增约束，失败或达到回溯上限时完整回滚本次尝试产生的赋值；
- secondary 尝试不把 fault 标记为 `test_tried`、redundant 或 aborted；
- 只有最终测试向量经过 fault simulation 确实检测到该 fault 时，才能把它从
  remaining 集合中 drop。

DTC 完成后，使用会话级 seed 14 随机数生成器填充仍然未知的输入位，然后只执行
一次 stuck-at fault simulation。本步返回的 `newly_detected_fault_ids` 以这次 fault
simulation 的实际检测结果为准，其中可以同时包含 primary fault、DTC 加入的
secondary fault，以及被测试向量顺带检测到的其他 fault。

DTC 的搜索开销与主 PODEM 分开记录：

```text
primary_podem_calls
dtc_secondary_calls
primary_backtracks
dtc_backtracks
total_backtracks = primary_backtracks + dtc_backtracks
```

`primary_podem_calls` 只统计策略或 native baseline 主动选择 primary fault 后的主
PODEM 调用；`dtc_secondary_calls` 统计 secondary 尝试次数，无论该次尝试成功还是
失败。为兼容现有输出，`podem_calls` 是 `primary_podem_calls` 的别名，不包含 DTC
secondary 尝试；`total_backtracks` 表示本次协议实际执行的全部搜索回溯。

## Stuck-at 静态测试压缩

当 remaining 集合为空、所有 primary 尝试结束后，对本回合生成的完整测试向量执行
一次 STC。STC 不参与逐步 mask 更新，也不会回写或改变已经记录的策略轨迹。

STC 分为两个确定性阶段：

1. 反向应用测试向量并执行 stuck-at fault simulation；删除没有新增检测贡献的向量。
2. 使用独立的会话级 seed 7 随机数生成器打乱剩余向量，再执行相同的压缩；连续
   5 次 shuffle 都不能删除向量时停止。

STC 必须保留压缩前全部已检测 fault 的集合及 equivalent fault coverage。压缩实现
不能覆盖 primary 求解阶段保存的 redundant、aborted、calls 或 backtracks 状态。
如果压缩后检测集合或 coverage 发生下降，应把它视为求解器错误并中止当前回合，
不能把它当作普通 coverage penalty 样本。

pattern 数明确区分为：

```text
current_pattern_count       # step 阶段已经生成的数量，只增不减
patterns_before_stc         # episode 结束、STC 开始前的数量
patterns_after_stc          # STC 结束后的最终数量
pattern_count               # 对外兼容字段，等于 patterns_after_stc
```

训练 reward、best 模型比较和最终评估使用 `patterns_after_stc`。逐步轨迹使用
`current_pattern_count`，不得把回合末尾 STC 的减少误报成某个 primary fault 步骤
生成了负数 pattern。

## Stateful PODEM 环境

当前 `run_stuck_at_ordered()` 一次接收完整 permutation 并运行到结束，Python 无法
在两次 fault selection 之间看到 fault-sim 状态。动态策略需要新增一个 episode
级 stateful binding。概念接口为：

```python
session = StuckAtSession(
    circuit_path,
    fault_map_path,
    backtrack_limit=200,
    seed=14,
    dtc_enabled=True,
    dtc_backtrack_limit=50,
    stc_enabled=True,
    stc_seed=7,
    stc_no_improvement_limit=5,
)

session.catalog()             # 初始、稳定的 fault catalog
session.remaining_fault_ids() # 当前合法候选
step = session.step(fault_id) # 只尝试一个 primary fault
summary = session.result()    # episode 结束后的累计指标
```

`step(fault_id)` 必须先确认 fault ID 属于当前候选集合，再执行一次 primary PODEM
分支：

- TRUE：运行 stuck-at PODEMX DTC、填充剩余未知输入位、生成一个 pattern，再进行
  fault simulation 和 fault dropping；
- FALSE：标记 redundant，并累计对应 equivalent fault 数；
- MAYBE：标记 aborted；
- 所有情况都把本 primary fault 标记为已尝试，并更新累计 primary/DTC calls 与
  backtracks。

每步至少返回：

```text
selected_fault_id
target_status                 # detected / redundant / aborted
generated_pattern             # 本步是否新增一个 pattern
newly_detected_fault_ids      # 本次 fault sim 实际 drop 的 catalog IDs
remaining_fault_ids
current_pattern_count
current_primary_podem_calls
current_dtc_secondary_calls
current_primary_backtracks
current_dtc_backtracks
current_total_backtracks
```

`result()` 延续当前环境的完整指标语义：pattern、detected collapsed/equivalent、
uncollapsed、aborted、redundant/equivalent、PODEM calls 和 total backtracks。
此外返回 `patterns_before_stc`、`patterns_after_stc`、DTC calls 和拆分后的 backtracks。
Python 层继续派生 covered equivalent faults 与 resolved coverage。

当仍存在 selectable fault 时，`result()` 可以返回尚未 STC 的中间摘要，但必须用
`finalized=false` 明确标识，且 `pattern_count` 此时等于 `current_pattern_count`。
当 selectable fault 为空时，第一次调用 `result()` 原子地执行一次 STC 并缓存结果；
后续调用返回同一个 `finalized=true` 摘要，禁止重复 shuffle 或重复压缩。

旧的 `run_stuck_at_ordered()` 可以保留用于兼容和回归测试，但动态训练、native
baseline 和动态评估统一通过 session 执行，避免两个执行路径产生指标漂移。native
baseline 在每一步选择当前 catalog 中最早的合法候选，并使用与模型策略完全相同的
回溯、DTC 和 STC 配置。

PODEM 输入填充与 STC shuffle 使用两个独立的会话级随机数生成器。不得使用共享的
全局 `srand/rand` 状态；同一个会话对象的可变操作也必须串行化。这样固定配置下
重复会话的 step trace 和最终压缩结果可复现，并且不同 Python 线程中的会话不会
互相改变随机轨迹。

## Episode 收集与梯度重算

PODEM 执行期间不保留 autograd graph。采样阶段在 `no_grad` 下运行，并为每个动作
记录足以确定性重建网络输入的最小轨迹：

```text
remaining catalog-row mask/indices
selected catalog row
step result and cumulative solver metrics
```

episode 完成后，训练器使用同一模型参数、冻结 embedding 和记录的 remaining indices
逐步重算 \(c_t\)、\(r_t\)、logits 及被选动作的 log probability，再对所有步骤求和。
remaining mask 是 PODEM 状态，不需要也不能求梯度；梯度只通过 scorer 生成的
log probability 传播。

当前 reward、EMA advantage 和 coverage guard 保持不变。coverage 有效时：

\[
reward=P^{previous}-P^{current}.
\]

其中 \(P\) 一律表示 `patterns_after_stc`，不是逐步生成数量或
`patterns_before_stc`。native baseline 也必须先执行相同的 DTC/STC 协议，再建立
previous pattern baseline。

coverage 低于 native resolved coverage 时，继续使用现有强负奖励，并且不更新
previous pattern baseline。模型不能通过制造 aborted fault 或漏检来获得较少
pattern 的正收益。STC 自身造成检测集合或 coverage 下降属于求解器错误，必须中止
本轮，而不是进入普通 coverage penalty 分支。

单电路损失固定除以该电路的初始 fault 数 \(N\)：

\[
L_c=-A_c\frac{1}{N_c}
\sum_{t=0}^{T_c-1}\log P_\theta(a_t\mid R_t).
\]

不使用实际决策数 \(T_c\) 作为分母，因为 \(T_c\) 会随策略及 fault dropping 改变。
固定的 \(N_c\) 延续当前跨电路的梯度尺度归一化，避免大电路仅因候选更多而过度
支配共享 optimizer。batch 内继续对电路等权平均，Adam、learning rate、gradient
clipping、temperature schedule 和 minibatch 事务语义保持当前行为。

## 动态评估与输出

动态策略不存在一个在 episode 开始时即可导出的最终完整 permutation。一个从未被
选择的 fault 可能在其他 fault 的 pattern 下被 drop，因此不能把旧式静态 rank
误报为真实执行次序。

每个电路的评估输出改为同时保存：

- 初始状态下所有 fault 的 score 和稳定排序，作为诊断信息；
- 实际被选为 primary target 的 catalog row 序列；
- 每个初始 fault 的 `selected_step`，从未被选时为 `-1`；
- 每个 fault 的 `resolved_step` 和 resolved 原因；
- 每一步的 remaining count、ratio、selected score、target status、是否生成 pattern、
  新 drop 的 fault IDs、primary/DTC calls 和拆分后的累计 backtracks；
- `patterns_before_stc`、`patterns_after_stc`、STC 删除数量、shuffle 次数和最终
  coverage-preservation 检查结果。

训练轮次中的 sampled trajectory 使用紧凑 NPZ 保存整数 mask/indices 和 selected rows；
面向人工检查的确定性 evaluation trace 使用 JSONL 保存。现有 per-circuit metrics、
comparison CSV、summary 和 coverage eligibility 保持不变。best comparison key 仍先
比较最终 pattern 数，但该字段现在明确取 `patterns_after_stc`；其后的 coverage、
primary calls 和 total backtracks 比较继续使用最终回合摘要。

## Checkpoint 与恢复

新模型输入维度、策略分布和轨迹格式均与当前 checkpoint 不兼容，因此 checkpoint
schema 升级。旧的静态 257 维 checkpoint 必须给出明确的 restart 错误，不能部分
加载 scorer 权重或 optimizer state 后继续训练。

新 checkpoint 除当前内容外，还需要记录动态策略版本、输入布局、初始 fault 数、
temperature、轨迹 RNG 状态和 session binding digest，并记录以下求解协议身份：

```text
primary_backtrack_limit = 200
dtc_enabled = true
dtc_secondary_backtrack_limit = 50
stc_enabled = true
stc_shuffle_seed = 7
stc_no_improvement_limit = 5
scoap_enabled = false
compression_algorithm_version = "stuck_at_podemx_reverse_shuffle_v1"
```

恢复或评估时任一字段不一致都必须拒绝加载。轮次事务边界不变：只有本轮全部
episode、梯度更新、必要评估和日志写入成功后才原子发布 `latest.pt`。

恢复测试必须证明连续训练与中断后恢复产生相同的 sampled action trajectory、模型、
optimizer、baseline 和 EMA 状态。

## 失败处理

- 空初始 catalog、非有限 embedding、515 维输入不匹配或非有限 score 在 baseline
  之前失败。
- `step()` 收到未知、已检测、redundant、aborted 或已经尝试的 fault ID 时，在调用
  PODEM 前失败。
- C++ 返回的 remaining IDs 必须是初始 catalog 的无重复子集，且每一步只能减少；
  Python 检测到状态回退或未知 ID 时中止当前轮。
- TRUE 步必须新增一个 pattern；FALSE/MAYBE 步不得新增 pattern。累计指标必须单调且
  最终与 `result()` 中的 `patterns_before_stc` 一致；最终 `pattern_count` 可以因 STC
  小于逐步累计值，但必须等于 `patterns_after_stc`。
- DTC 失败或达到 secondary 回溯上限后，若 primary/既有 secondary 的约束没有完整
  恢复，应立即中止回合。
- `patterns_after_stc` 必须小于或等于 `patterns_before_stc`，且 STC 前后的已检测
  fault 集合和 equivalent coverage 必须完全一致，否则中止当前轮。
- solver 异常、非有限 loss、无效 mask 或不能重放的轨迹中止当前轮，不提交模型、
  optimizer、baseline、EMA 或 RNG 状态。
- coverage 不合格仍作为带强负 reward 的有效训练样本处理，不当作程序错误。

## 测试

### 模型与策略单元测试

- 对 `[N,257]` embedding 和合法 remaining mask 构造 `[K,515]` 输入。
- remaining mean、ratio 和候选行与手算结果一致；空 mask 被拒绝。
- 相同 fault 在不同 remaining 集合下获得不同的 515 维输入。
- masked softmax 对非候选 fault 给出零概率，概率在候选集合内归一化。
- 固定 seed 可复现动作；确定性模式按 score 和 catalog row 破同分。
- 动态轨迹 log probability 等于每步直接 categorical log probability 之和。
- `loss = -advantage * trajectory_log_prob / initial_fault_count`，不使用实际步数作为
  分母，并能产生有限、非零的 scorer 梯度。

### C++ session 与 binding 测试

- session 初始 remaining IDs 与 catalog 完全一致。
- primary PODEM 对每个 fault 使用 200 次回溯上限；达到上限时返回 MAYBE。
- TRUE 后运行 stuck-at PODEMX；每个 secondary fault 最多回溯 50 次。
- PODEMX 只能约束未知输入，并在保持 primary 可检测的情况下加入 secondary fault。
- DTC 失败会完整回滚本次赋值；成功的 secondary 只有经过最终 fault simulation
  检测后才从 remaining 集合 drop，且不会被标记为 primary selection。
- TRUE 后返回精确的 newly detected IDs，并从 remaining 集合 drop。
- FALSE 和 MAYBE 不增加 pattern，并且目标不再可选。
- 已 drop、已尝试、重复及未知 ID 在 PODEM 前被拒绝。
- STC 先反向压缩，再用 seed 7 shuffle；连续 5 次无改进后停止。
- STC 前后检测集合与 equivalent coverage 完全一致，且
  `patterns_after_stc <= patterns_before_stc`。
- 按 catalog 首个剩余 fault 逐步运行的 native session 与动态 session 使用相同的
  backtrack/DTC/STC 协议。
- 固定 seed 下重复 session 的 step trace、DTC 统计、STC 结果和最终指标一致。
- 并发创建和运行独立 session 不会互相改变随机轨迹；同一 session 的可变操作被
  串行化。

### Trainer 与端到端测试

- fake session 构造一个首步可 drop 多个 fault 的场景，证明第二步重新计算 mean、
  ratio 和 scores，而不是沿用初始顺序。
- sampled trajectory 在 `no_grad` 收集后可用相同参数精确重算 log probability。
- coverage penalty、previous-pattern baseline、EMA advantage、minibatch 和 best 选择
  保持当前语义，但所有 pattern 比较使用 `patterns_after_stc`。
- 动态 evaluation 生成完整 decision/drop trace，不把被顺带 drop 的 fault 标记为
  primary selection。
- evaluation 和 round artifacts 同时保存 DTC 开销及 STC 前后 pattern 数。
- checkpoint resume 精确恢复下一条动态轨迹；旧 schema 被明确拒绝。
- 至少一个微型真实电路完成 native baseline、动态 episode、optimizer update、
  DTC、STC、checkpoint、resume 和确定性评估。

## 验收标准

1. 每次 primary-fault 尝试后都依据新的合法候选集合重算 remaining mean、ratio 和
   全部候选分数。
2. scorer 的每步输入严格为 257 维 fault embedding、257 维 remaining mean 和
   1 维 remaining ratio，共 515 维。
3. 训练使用动态 masked categorical trajectory log probability，损失除以初始
   fault 数，不除以实际决策步数。
4. 梯度不穿过 PODEM；动态轨迹可在 episode 后重放并产生有限 scorer 梯度。
5. 每个 primary fault 的主 PODEM 回溯上限固定为 200；TRUE 后执行回溯上限为 50
   的 stuck-at PODEMX DTC，失败尝试不污染已建立的测试 cube。
6. episode 结束后执行 reverse-order 加固定 seed shuffle 的 STC；最终 reward 使用
   `patterns_after_stc`，并且压缩前后检测集合与 coverage 完全一致。
7. native baseline、coverage guard、pattern-count reward、best 资格和最终指标使用与
   模型策略完全相同的 backtrack/DTC/STC 协议。
8. 确定性评估可复现，并明确区分 primary-selected fault、DTC secondary 尝试与
   fault-sim-dropped fault。
9. 静态旧 checkpoint 或压缩配置不一致的 checkpoint 不得静默迁移；新 checkpoint
   可精确中断恢复。

## 不在本次范围内

- 微调 DeepGate2 或重新生成 257 维 embedding。
- 修改 fault collapsing、fault catalog、primary PODEM objective/backtrace 或 fault
  simulation 的检测语义。
- attention、Transformer、显式 embedding 乘积/差值或 marginal-coverage 辅助头。
- Top-K primary PODEM lookahead、SCOAP fault ordering、transition-delay ATPG 或并行
  solver worker。
- 将 PODEM calls、backtracks 或运行时间加入主要 reward；第一版仍只以 pattern count
  改进为正向目标，并用 coverage penalty 保证有效性。
