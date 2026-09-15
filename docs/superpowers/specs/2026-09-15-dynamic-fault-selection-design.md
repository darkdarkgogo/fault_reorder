# 基于剩余 fault 上下文的动态 fault 选择

## 目标

把当前“一次生成完整 fault permutation，再由 PODEM 沿固定顺序执行”的静态策略，
改成“每次 primary-fault 尝试后，根据 fault simulation 更新后的候选集合重新打分并
选择下一个 fault”的动态策略。唯一主要优化目标仍是减少最终 test pattern 数量；
模型不得通过降低当前定义的 resolved coverage 获益。

现有 257 维 DeepGate2 fault embedding 保持冻结，也不改变 fault catalog、fault
collapsing、原始 BENCH/AIG anchor 或 PODEM 搜索算法。本功能只改变 scorer 输入、
策略采样方式以及 Python 与 PODEM 之间的执行粒度。

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

## Stateful PODEM 环境

当前 `run_stuck_at_ordered()` 一次接收完整 permutation 并运行到结束，Python 无法
在两次 fault selection 之间看到 fault-sim 状态。动态策略需要新增一个 episode
级 stateful binding。概念接口为：

```python
session = StuckAtSession(
    circuit_path,
    fault_map_path,
    backtrack_limit=5000,
    seed=14,
)

session.catalog()             # 初始、稳定的 fault catalog
session.remaining_fault_ids() # 当前合法候选
step = session.step(fault_id) # 只尝试一个 primary fault
summary = session.result()    # episode 结束后的累计指标
```

`step(fault_id)` 必须先确认 fault ID 属于当前候选集合，再执行一次现有 PODEM 分支：

- TRUE：生成一个 pattern，立即进行 fault simulation 和 fault dropping；
- FALSE：标记 redundant，并累计对应 equivalent fault 数；
- MAYBE：标记 aborted；
- 所有情况都把本 primary fault 标记为已尝试，并更新累计 calls/backtracks。

每步至少返回：

```text
selected_fault_id
target_status                 # detected / redundant / aborted
generated_pattern             # 本步是否新增一个 pattern
newly_detected_fault_ids      # 本次 fault sim 实际 drop 的 catalog IDs
remaining_fault_ids
current_pattern_count
current_podem_calls
current_total_backtracks
```

`result()` 延续当前环境的完整指标语义：pattern、detected collapsed/equivalent、
uncollapsed、aborted、redundant/equivalent、PODEM calls 和 total backtracks。
Python 层继续派生 covered equivalent faults 与 resolved coverage。

旧的 `run_stuck_at_ordered()` 可以保留用于兼容和回归测试，但动态训练、native
baseline 和动态评估统一通过 session 执行，避免两个执行路径产生指标漂移。native
baseline 在每一步选择当前 catalog 中最早的合法候选，等价于现有原始顺序行为。

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

coverage 低于 native resolved coverage 时，继续使用现有强负奖励，并且不更新
previous pattern baseline。模型不能通过制造 aborted fault 或漏检来获得较少
pattern 的正收益。

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
  新 drop 的 fault IDs 和累计 solver 指标。

训练轮次中的 sampled trajectory 使用紧凑 NPZ 保存整数 mask/indices 和 selected rows；
面向人工检查的确定性 evaluation trace 使用 JSONL 保存。现有 per-circuit metrics、
comparison CSV、summary、coverage eligibility 和 best comparison key 保持不变。

## Checkpoint 与恢复

新模型输入维度、策略分布和轨迹格式均与当前 checkpoint 不兼容，因此 checkpoint
schema 升级。旧的静态 257 维 checkpoint 必须给出明确的 restart 错误，不能部分
加载 scorer 权重或 optimizer state 后继续训练。

新 checkpoint 除当前内容外，还需要记录动态策略版本、输入布局、初始 fault 数、
temperature、轨迹 RNG 状态和 session binding digest。轮次事务边界不变：只有本轮
全部 episode、梯度更新、必要评估和日志写入成功后才原子发布 `latest.pt`。

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
  最终与 `result()` 一致。
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
- TRUE 后返回精确的 newly detected IDs，并从 remaining 集合 drop。
- FALSE 和 MAYBE 不增加 pattern，并且目标不再可选。
- 已 drop、已尝试、重复及未知 ID 在 PODEM 前被拒绝。
- 按 catalog 首个剩余 fault 逐步运行的 native session 指标与当前
  `run_stuck_at_ordered(native_order)` 完全一致。
- 固定 seed 下重复 session 的 step trace 和最终指标一致。

### Trainer 与端到端测试

- fake session 构造一个首步可 drop 多个 fault 的场景，证明第二步重新计算 mean、
  ratio 和 scores，而不是沿用初始顺序。
- sampled trajectory 在 `no_grad` 收集后可用相同参数精确重算 log probability。
- coverage penalty、previous-pattern baseline、EMA advantage、minibatch 和 best 选择
  保持当前语义。
- 动态 evaluation 生成完整 decision/drop trace，不把被顺带 drop 的 fault 标记为
  primary selection。
- checkpoint resume 精确恢复下一条动态轨迹；旧 schema 被明确拒绝。
- 至少一个微型真实电路完成 native baseline、动态 episode、optimizer update、
  checkpoint、resume 和确定性评估。

## 验收标准

1. 每次 primary-fault 尝试后都依据新的合法候选集合重算 remaining mean、ratio 和
   全部候选分数。
2. scorer 的每步输入严格为 257 维 fault embedding、257 维 remaining mean 和
   1 维 remaining ratio，共 515 维。
3. 训练使用动态 masked categorical trajectory log probability，损失除以初始
   fault 数，不除以实际决策步数。
4. 梯度不穿过 PODEM；动态轨迹可在 episode 后重放并产生有限 scorer 梯度。
5. native baseline、coverage guard、pattern-count reward、best 资格和最终指标与当前
   定义一致。
6. 确定性评估可复现，并明确区分 primary-selected fault 与 fault-sim-dropped fault。
7. 静态旧 checkpoint 不得静默迁移；新 checkpoint 可精确中断恢复。

## 不在本次范围内

- 微调 DeepGate2 或重新生成 257 维 embedding。
- 修改 fault collapsing、fault catalog、PODEM objective/backtrace 或 fault simulation
  语义。
- attention、Transformer、显式 embedding 乘积/差值或 marginal-coverage 辅助头。
- Top-K PODEM lookahead、STC、DTC、SCOAP、transition-delay 或并行 solver worker。
- 将 PODEM calls、backtracks 或运行时间加入主要 reward；第一版仍只以 pattern count
  改进为正向目标，并用 coverage penalty 保证有效性。
