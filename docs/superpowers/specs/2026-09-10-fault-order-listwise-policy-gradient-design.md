# 基于列表级策略梯度的共享 fault 排序

## 目标

使用全部 16 个内置二值 benchmark 电路，共同训练一个共享的 fault 神经网络
打分器。对于每个电路，模型为现有的每个折叠逻辑 fault 打分，对完整 fault
列表进行排序，然后运行一次 stuck-at PODEM ATPG，并以生成的 pattern count
作为强化学习反馈。训练按轮次重复，并保存确定性评估结果最好的 scorer
checkpoint。

优化目标是在不降低 detected fault count 的前提下减少 ATPG pattern 数量。
ATPG 固定使用以下配置：每个 fault 尝试一次、backtrack limit 为 3000、seed
为 14，并且不开启 STC、DTC、SCOAP 排序、transition-delay 模式或其他压缩。

## 固定的 fault 与 embedding 语义

本功能直接使用现有的逻辑折叠 fault catalog 和 embedding，不重新生成 fault，
也不修改 fault collapsing 规则。

- XOR 和多输入门拆分产生的 helper 只能作为物理执行位置和 embedding anchor，
  helper 名称绝不能成为 `fault_id`。
- 每个动作必须是 companion fault map 中一个完整的 `fault_id`。
- 每次提交的 permutation 中，每个 catalog fault 必须恰好出现一次。
- GI/GO 映射、逻辑 XOR 输入 fault、重复的 `@dup` 记录和等价 fault 数量均保持
  当前行为。
- 每个 fault 必须具有一条由现有预训练 DeepGate2 流程生成的 257 维
  embedding。scorer 不得为缺失 embedding 静默生成替代值，也不得回退到随机
  特征。
- 训练开始前，embedding 文件中有序的 fault ID 必须与同一份 binary BENCH
  和 fault map 生成的 PODEM catalog ID 完全一致。

训练集使用全部 16 个内置 binary 电路：

```text
c1355, c1908, c2670, c3540, c432, c499, c5315, c6288, c7552,
s13207_scan, s15850_scan, s35932_scan, s38417_scan, s38584_scan,
s5378_scan, s9234_scan
```

## 总体架构

系统拆分为四个职责独立的层次：

1. **PODEM 有序运行接口**：校验完整逻辑 fault permutation，将其应用到
   `flist_undetect`，运行 stuck-at ATPG，并返回结构化指标。
2. **数据集层**：加载不可变的 embedding 文件，并根据 PODEM catalog 校验其
   来源信息和 fault ID。
3. **列表级策略**：将每条 257 维 embedding 映射为一个标量分数；训练时采样
   完整 permutation，评估时生成按分数降序排列的确定性 permutation。
4. **训练器**：管理每个电路的 pattern baseline、reward 计算、共享优化器更新、
   checkpoint、评估和日志。

PODEM solver 作为强化学习环境。梯度不穿过 PODEM；完整排序完成一次 ATPG 后，
采样得到的 ranking 接收一个终局 reward。

## PODEM 接入

### 结构化运行结果

重构 `ATPG::test()` 中的 stuck-at 路径，使底层执行函数可以返回结果结构体，
同时保持现有可执行程序能够继续打印当前报告。Python 接口至少返回：

```text
pattern_count
detected_collapsed_faults
detected_equivalent_faults
uncollapsed_faults
aborted_faults
redundant_faults
podem_calls
total_backtracks
```

每次 Python 调用都创建新的 `ATPG` 实例，确保 vectors、detection 状态和列表
修改不会泄漏到下一个 episode。

### Python 排序接口

扩展 `cpp_podem`，提供等价于以下签名的函数：

```python
run_stuck_at_ordered(
    circuit_path: str,
    fault_map_path: str,
    ordered_fault_ids: list[str],
    backtrack_limit: int = 3000,
    seed: int = 14,
) -> dict
```

该函数初始化映射后的逻辑 fault catalog，校验 `ordered_fault_ids` 是 catalog ID
的精确 permutation，重排 `flist_undetect`，然后运行每 fault 一次尝试的
stuck-at ATPG。遇到未知 ID、重复 ID、缺失 ID、helper 派生 ID、错误 fault map
或不完整的空顺序时，必须在 ATPG 启动前抛出硬错误。

不新增 `-fault-order <file>` 可执行程序选项。强化学习路径通过 Python binding
在内存中传递顺序。现有可执行程序行为和原始顺序继续保留，用于 baseline 对比。

## 共享打分器

所有电路使用同一个 scorer：

```text
LayerNorm(257)
Linear(257, 256) + ReLU
Linear(256, 128) + ReLU
Linear(128, 1)
```

scorer 不使用 dropout，从而保证确定性评估稳定。必要时可以分块处理 fault，
但最终必须按 catalog 顺序为每个 fault 生成一个标量。推理时按分数从大到小
排序；分数完全相同时按原始 catalog 行号排序，保证结果确定。

所有电路更新同一套模型参数。第一版不设置电路专属输出 head，也不把 circuit
identity 加入输入特征。

## 列表级策略

对于 fault 分数 `s`，先在每个电路内对分数进行中心化，再除以温度 `tau` 得到
logits。训练时采样相互独立的 Gumbel noise，并执行以下排序：

```text
permutation = argsort(logits + gumbel_noise, descending=True)
```

该操作采样一个 Plackett-Luce permutation。对于采样结果 `p`，根据未加入噪声
的 logits 计算其 log probability：

```text
log P(p) = sum_t(logit[p[t]] - logsumexp(logit[p[t:]]))
```

使用反向 `logcumsumexp`，使排序后的计算复杂度为线性，而不是相对于 fault 数量
的平方复杂度。在合并多个电路前，用每个 permutation 的 fault 数量除其 log
probability；否则大电路会仅仅因为列表项更多而支配优化器。

训练阶段使用 Gumbel noise 进行探索。评估和导出排序时不加入噪声，直接使用
scorer 原始分数。温度可配置并写入 checkpoint；默认从 1.0 开始，在配置的训练
轮数内指数衰减，最低为 0.1。

## 基线与奖励

每个电路维护两个用途不同的值：

- `previous_pattern_count` 是计算 ATPG reward 的环境 baseline，初始值来自原始
  catalog 顺序。
- `reward_ema` 是仅用于降低策略梯度方差的控制变量，不改变日志中报告的 reward。

对于 coverage 有效的 episode：

```text
raw_reward = previous_pattern_count - current_pattern_count
previous_pattern_count = current_pattern_count
```

pattern count 减少时 reward 为正，增加时 reward 为负。策略 advantage 为：

```text
advantage = (raw_reward - reward_ema) / max(native_pattern_count, 1)
```

使用 native pattern count 作为分母，使 16 个规模不同的电路具有可比较的影响，
但不改变日志中记录的原始 pattern-count reward。

`reward_ema` 初始值为 0。计算 advantage 时使用当前保存的 EMA；optimizer step
完成后，再使用衰减系数 0.9 和本轮 raw reward 更新 EMA。coverage penalty 也要
参与 EMA 更新。advantage 作为已 detach 的常量，梯度只通过 permutation log
probability 传播。

每个电路还保存 `required_detected_equivalent_faults`。它从原始顺序运行结果
初始化；确定性模型评估检测到更多 fault 时才提高该值。随机 episode 不能提高
这个门槛，否则一次难以复现的随机排序可能使所有确定性 best checkpoint 失效。
如果某个 episode 检测到的 fault 少于当前要求，则该 episode 无效，并获得以下惩罚：

```text
raw_reward = -max(native_pattern_count,
                  previous_pattern_count,
                  current_pattern_count,
                  1)
```

episode 不更新 required detection count；无效 episode 也不更新
`previous_pattern_count`，防止 learner 通过牺牲 coverage 来减少 pattern count。

每轮的 REINFORCE 目标为：

```text
loss = -mean_circuit(advantage[c] * log_probability[c] / fault_count[c])
```

gradient norm 进行裁剪，默认上限为 1.0。优化器使用 Adam，默认 learning rate
为 `1e-4`。这些默认值必须是可配置项，不能成为训练循环中的隐藏常量。

## 训练轮次

一轮表示全部 16 个电路各参与一次：

1. 冻结一份共享 scorer snapshot，用于本轮数据采集。
2. 为每个电路的 fault 打分并采样一个完整 permutation。
3. 每个电路使用该 permutation 运行一次有序 stuck-at ATPG episode。
4. 记录 solver 指标并计算每个电路的 reward。
5. 使用冻结的 snapshot、采样 permutation 和缓存 embedding 重新计算 policy log
   probability。这样不需要在耗时较长的 ATPG 运行期间保留 autograd graph。
6. 聚合全部 16 个电路的 policy loss，并执行一次共享 optimizer step。
7. 更新有效的环境 baseline、reward EMA、RNG 状态和轮次日志。
8. 以原子方式保存可恢复的 latest checkpoint。

首次训练前，使用原始 catalog 顺序为每个电路运行一次 ATPG，以建立 pattern 和
detection baseline。这 16 次 baseline 运行不执行策略更新。

第一版按顺序执行 ATPG episode。在验证确定性等价和进程隔离前，不加入并行
solver worker，因为预计主要耗时来自 ATPG，而不是 scorer inference。

## 确定性评估与模型选择

按照可配置间隔进行评估，默认每 10 个训练轮次一次。评估时不加入 Gumbel noise，
为所有 fault 打分并按降序排列，然后对全部 16 个电路各运行一次 ATPG。只有当
每个电路都满足当前 required detected-equivalent-fault count 时，候选 checkpoint
才有资格参与最优模型比较。

合格 checkpoint 按 16 个电路确定性 pattern count 总和比较，越小越好。总和
相同时，依次使用以下规则：

1. detected equivalent fault 总数更多。
2. PODEM call 总数更少。
3. total backtrack 更少。
4. 训练轮次更早。

同时保留 `latest` 和 `best` checkpoint。最终报告优先来自对 `best` 进行的一次
全新确定性评估，不能直接采用带探索噪声的训练 episode 结果。如果训练结束时
没有 coverage 合格的 best，则对 `latest` 做确定性评估并输出 coverage 缺口，
训练命令正常结束，`best.pt` 仍保持不可用标记。

## 文件、配置与恢复

新增训练 manifest，列出全部 16 个电路以及每个电路的 binary BENCH、fault map、
embedding NPZ 和 embedding metadata JSON。所有路径相对于 manifest 所在目录解析。
程序启动时校验全部文件存在，并确保 catalog ID、行数、特征维度、circuit digest、
fault-map digest 和 checkpoint 来源信息一致。

Python package 按职责划分：

```text
fault_order_rl/
  data.py          manifest 和 embedding 校验
  model.py         共享 fault scorer
  policy.py        Gumbel/Plackett-Luce 排序和 log probability
  environment.py   cpp_podem 有序运行封装及结果校验
  trainer.py       baseline、轮次、reward、更新和评估
  checkpoint.py    原子保存/加载和 RNG 恢复
  __main__.py      train 和 evaluate 命令
```

公开命令为：

```powershell
python -m fault_order_rl validate --manifest configs/all_benchmarks.json
python -m fault_order_rl train --manifest configs/all_benchmarks.json --rounds 100 --output runs/shared_scorer
python -m fault_order_rl train --resume runs/shared_scorer/latest.pt
python -m fault_order_rl evaluate --checkpoint runs/shared_scorer/best.pt
```

初始实验默认训练 100 轮，但这不是验收阈值。`validate` 只执行 catalog、feature
和来源校验，不运行 ATPG。新建训练时，`train` 负责收集 native baseline；使用
`--resume` 时，从 checkpoint 恢复 manifest 和配置。

checkpoint 包含模型和优化器状态、已完成轮次、温度状态、Python/NumPy/PyTorch
RNG 状态、每个电路的 pattern baseline、required detection count、reward EMA、
manifest digest、embedding 来源信息和 best evaluation summary。恢复时如果 manifest
或 embedding 不兼容，必须拒绝继续。

写入仅追加的 JSONL episode/evaluation 日志和精简 JSON summary。每个 episode
记录 circuit、round、seed、temperature、pattern count、detection 指标、raw reward、
advantage、solver 工作量、耗时和 checkpoint identity。fault permutation 使用
catalog 行号整数数组保存在压缩文件中，不在 JSONL 中重复写入 fault ID 字符串。

## 失败处理

- embedding 缺失或不兼容时，必须在运行 baseline ATPG 前失败。
- 错误 permutation 必须在 solver 执行前失败，且不能成为训练样本。
- solver 异常、非有限 score/loss 或错误结果指标会中止当前轮，不修改模型参数或
  baseline。
- 只有全部 16 个 episode 和 optimizer step 成功后，才写入 latest checkpoint。
- 训练中断后从最后一个完整轮次恢复，并还原 RNG 状态，确保下一个采样
  permutation 可复现。
- coverage 无效的 episode 是有效的负向训练样本，不作为程序崩溃处理。

## 测试

### C++ 与 binding 测试

- 通过新接口运行 native catalog 顺序时，结果与现有 stuck-at 指标一致。
- reversed 顺序和指定的已知 permutation 被精确应用。
- 缺失、重复、未知、helper 派生及跨电路 fault ID 在 ATPG 前被拒绝。
- 小型 mapped 电路的结果计数与现有 executable 一致。
- 使用全新实例和 seed 14 重复调用时结果确定。
- 有序运行中始终关闭 STC、DTC、SCOAP 和 transition-delay 路径。

### Python 单元测试

- scorer 为每条 257 维 fault embedding 输出一个有限标量。
- 确定性排序使用 catalog 行号处理相同分数。
- 固定 RNG seed 时，Gumbel sampling 返回精确且可复现的 permutation。
- 线性时间 log-probability 与小规模直接 Plackett-Luce 计算一致。
- reward、baseline 更新、coverage penalty、EMA advantage 和电路等权重符合本文
  公式。
- checkpoint resume 能恢复模型、优化器、baseline 和采样顺序。
- manifest 校验能发现 ID 顺序和来源信息不匹配。

### 端到端测试

- 一个包含多输入门/XOR 映射的微型电路能够完成 baseline、一个训练轮次、
  checkpoint、resume 和确定性评估，且不产生 helper fault ID。
- 一个包含至少两个真实 benchmark 的短 smoke run 能证明一次 optimizer update
  同时接收来自两个电路的 loss。
- 完整实验开始前，全部 16 份文件通过校验，并在固定 ATPG 配置下记录全部 16 个
  native baseline。

## 验收标准

满足以下条件后，才可以开始完整训练：

1. 全部 16 个电路的 catalog 与 embedding 完全匹配。
2. 有序 ATPG 只接受完整逻辑 fault permutation，并返回结构化确定性指标。
3. 一轮能够运行全部 16 个电路、计算指定 reward，并执行一次共享 scorer 更新。
4. 所有训练和评估运行均不开启 STC、DTC、SCOAP、compression 或 TDF。
5. coverage 下降的排序获得指定惩罚，且不能成为 best checkpoint。
6. 中断的训练能够在轮次边界恢复，并复现后续采样。
7. best checkpoint 能为每个电路的每个 fault 导出确定性 score 和 rank，并报告
   每个电路及总体 ATPG 结果。

## 不在本次范围内

- 修改 fault 创建、折叠、helper 映射或 embedding 语义。
- 新增文本文件形式的 `-fault-order` CLI 选项。
- 完全照搬 DeepTPI 的逐步 DQN 动作选择。
- STC、DTC 和 transition-delay ATPG。
- 联合微调预训练 DeepGate2 encoder。
- 第一版实现并行或分布式 ATPG。
