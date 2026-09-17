# Fault Reorder 当前算法说明

> 对应代码快照：2026-09-15，commit `e46cd8e`  
> 核心实现：`fault_order_rl/` 与 `PODEM/src/python_bindings.cpp`  
> 当前训练入口：`scripts/run_anchor_linux.sh`

## 1. 文档目的

本文说明仓库当前实现的 fault reorder 算法，包括：

- fault 表示与 257 维 DeepGate2 embedding；
- 共享神经打分器；
- Gumbel/Plackett–Luce 完整列表采样；
- PODEM 终局反馈、覆盖约束奖励与 REINFORCE 更新；
- minibatch 训练、确定性评估、best 选择和断点恢复；
- 当前算法的边界、复杂度与输出。

这里的 reorder 不是修改 fault collapsing，也不是在 PODEM 内逐步选择动作。模型一次为一个电路的全部折叠 stuck-at fault 打分，产生一个完整 permutation，再把该顺序整体交给 PODEM。排序可能改变哪些 fault 先成为 primary target，从而改变 fault simulation 的顺带检测效果、最终 pattern 数、PODEM 调用数和回溯数。

## 2. 优化目标与约束

对于电路集合 \(\mathcal C\)，电路 \(c\) 的折叠 fault catalog 为

\[
F_c=(f_{c,1},f_{c,2},\ldots,f_{c,N_c}).
\]

模型学习共享参数 \(\theta\)，为每个 fault 产生分数，并据此生成完整排序 \(\pi_c\)。主要优化目标是在每个电路的“已解决非折叠 fault 数”不低于原始 catalog 顺序的前提下，减少确定性排序产生的 test pattern 总数：

\[
\min_\theta \sum_{c\in\mathcal C}
\operatorname{Patterns}\bigl(c,\pi_c(\theta)\bigr).
\]

当前覆盖定义为：

\[
\operatorname{Covered}_c
=\operatorname{DetectedEquivalent}_c
+\operatorname{RedundantEquivalent}_c.
\]

\[
\operatorname{FaultCoverage}_c
=\frac{\operatorname{Covered}_c}{\operatorname{UncollapsedTotal}_c}.
\]

因此，当前代码中的 coverage 更准确地说是“resolved coverage”：已检测和已证明 redundant 的等价 fault 都计入覆盖。训练与 best 选择均要求：

\[
\operatorname{Covered}_c(\pi_c)
\geq \operatorname{Covered}_c(\pi_c^{native}),
\quad \forall c.
\]

约束是逐电路检查，而不是只比较整个数据集的覆盖总和，所以不能用某个电路增加的覆盖抵消另一个电路的覆盖下降。

## 3. 系统总览

```text
原始 BENCH ───────────────┐
                         ├─ PODEM catalog ─ fault ID / 等价数校验
AIG BENCH + AIG map ─────┤
                         └─ DeepGate2 hf ─ 257 维 fault embedding
                                                │
                                                ▼
                                  共享 MLP scorer sθ(f)
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         │ 训练：加 Gumbel 噪声，采样完整 permutation │
                         │ 评估：按原始 score 降序，行号稳定破同分     │
                         └──────────────────────┬──────────────────────┘
                                                ▼
                                  ordered stuck-at PODEM
                                                │
                         pattern / coverage / calls / backtracks
                                                │
                                                ▼
                         reward + REINFORCE / best checkpoint 选择
```

代码职责如下：

| 模块 | 职责 |
|---|---|
| `fault_order_rl/data.py` | manifest、catalog、embedding 与来源一致性校验 |
| `fault_order_rl/model.py` | 所有电路共享的 fault scorer |
| `fault_order_rl/policy.py` | Gumbel 采样、确定性排序、Plackett–Luce log probability |
| `fault_order_rl/environment.py` | Python 到 ordered PODEM 的封装与指标校验 |
| `fault_order_rl/trainer.py` | baseline、奖励、minibatch 更新、评估、best 选择 |
| `fault_order_rl/checkpoint.py` | 原子 checkpoint 与 RNG 状态保存/恢复 |
| `PODEM/src/python_bindings.cpp` | catalog 和完整 fault permutation 的 C++ 接口 |
| `PODEM/src/atpg.cpp` | 固定协议下的单次 stuck-at ATPG 执行 |

## 4. Fault catalog 与输入特征

### 4.1 Fault 粒度

模型的一个排序项对应一个 PODEM 折叠逻辑 fault。每项至少具有：

- 唯一 `fault_id`；
- GO 或 GI 类型；
- SA0 或 SA1 极性；
- `eqv_fault_num`，即该折叠 fault 代表的非折叠 fault 数；
- 与 embedding 行严格对齐的 catalog 行号。

每次送入 PODEM 的顺序必须是 catalog 的精确 permutation：长度相等、无重复、无未知 ID、无缺失 ID。模型只改变顺序，不创建、删除或重新折叠 fault。

### 4.2 Anchor-preserving 双路径数据

当前主训练 manifest `configs/anchor_train_1024.json` 使用 1024 个训练电路，每个条目包含两条用途不同的路径：

- `bench`：原始 BENCH，用于 PODEM 建立 catalog 和执行 ATPG；
- `aig_bench` 与 `aigmap`：用于构造和校验 DeepGate2 图及 fault anchor；
- `embeddings`：与原始 fault catalog 对齐的 257 维向量；
- `metadata`：fault ID、等价数、anchor、图摘要和推理来源。

该数据不使用 `faultmap`。Python 层向 C++ 传空字符串，使 PODEM 直接采用原始 BENCH 的 fault。仓库同时兼容旧式 binary BENCH + V2/V3 fault map 数据，但两种数据语义会被严格区分。

### 4.3 257 维 embedding

DeepGate2 encoder 当前被冻结，不参与 RL 微调。每个图节点提供 128 维 functional embedding \(h_f\)。fault 特征为：

\[
x_f=[h_f^{gate}\,\Vert\,h_f^{connected}\,\Vert\,sa],
\quad x_f\in\mathbb R^{257}.
\]

其中：

- GI：`[receiver_gate_hf(128), connected_signal_hf(128), sa_value(1)]`；
- GO：`[driver_gate_hf(128), driver_gate_hf(128), sa_value(1)]`；
- `sa_value=0` 表示 SA0，`sa_value=1` 表示 SA1。

数值特征不包含 structural embedding、电路 ID 或 pin-position one-hot。pin、GI/GO、等价数等信息保留在 metadata 中，但不直接输入 scorer。

### 4.4 启动前校验

在任何 baseline ATPG 之前，代码会检查：

1. manifest 版本、路径和电路名合法；
2. 所有输入文件存在；
3. embedding 形状严格为 `[N, 257]`，非空且全部有限；
4. embedding、metadata 和 PODEM catalog 的 fault ID 顺序完全一致；
5. 三者的 `eqv_fault_num` 完全一致，且其总和等于 uncollapsed total；
6. BENCH、AIG BENCH、AIG map、图和 embedding 来源摘要一致；
7. DeepGate2 来源、官方 checkpoint 哈希和推理 backend 符合固定来源。

该校验阻止错位 embedding、陈旧派生文件或不同 fault catalog 被静默混入训练。

## 5. 共享神经打分器

所有训练和验证电路共用一个 `FaultScorer`：

```text
输入 x_f ∈ R^257
  → LayerNorm(257)
  → Linear(257, 256) + ReLU
  → Linear(256, 128) + ReLU
  → Linear(128, 1)
  → 标量 score sθ(f)
```

模型没有 dropout，也没有电路专属 head。若一个电路有 \(N_c\) 个 fault，则模型输出：

\[
s_c=(s_{c,1},\ldots,s_{c,N_c})\in\mathbb R^{N_c}.
\]

## 6. 列表级排序策略

### 6.1 Logit 与温度

训练时先在当前电路内部中心化 score，再除以温度 \(\tau\)：

\[
z_{c,i}=\frac{s_{c,i}-\bar s_c}{\tau}.
\]

温度按轮次指数衰减：

\[
\tau_r=\tau_{start}
\left(\frac{\tau_{min}}{\tau_{start}}\right)^q,
\]

\[
q=\min\left(
\frac{\max(r-1,0)}{\max(R_\tau-1,1)},1
\right).
\]

默认 \(\tau_{start}=1.0\)、\(\tau_{min}=0.1\)、\(R_\tau=100\)。中心化不改变同一列表内的相对概率，但改善数值表达；温度越低，策略越偏向高分 fault。

### 6.2 Gumbel-Top-k 采样

对每个 fault 独立采样：

\[
u_i\sim U(0,1),\qquad
g_i=-\log(-\log u_i).
\]

完整训练顺序为：

\[
\pi_c=\operatorname{argsort}_{desc}(z_c+g).
\]

这等价于从 Plackett–Luce 分布采样一个完整 permutation，不是对 fault 逐个执行 ATPG 后再决定下一个动作。一次 episode 只有一个终局 reward。

### 6.3 Plackett–Luce log probability

对已采样顺序 \(\pi=(\pi_1,\ldots,\pi_N)\)：

\[
\log P_\theta(\pi)
=\sum_{t=1}^{N}
\left[z_{\pi_t}
-\log\sum_{j=t}^{N}\exp(z_{\pi_j})\right].
\]

实现先按 \(\pi\) 排列 logits，再使用反向 `logcumsumexp` 计算所有分母，因此给定 permutation 后的 log probability 计算为 \(O(N)\)，而不是 \(O(N^2)\)。排序本身仍为 \(O(N\log N)\)。

### 6.4 确定性排序

评估和导出不加噪声，也不使用训练温度，直接按原始 score 降序排列。score 相同时以 catalog 原始行号升序破同分：

\[
\pi_c^{eval}=\operatorname{lexsort}(\text{row},-s_c).
\]

因此同一模型、数据和 PODEM 二进制的评估顺序稳定可复现。

## 7. Ordered PODEM 环境

### 7.1 固定执行协议

当前 RL 环境固定为：

| 配置 | 当前值 |
|---|---:|
| fault 模型 | stuck-at |
| 每个 fault 最大尝试次数 | 1 |
| backtrack limit | 5000 |
| PODEM seed | 14 |
| dynamic test compression | 关闭 |
| static test compression | 关闭 |
| SCOAP fault order | 关闭 |
| transition-delay 模式 | 关闭 |
| test vector 文本输出 | 关闭 |

训练命令中的 `--seed` 只控制模型初始化、batch shuffle 和策略采样；PODEM 始终使用 seed 14。`backtrack_limit` 虽存在于 checkpoint 配置中，但当前实验强制必须为 5000。

物理 XOR/EQV 不在该 PODEM backtrace 支持范围内，ordered 路径会在执行前拒绝这类门；用于 PODEM 的 BENCH 必须先展开为受支持的 AND/OR/NAND/NOR/NOT/BUF 结构。

### 7.2 顺序如何影响 ATPG

PODEM 按 permutation 寻找下一个尚未尝试的 primary fault。对每个 primary fault：

1. `podem()` 成功：生成一个 pattern，随后对该 pattern 做 fault simulation；该 pattern 可能顺带检测后续多个 fault；
2. `podem()` 返回不可测：把该 fault 标为 redundant，并累加其等价 fault 数；
3. 达到限制而不确定：计为 aborted；
4. 将该 primary fault 标为已尝试，再从重排后的列表中找下一个未尝试 fault。

因此，排序优化的是“哪些 fault 更早被直接求解”。早期 pattern 的顺带检测会改变后续仍需调用 PODEM 的 fault 集合，最终影响 pattern count、calls 和 backtracks。

环境返回并校验以下指标：

```text
pattern_count
detected_collapsed_faults
detected_equivalent_faults
uncollapsed_faults
aborted_faults
redundant_faults
redundant_equivalent_faults
podem_calls
total_backtracks
```

Python 层再派生 `covered_equivalent_faults` 和 `fault_coverage`。

## 8. Baseline、奖励与 advantage

### 8.1 Native baseline

新训练开始时，每个电路先用 catalog 原始顺序运行一次 PODEM，并保存：

- \(P_c^{native}\)：原始顺序的 pattern count；
- \(C_c^{native}\)：原始顺序的 covered equivalent faults；
- \(P_c^{prev}\)：上一个 coverage 有效 episode 的 pattern count，初始为 \(P_c^{native}\)；
- \(E_c\)：reward 的指数移动平均，初始为 0。

注意：\(P_c^{prev}\) 是“上一个有效 episode”，不是历史最好 pattern count。

完成 native baseline 后，程序还会对随机初始化 scorer 做一次 round 0 的确定性评估。若它逐电路满足 native coverage，也可以成为初始 best。

### 8.2 Coverage 有效的 episode

若本次结果 \(C_c\ge C_c^{native}\)，则：

\[
r_c=P_c^{prev}-P_c,
\qquad P_c^{prev}\leftarrow P_c.
\]

pattern 减少时奖励为正，增加时为负。

### 8.3 Coverage 无效的 episode

若 \(C_c<C_c^{native}\)，则使用强负奖励：

\[
r_c=-\max(P_c^{native},P_c^{prev},P_c,1).
\]

此时不更新 \(P_c^{prev}\)，避免模型通过牺牲 resolved coverage 获得更少 pattern 的表面收益。native coverage 门槛也是固定的，不会被随机 episode 或后续评估提高。

### 8.4 EMA advantage

使用更新前的 reward EMA 计算 advantage，并以 native pattern count 归一化：

\[
A_c=\frac{r_c-E_c}{\max(P_c^{native},1)}.
\]

随后更新 EMA：

\[
E_c\leftarrow \beta E_c+(1-\beta)r_c,
\]

默认 \(\beta=0.9\)。coverage penalty 同样进入 EMA。EMA 只作为降方差控制变量，不参与环境结果和 best 比较。

## 9. REINFORCE 与 minibatch 更新

对一个包含 \(B\) 个电路的 batch，损失为：

\[
\mathcal L
=-\frac{1}{B}\sum_{c=1}^{B}
A_c\frac{\log P_\theta(\pi_c)}{N_c}.
\]

除以 \(N_c\) 是为了避免 fault 较多的电路仅因 permutation 更长而产生更大的 log probability 量级；再除以 \(B\)，使同一 batch 内的电路等权。

当前实现的 batch 流程是：

1. 用 batch 开始时的同一参数快照，在 `no_grad` 下为 batch 内所有电路采样顺序；
2. 顺序执行各电路的 PODEM episode 并计算 reward；
3. 使用相同模型参数重新前向，计算采样 permutation 的可微 log probability；
4. 累积 batch loss；
5. 将全局 gradient norm 裁剪到默认 1.0；
6. 使用 Adam 更新共享 scorer，默认 learning rate 为 \(10^{-4}\)。

不在耗时的 PODEM 执行期间保留 autograd graph。梯度不穿过 PODEM，而是通过 REINFORCE 的 \(A_c\log P_\theta(\pi_c)\) 估计传播。

### 9.1 “一轮”与 batch 的准确含义

一轮内，manifest 中每个电路恰好参与一次训练 episode：

- `batch_size=0`：使用整个 manifest 作为一个 batch，一轮更新一次；
- `batch_size>0`：每轮先随机打乱电路，再切成 batch，每个 batch 更新一次；
- 当前 `run_anchor_linux.sh` 默认 `batch_size=16`，所以 1024 个训练电路每轮形成 64 个 batch，并产生 64 次参数更新。

同一 batch 内 episode 使用同一参数快照；后续 batch 会看到前一 batch 更新后的模型。当前 1024 可被 16 整除，因此所有生产 batch 大小一致。

## 10. 训练算法伪代码

```text
输入：manifest、电路集合 C、目标轮数 R、batch size B
输出：latest.pt、best.pt、逐轮日志与确定性评估

校验 manifest、catalog、embedding、等价数和来源
初始化共享 scorer θ、Adam、RNG

for 每个电路 c:
    native[c] ← PODEM(c, catalog 原始顺序)
    state[c] ← {
        native_pattern_count = native[c].patterns,
        previous_pattern_count = native[c].patterns,
        native_covered_equivalent_faults = native[c].covered,
        reward_ema = 0
    }

report0 ← 用 θ 做确定性排序并评估所有训练电路
best ← report0 满足逐电路 native coverage 时保存，否则为空
原子写入 round 0 与 latest.pt；发布 best.pt

for round r = 1 ... R:
    保存本轮开始前 RNG
    candidate θ' ← θ 的深拷贝
    candidate optimizer ← optimizer 的深拷贝
    candidate state ← state 的深拷贝
    τ ← temperature(r)
    batches ← 全部电路（B>0 时先 shuffle 再分批）

    try:
        for batch in batches:
            for c in batch, no_grad:
                score ← scorerθ'(embedding[c])
                π[c] ← Gumbel-Top-k(score, τ)
                metrics[c] ← PODEM(c, π[c])
                reward[c], candidate state[c] ← reward_transition(...)

            清空梯度
            for c in batch:
                logP[c] ← PlackettLuceLogProb(θ', π[c], τ)
                loss[c] ← -advantage[c] × logP[c] / faults[c] / |batch|
                反向传播 loss[c]
            裁剪梯度并执行 Adam step

        if r 到达评估间隔或为最终轮:
            report ← 用 θ' 确定性评估所有训练电路
            若 report coverage 合格且 evaluation_key 更优：更新 best

        写本轮 JSONL/NPZ
        原子替换 latest.pt              # 本轮提交点
        θ, optimizer, state ← candidate
        发布可由 latest 派生的 best.pt

    except 任意异常:
        恢复本轮开始前 RNG
        保持 live model、state 与 latest.pt 不变
        抛出异常
```

## 11. 确定性评估与 best 选择

默认每 10 轮评估一次，最终轮必定评估。候选模型必须先通过逐电路 native coverage 资格检查。合格候选按以下字典序选择，越小越优：

\[
K=(P_{total},-C_{total},Calls_{total},Backtracks_{total},Round).
\]

即依次比较：

1. pattern 总数更少；
2. pattern 数相同时，covered equivalent fault 总数更多；
3. 再相同时，PODEM calls 更少；
4. 再相同时，总回溯数更少；
5. 完全相同时，较早轮次优先。

覆盖更多不会优先于 pattern 更少；它只在两个候选都逐电路达标且 pattern 总数相同时作为第二关键字。

训练结束时：

- 有 coverage 合格的 best：重新运行一次 `best.pt` 的全新确定性评估；
- 没有合格 best：`best.pt` 标记 `available=false`，改为评估 `latest.pt`，并报告 `coverage_shortfall`；
- `coverage_shortfall` 是各电路相对 native resolved coverage 的非折叠 fault 缺口之和。

独立验证可在与训练不同的 manifest 上执行。当前验证集 `configs/anchor_validation_6.json` 包含 `b12_C`、`b15_C`、`b17_C`、`b20_C`、`b21_C`、`b22_C`。外部验证会为每个电路分别重跑 native 顺序和模型顺序，并支持按电路断点续评。

## 12. Checkpoint、一致性与失败语义

`latest.pt` 是训练事务的提交点，保存：

- schema v2、run ID、已完成轮次与完整训练配置；
- 模型、Adam optimizer；
- 每个电路的 native 指标、previous pattern baseline 和 reward EMA；
- Python、NumPy、PyTorch RNG 状态；
- manifest、数据 artifact、PODEM 二进制和 PyTorch 版本摘要；
- 当前 best 的模型快照和评估报告。

`best.pt` 是从 `latest.pt` 内 best 快照派生的只读评估文件。若进程在提交 latest 后、发布 best 前中断，恢复训练时会自动修复 best；独立评估会拒绝与 latest 不一致的陈旧 best。

恢复时必须保持 manifest、artifact、PODEM 二进制和 PyTorch 版本兼容。只允许改变总目标轮数，不能改变学习率、batch size、线程数等已保存配置。

一轮中的 episode 收集、更新或评估失败时：

- 不提交 candidate 模型、optimizer 或 baseline；
- 恢复本轮开始前的 RNG；
- `latest.pt` 仍指向上一个完整轮次；
- 恢复后会从最后提交轮次重新执行。

## 13. 输出文件

```text
runs/<run>/
  latest.pt
  best.pt
  rounds/
    round-000000.jsonl
    round-000001.jsonl
    round-000001.npz
    ...
  evaluation/
    summary.json
    comparison_by_circuit.csv
    <circuit>.ranking.npz
```

- `round-NNNNNN.jsonl`：episode、reward、advantage、log probability、loss contribution、梯度 norm 和评估记录；
- `round-NNNNNN.npz`：该轮采样 permutation 的 catalog 行号（0-based）；
- `<circuit>.ranking.npz`：`fault_ids`、原始 `scores`、1-based `ranks` 和 0-based `permutation`；
- `comparison_by_circuit.csv`：native/model coverage、pattern 数、绝对与百分比变化；
- `summary.json`：checkpoint 身份、逐电路结果、资格与 coverage 缺口。

## 14. 当前默认值与生产配置

| 参数 | `TrainConfig` 默认值 | 当前生产脚本默认值 |
|---|---:|---:|
| rounds | 100 | 100 |
| learning rate | `1e-4` | 继承默认 |
| gradient clip | `1.0` | 继承默认 |
| EMA decay | `0.9` | 继承默认 |
| temperature | `1.0 → 0.1 / 100 rounds` | 继承默认 |
| evaluate every | 10 rounds | 继承默认 |
| training seed | 14 | 继承默认 |
| PODEM seed | 固定 14 | 固定 14 |
| backtrack limit | 强制 5000 | 强制 5000 |
| PyTorch threads | 1 | 1 |
| batch size | 0（全 manifest） | 16 |
| training circuits | 由 manifest 决定 | 1024 |

## 15. 复杂度与主要成本

对一个含 \(N\) 个 fault 的电路：

- MLP 打分：\(O(N)\)，常数由 257→256→128→1 的全连接层决定；
- Gumbel/确定性排序：\(O(N\log N)\)；
- Plackett–Luce log probability：排序后 \(O(N)\)；
- 内存：embedding 与中间激活约为 \(O(N)\)；
- PODEM：取决于电路结构、fault 顺序与 backtrack，通常远高于神经网络开销。

设训练电路数为 \(C\)：

- 新训练启动需要 \(C\) 次 native baseline ATPG，加 \(C\) 次 round 0 模型评估；
- 每个训练轮需要 \(C\) 次采样排序 ATPG；
- 每个评估轮额外需要 \(C\) 次确定性 ATPG；
- 参数更新次数为 \(\lceil C/B\rceil\)，`batch_size=0` 时为 1。

当前 1024/16 配置下，每轮有 1024 次训练 ATPG 和 64 次 Adam 更新；到达评估轮时再增加 1024 次 ATPG。

## 16. 算法边界与已知限制

- 只优化 fault 顺序，不修改 PODEM 搜索算法、fault collapsing 或 test compression；
- 只支持当前 ordered stuck-at、one-attempt 协议；
- reward 是完整排序执行后的终局信号，credit assignment 较粗；
- scorer 单独处理每个 fault，跨 fault 关系只通过共享参数和列表分布间接体现，没有显式 self-attention；
- 257 维特征不含电路 ID、pin one-hot、显式 SCOAP 或历史 ATPG 状态；
- `previous_pattern_count` 是移动的环境 baseline，reward 不是相对 native 或历史最优的绝对改进；
- coverage 门槛固定为 native resolved coverage，不会要求模型保留某次偶然获得的额外覆盖；
- 当前 ATPG episode 顺序执行，没有并行 solver worker；
- 训练可精确恢复依赖相同数据、PODEM 二进制、PyTorch 版本和确定性 CPU 运行环境。

## 17. 运行入口

校验数据但不运行 ATPG：

```bash
python3 -m fault_order_rl validate \
  --manifest configs/anchor_train_1024.json
```

按当前生产默认配置训练或自动续训：

```bash
./scripts/run_anchor_linux.sh
```

显式新建训练：

```bash
python3 -m fault_order_rl train \
  --manifest configs/anchor_train_1024.json \
  --output runs/anchor_train_1024 \
  --rounds 100 \
  --threads 1 \
  --batch-size 16
```

分别验证 best 与 latest：

```bash
./scripts/evaluate_anchor_linux.sh runs/anchor_train_1024
```

## 18. 一句话总结

当前 fault reorder 是一个共享的列表级策略梯度系统：它以冻结的 DeepGate2 fault embedding 为输入，用 MLP 为完整 fault catalog 打分，通过 Gumbel/Plackett–Luce 采样排序，以固定协议 PODEM 的 pattern 数作为主要反馈，并用逐电路 native resolved coverage 作为硬资格约束，最终输出可复现的确定性 fault ranking。

