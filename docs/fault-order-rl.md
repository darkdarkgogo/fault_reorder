# BFS 筛选 Ranked-DTC PPO：训练、恢复与评估

`fault_order_rl` 使用共享 Actor-Critic 为每个动态 remaining fault set 计算一次分数。
Primary 从完整 remaining set 中选择；DTC secondary 候选由 C++ 从 unknown PO 沿全 `U`
逻辑锥反向 BFS 得到。RL 只使用同一次 Primary 分数对每个 BFS batch 排序。

## 固定实验协议

- 训练 5 轮，每个 circuit rollout 后立即执行 4 次 PPO epoch。
- `gamma=1.0`、`GAE lambda=0.95`、`clip=0.2`。
- value loss 系数 `0.5`，entropy 系数 `0.01`，Adam 学习率 `1e-4`。
- step shaping `alpha=0.1`，覆盖短缺惩罚 `beta=10.0`。
- Primary backtrack limit 固定为 `100`；DTC secondary limit 固定为 `50`。
- `ncktin <= 32` 时每个 unknown PO 的 `select_fault_try=15`，否则为 `100`。
- PODEM seed 固定为 `14`，DTC 与 STC 均启用。
- 默认独立验证集为 `configs/anchor_validation_6.json`。

训练集和验证集的 manifest 必须不同，且 artifact provenance 不能重叠。验证集不参与
梯度更新，只用于根据 `(覆盖短缺总量, STC 后向量总数)` 选择 `best.pt`。

## 构建与训练

Windows：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m fault_order_rl train `
  --manifest configs\anchor_train_1024.json `
  --validation-manifest configs\anchor_validation_6.json `
  --output runs\anchor_train_1024
```

Linux：

```bash
./scripts/setup_anchor_linux.sh
./scripts/run_anchor_linux.sh runs/anchor_train_1024
```

恢复固定的五轮训练：

```bash
python3 -m fault_order_rl train \
  --resume runs/anchor_train_1024/latest.pt
```

恢复不能改变目标轮数或训练配置。Schema 1–4 与当前 BFS-filtered action mask、概率及
回滚协议不兼容，必须重新训练。

## Primary、DTC 与缓存分数

每个 Primary step 的执行流程为：

```text
remaining set F_t
  -> build_dynamic_features(F_t)
  -> Actor-Critic forward 一次
  -> 从 F_t 选择 Primary
  -> C++ Primary PODEM
  -> unknown PO 反向 BFS 得到 C_t,1
  -> 用缓存 scores_t[C_t,1] 排序
  -> 必要时重新 BFS 得到下一批候选并继续复用 scores_t
  -> fault simulation
```

Heuristic/native baseline 不提交完整 batch，而是复用原 TDF 的 lazy `q_wire`/`q_fault`
流程：`q_fault` 为空时才展开下一个 `U` wire，发现 eligible faults 后立即逐个尝试。
`StuckAtSession.step()` 与 `run_stuck_at_ordered()` 都使用这条 baseline 路径。RL 训练
仍先取得预算内完整 batch，再在该 batch 内按 Plackett-Luce 分布采样；确定性评估按
缓存 score 降序，同分按 catalog row 升序。

每次 secondary 尝试完成后，无论 TRUE、FALSE 或 MAYBE，都恢复 accepted PI cube，
重新建立 canonical good-circuit implication，再检查目标 PO。PO 不再为 `U` 时立即停止
当前 ranking，未执行尾部不进入动作概率。

回滚不再逐 candidate 保存完整 wire snapshot。持久状态只保存 accepted PI cube；内部
wire 和 transient flags 通过 fault-free implication 确定性恢复。Preserved-fault 的临时
injection 检查后也必须恢复 good-circuit 状态。

## 奖励与 PPO

每步 shaping：

```text
(-pattern_increment + 0.1 * newly_detected_equivalent_faults) / InitialEqv
```

完整 episode 的目标回报：

```text
覆盖不短缺：1 - patterns_after_stc / InitialEqv
覆盖有短缺：-10 * coverage_shortfall / InitialEqv
```

最后一步加入 terminal correction，使所有 step reward 总和严格等于目标回报，然后计算
GAE。一个 step 的 joint log probability 是 Primary categorical 项，加上各 BFS batch
实际执行前缀的 Plackett-Luce 项。Rollout 和 PPO replay 中每个 transition 都只调用一次
模型。

## Checkpoint 与 artifacts

- `latest.pt`：schema 5，可恢复；每完成一个 training circuit 就原子提交。
- `best.pt`：独立验证集上的最佳模型，不含 optimizer 和 RNG。
- `final.pt`：第 5 轮最终模型，不含 optimizer 和 RNG；可能与 best 来自不同轮。
- `rounds/round-NNNNNN/circuit-NNNNNN.{json,npz}`：Primary rows、各 BFS batch 的
  candidate/requested/executed/embedded rows、奖励、GAE 与 PPO 指标。
- `validation/round-NNNNNN/`：每轮确定性验证的排名、轨迹和 summary。

若进程在某个 circuit 中断，恢复从上一个已提交 circuit 的下一索引继续，并恢复模型、
optimizer 与 Python/NumPy/Torch RNG。

## 评估

```bash
python3 -m fault_order_rl evaluate \
  --checkpoint runs/anchor_train_1024/best.pt \
  --manifest configs/anchor_validation_6.json \
  --output runs/anchor_train_1024/evaluation-best

python3 -m fault_order_rl evaluate \
  --checkpoint runs/anchor_train_1024/final.pt \
  --manifest configs/anchor_validation_6.json \
  --output runs/anchor_train_1024/evaluation-final
```

不传 `--manifest` 时使用 checkpoint 保存的 validation manifest。评估输出逐电路
comparison CSV、summary、ranking NPZ 和 trajectory JSONL。
