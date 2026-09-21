# 动态故障排序 PPO：训练、恢复与评估

`fault_order_rl` 使用共享 Actor-Critic 为每个动态 remaining fault set 生成完整排名。
排名第一的故障作为 Primary，其余排名原样传给 C++ 作为 DTC secondary 优先级。

## 固定实验协议

- 训练 5 轮，每个 circuit rollout 后立即执行 4 次 PPO epoch。
- `gamma=1.0`、`GAE lambda=0.95`、`clip=0.2`。
- value loss 系数 `0.5`，entropy 系数 `0.01`，Adam 学习率 `1e-4`。
- step shaping `alpha=0.1`，覆盖短缺惩罚 `beta=10.0`。
- Primary backtrack limit 固定为 `100`；DTC secondary limit 固定为 `50`。
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

恢复不能改变目标轮数或训练配置。Schema 1–3 与新模型、奖励及进度语义不兼容，必须
重新训练。

## 奖励和 PPO

每步 shaping：

```text
(-pattern_increment + 0.1 * newly_detected_equivalent_faults) / InitialEqv
```

完整 episode 的目标回报：

```text
覆盖不短缺：1 - patterns_after_stc / InitialEqv
覆盖有短缺：-10 * coverage_shortfall / InitialEqv
```

最后一步加入 terminal correction，使所有 step reward 的总和严格等于目标回报，然后
计算 GAE。策略概率只计算 C++ 实际执行的排名前缀：Primary 加上
`dtc_attempted_fault_ids`；未执行的 ranking 后缀不进入 PPO ratio。

## Checkpoint 与 artifacts

- `latest.pt`：schema 4，可恢复；每完成一个 training circuit 就原子提交。
- `best.pt`：独立验证集上最佳模型，不含 optimizer 和 RNG。
- `final.pt`：第 5 轮最终模型，不含 optimizer 和 RNG；可能与 best 来自不同轮。
- `rounds/round-NNNNNN/circuit-NNNNNN.{json,npz}`：requested ranking、executed
  prefix、奖励、GAE 与 PPO 指标。
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

不传 `--manifest` 时使用 checkpoint 保存的 validation manifest。评估采用 score 降序；
同分时按原始 catalog row 升序，输出逐电路 comparison CSV、summary、ranking NPZ 和
trajectory JSONL。
