# 共享故障排序强化学习

`fault_order_rl` 使用已有的 257 维 DeepGate2 故障 embedding，为完整故障列表
采样排序，再调用有序 stuck-at PODEM。所有电路共享一个 scorer；一轮包含 manifest
中每个电路各一次 episode，并进行一次等电路权重的 REINFORCE 更新。

## 本机运行

Linux 脚本使用 `PATH` 中的 `python3`，不会联网、执行 `pip install`、创建 Conda
环境或调用 sudo。仓库已经包含训练 embedding 和 pybind11 头文件。服务器需要
已有 PyTorch、NumPy、setuptools、C++ 编译器和 Python 开发头文件。

```bash
sudo apt update
sudo apt install build-essential python3-dev
```

运行一次构建脚本：

```bash
chmod +x scripts/setup_linux.sh scripts/run_linux.sh
./scripts/setup_linux.sh
```

这个脚本只执行下面这一条 Python 构建命令，生成当前 Python 3 对应的 Linux
`cpp_podem*.so`：

```bash
python3 PODEM/setup.py build_ext --inplace
```

训练脚本默认运行 100 轮并写入 `runs/shared_scorer`：

```bash
./scripts/run_linux.sh
```

也可以指定轮数和输出目录：

```bash
./scripts/run_linux.sh 200 runs/experiment-200
```

断点续训；可选的第二个参数表示总目标轮数：

```bash
python3 -m fault_order_rl train \
  --resume runs/shared_scorer/latest.pt --rounds 200
```

重新评估 best 并导出排名：

```bash
python3 -m fault_order_rl evaluate \
  --checkpoint runs/shared_scorer/best.pt
```

`configs/all_benchmarks.json` 包含 16 个内置二值电路；`smoke_benchmarks.json`
只包含 c432 和 c499。路径均相对于 manifest 所在目录，并指向仓库中的
`training_data/fault-order-embeddings`。服务器无需 DeepGate2 源码、预训练权重或
联网下载。缺少这些文件说明仓库没有更新完整，应重新执行 `git pull origin main`。

物理 XOR/EQV 门需要先展开为 PODEM 支持的门；内置二值电路已经使用这种形式。
逻辑 XOR 输入 fault 仍通过 V3 map 保留。直接把未展开的 XOR 门交给有序
PODEM 会报错，避免旧 backtrace 实现的未定义行为。

## 奖励与模型选择

- 有效 episode 的原始奖励为 `previous_pattern_count - current_pattern_count`。
- 检测等价故障数少于当前要求时，奖励为负的
  `max(native_pattern_count, previous_pattern_count, current_pattern_count, 1)`，
  并保留之前的 pattern baseline 和检测要求。
- advantage 使用更新前的 reward EMA，并除以 native pattern 数。每个电路的
  排序 log probability 除以自身 fault 数，然后对电路等权平均。
- 评估采用分数降序和稳定行号破同分，不加入探索噪声。best 必须在每个电路上
  满足当前检测要求，再比较总 pattern 数、检测数、PODEM 调用数和回溯数。
- 新观察到的更高检测数会提高门槛；此前不再达标的 best 会失效。没有合格模型时，
  `best.pt` 明确标记不可用，评估报错，可从 `latest.pt` 继续训练。

PODEM 固定 seed=14、backtrack limit=3000、每 fault 尝试一次；STC、DTC、SCOAP
和 TDF 不启用。默认 Adam 学习率为 1e-4，梯度裁剪为 1，EMA 衰减为 0.9，
温度在 100 轮内由 1 衰减至 0.1。`--seed` 只控制模型和采样 RNG；PODEM 的
backtrack limit 不通过训练命令修改。用 `train --help` 查看配置项。scorer 和
embedding 使用 CPU，单线程和确定性运算，便于精确恢复；不微调 DeepGate2。

## 输出与恢复

- `latest.pt`：最后完整轮次的模型、优化器、RNG、baseline、来源摘要和 best 快照。
- `best.pt`：由 latest 中的 best 快照生成，供独立确定性评估使用。
- `rounds/round-NNNNNN.jsonl`：每轮 episode、奖励、loss contribution 和评估日志。
  `round-000000` 记录原始顺序 baseline 及未训练模型评估。
- `rounds/round-NNNNNN.npz`：按电路名保存采样排序的 catalog 行号，行号从 0 开始。
- `evaluation/summary.json`：重新运行 best 得到的指标和原始顺序对比。
- `evaluation/<circuit>.ranking.npz`：catalog 顺序的 `fault_ids`、`scores`、从 1
  开始的 `ranks`，以及从 0 开始的排序行号 `permutation`。

每轮日志文件在提交后不修改。以 `latest.pt` 的 round 为提交界限；中断时可能有
更高轮次的未提交日志，恢复时会重新生成这些文件。只有全部 episode、梯度更新
和本轮评估成功后才原子替换 latest；best 是可从 latest 修复的派生文件。若恰好
在 latest 提交后、best 发布前中断，独立评估会拒绝旧 best；执行一次
`train --resume latest.pt` 会先修复它。

恢复检查 manifest、BENCH、fault map、embedding、PODEM 二进制和 PyTorch 版本；
不兼容时拒绝继续。只加载本包生成的可信本地 checkpoint，因为其中保存了
Python/NumPy 的 RNG 对象。新训练要求空输出目录；恢复只允许调整总目标轮数。

程序启动时的 `validate` 不运行 ATPG；新训练额外执行全部原始顺序 baseline
和初始确定性评估，结束时再对 best 做全新评估。短测试证明训练链路可运行，
不代表训练已收敛或排序性能一定优于原始顺序。
