# 共享故障排序强化学习

`fault_order_rl` 使用已有的 257 维 DeepGate2 故障 embedding，为完整故障列表
采样排序，再调用有序 stuck-at PODEM。所有电路共享一个 scorer；一轮包含 manifest
中每个电路各一次 episode，并进行一次等电路权重的 REINFORCE 更新。

anchor-preserving 新数据使用双路径 manifest：`bench` 指向原始 BENCH，供 PODEM
生成 catalog 和执行排序；`aig_bench` 与 `aigmap` 只用于校验 DeepGate2 embedding
来源。这类条目不写 `faultmap`：Python wrapper 会把空字符串传给 C++，使 ATPG
直接使用原始网表 fault。最小真实训练示例：

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m fault_order_rl train `
  --manifest configs\anchor_smoke_train.json `
  --output runs\anchor-smoke --rounds 1 --evaluate-every 1 --threads 1
```

## 本机运行

Linux 脚本使用 `PATH` 中的 `python3`，不会联网、执行 `pip install`、创建 Conda
环境或调用 sudo。仓库已经包含训练 embedding 和 pybind11 头文件。服务器需要
已有 PyTorch、NumPy、setuptools、C++ 编译器和 Python 开发头文件。

运行一次构建脚本：

```bash
./scripts/setup_anchor_linux.sh
```

脚本先检查 Python 3.9+、`c++`、Python 开发头文件、NumPy、setuptools 和
PyTorch，然后生成当前 Python 对应的 Linux `cpp_podem*.so`。它只使用服务器
当前已经提供的工具和 Python 环境；缺少依赖时明确报错，不尝试获取 root 权限或
联网安装。

```bash
python3 PODEM/setup.py build_ext --inplace
```

训练脚本默认在 1024 个 anchor-preserving 训练电路上运行 100 轮，并写入
`runs/anchor_train_1024`：

```bash
./scripts/run_anchor_linux.sh
```

如果输出目录已经有 `latest.pt`，同一条命令会自动从它恢复。训练已经达到目标
轮数时不会重复训练，只执行最终评估并打印结果。

四个位置参数依次是总轮数、输出目录、PyTorch CPU 线程数和新训练使用的 circuit
batch size：

```bash
./scripts/run_anchor_linux.sh 200 runs/anchor-200 1 16
```

续训时 checkpoint 中保存的线程数和 batch size 不允许改变，脚本只使用前两个参数
定位 checkpoint 并延长目标轮数。

断点续训；可选的第二个参数表示总目标轮数：

```bash
python3 -m fault_order_rl train \
  --resume runs/anchor_train_1024/latest.pt --rounds 200
```

默认使用 6 个 validation 电路分别重新评估 best 和 latest，并导出各自排名：

```bash
./scripts/evaluate_anchor_linux.sh runs/anchor_train_1024
```

第二个参数可以指定整体验证或单电路 manifest，第三个参数可以指定输出目录前缀：

```bash
./scripts/evaluate_anchor_linux.sh \
  runs/anchor_train_1024 \
  configs/anchor_validation_single/b12_C.json \
  runs/anchor_train_1024/evaluation-b12
```

以上示例分别写入 `evaluation-b12-best` 和 `evaluation-b12-latest`。终端为每个
checkpoint 打印逐电路表格，展示 native/model coverage、coverage 百分点变化、
native/model pattern count、pattern 减少数和减少比例，不打印跨电路总计。每个
输出目录的 `comparison_by_circuit.csv` 和 `summary.json` 保存同一套逐电路对比，
验证集 summary 不写跨电路对比总计。若 native pattern count 为零而模型产生了
pattern，绝对变化显示为负数，无法定义的减少百分比显示为 `N/A`。

`configs/anchor_train_1024.json` 包含 1024 个训练电路；
`configs/anchor_validation_6.json` 包含 6 个验证电路；
`configs/anchor_smoke_train.json` 用于快速检查链路。路径均相对于 manifest 所在
目录。服务器无需 DeepGate2 源码、预训练权重或联网下载。

完整校验不会信任生成机器的 Windows 绝对路径，而是以当前仓库文件位置和内容哈希
为准；fault ID、anchor、embedding row、数量和向量数值仍严格核对：

```bash
python3 scripts/generate_anchor_aig.py --check-only
python3 -m fault_order_rl validate --manifest configs/anchor_train_1024.json
```

物理 XOR/EQV 门需要先展开为 PODEM 支持的门；内置二值电路已经使用这种形式。
逻辑 XOR 输入 fault 仍通过 V3 map 保留。直接把未展开的 XOR 门交给有序
PODEM 会报错，避免旧 backtrace 实现的未定义行为。

## 奖励与模型选择

- 有效 episode 的原始奖励为 `previous_pattern_count - current_pattern_count`。
- 非折叠已解决故障数少于原始顺序时，奖励为负的
  `max(native_pattern_count, previous_pattern_count, current_pattern_count, 1)`，
  并保留之前的 pattern baseline。已解决故障数等于 detected equivalent fault 与
  redundant equivalent fault 之和。
- advantage 使用更新前的 reward EMA，并除以 native pattern 数。每个电路的
  排序 log probability 除以自身 fault 数，然后对电路等权平均。
- 评估采用分数降序和稳定行号破同分，不加入探索噪声。best 必须在每个电路上
  达到原始顺序的非折叠覆盖数，再依次比较总 pattern 数、总覆盖数、PODEM 调用数
  和回溯数。训练期间不提高原始覆盖门槛。
- 没有合格模型时，`best.pt` 明确标记不可用；训练命令仍会评估 `latest.pt`，并
  输出 `coverage_eligible: false`、`coverage_shortfall`、pattern 数和减少量。
  `coverage_shortfall` 的单位是 uncollapsed fault，等于各电路相对其原始覆盖数的
  缺口之和。

PODEM 固定 seed=14、backtrack limit=5000、每 fault 尝试一次；STC、DTC、SCOAP
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
- `evaluation/summary.json`：重新运行 best 得到的指标和原始顺序对比；没有合格
  best 时保存 latest 的确定性评估，并以 uncollapsed fault 数记录 coverage 缺口。
- `evaluation/<circuit>.ranking.npz`：catalog 顺序的 `fault_ids`、`scores`、从 1
  开始的 `ranks`，以及从 0 开始的排序行号 `permutation`。

`fault_order_rl evaluate` 可读取 `best.pt` 或 `latest.pt`。best 继续执行覆盖资格、
陈旧派生文件和确定性复验检查；latest 只读使用最后完整轮次的模型权重，不改变其
断点续训内容，也不要求最后一轮满足 best 的训练集覆盖门槛。

每轮日志文件在提交后不修改。以 `latest.pt` 的 round 为提交界限；中断时可能有
更高轮次的未提交日志，恢复时会重新生成这些文件。只有全部 episode、梯度更新
和本轮评估成功后才原子替换 latest；best 是可从 latest 修复的派生文件。若恰好
在 latest 提交后、best 发布前中断，独立评估会拒绝旧 best；执行一次
`train --resume latest.pt` 会先修复它。

恢复检查 manifest、BENCH、fault map、embedding、PODEM 二进制和 PyTorch 版本；
不兼容时拒绝继续。只加载本包生成的可信本地 checkpoint，因为其中保存了
Python/NumPy 的 RNG 对象。新训练要求空输出目录；恢复只允许调整总目标轮数。
本次覆盖定义升级后的 checkpoint schema 为 v2，旧 checkpoint 必须丢弃并从
round 0 重新训练。

程序启动时的 `validate` 不运行 ATPG；新训练额外执行全部原始顺序 baseline
和初始确定性评估。结束时优先对 best 做全新评估；best 不可用时评估 latest，
明确报告 coverage 是否达标。短测试证明训练链路可运行，不代表训练已收敛或
排序性能一定优于原始顺序。
