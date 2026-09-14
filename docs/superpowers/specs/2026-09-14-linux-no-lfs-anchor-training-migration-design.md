# Linux 无 LFS 的 Anchor-AIG 训练迁移设计

## 目标

把已经生成的原始 BENCH、anchor-preserving AIG、DeepGate2 embedding、fault metadata
和强化学习 manifest 纳入普通 Git，使一台没有 Git LFS、没有 root 权限的 Linux
服务器在克隆仓库后可以直接编译 PODEM、训练并验证。

迁移后的关键语义保持不变：DeepGate2 的输入是 AIG，强化学习读取与原始网表 fault
严格对应的 embedding，PODEM ATPG 始终读取原始 BENCH。Linux 上不重新运行
DeepGate2，也不重新生成数据。

## 已选方案

所有生成数据直接作为普通 Git blob 提交，不使用 LFS，也不拆包。当前最大单文件约
56 MiB，总数据约 1 GiB。这个方案的代价是首次 push 和 clone 较慢，优点是 Linux
端不需要额外工具、安装权限或解包步骤。服务器建议使用浅克隆；后续避免频繁改写
embedding 文件，以免 Git 历史继续膨胀。

不采用以下方案：

- Git LFS：服务器当前没有 LFS，目标是零额外系统依赖。
- 分卷压缩：不能显著压缩 NPZ，还会增加恢复和校验步骤。

## 仓库内容和提交边界

本次迁移提交包括：

- `datasets/train` 和 `datasets/validation`：PODEM 使用的原始 BENCH；
- `datasets/train_AIG` 和 `datasets/validation_AIG`：每个电路的 AIG、映射、fault
  metadata、门级特征和 fault embedding；
- `datasets/anchor_aig_dataset.json`：完整数据集摘要与哈希；
- anchor 训练、smoke、整体验证和单电路验证 manifest；
- anchor-AIG 生成与校验代码、双路径 RL loader、相关测试和文档；
- 无 root、无 LFS 的 Linux 编译、训练和验证脚本。

`runs/`、平台相关的 `.pyd`/`.so`、构建目录和本机 DeepGate2 checkout 不提交。
工作区中与本功能无关的历史样例删除、DeepTPI 导入改动和其他未确认文件不进入本次
提交。采用显式文件列表暂存，避免把脏工作区整体提交。

## 可迁移路径规则

现有 artifacts 是在 Windows 生成的，一些 JSON/NPZ metadata 中记录了 Windows
绝对路径。这些路径只作为生成时的审计信息，不能作为数据身份的一部分。

迁移后校验遵循以下规则：

1. manifest 中实际文件位置相对仓库解析；
2. 原始 BENCH、AIG、mapping 和 embedding 的内容哈希是权威身份；
3. metadata 中记录的源路径和输出路径不参与跨机器一致性比较；
4. fault 数量、fault key、anchor 名称、row index、图节点数、特征维度以及内容哈希
   仍进行严格检查；
5. 不重写已经生成的 NPZ，因此 embedding 数值和 fault 行序保持逐字节不变。

`generate_anchor_aig.py --check-only` 必须能在仓库整体移动后通过。路径放宽只适用于
审计字段，不能放宽 fault 到 embedding 的对应关系或文件哈希。

## Linux 入口

提供三个职责分离的 Bash 入口：

- setup：使用当前 `python3` 原地编译 `cpp_podem`，不调用 sudo、apt、conda 或 pip；
- train：默认使用 `configs/anchor_train_1024.json`，支持轮数、输出目录、线程数和
  batch size 参数，检测 `latest.pt` 后自动续训；
- evaluate：读取训练目录的 `best.pt`，默认在
  `configs/anchor_validation_6.json` 上验证，也允许传入单电路 manifest。

脚本从自身位置定位仓库根目录，可以从任意当前目录启动。错误直接返回非零状态并
保留底层 Python/编译器错误。服务器至少需要用户可执行的 Python 3.9+、C++ 编译器、
setuptools、NumPy 和 PyTorch；这些系统条件无法由无 root、离线脚本代装，因此 setup
先给出明确的缺失项诊断。

推荐服务器命令：

```bash
git clone --depth 1 git@github.com:darkdarkgogo/fault_reorder.git
cd fault_reorder
./scripts/setup_anchor_linux.sh
./scripts/run_anchor_linux.sh 100 runs/anchor_train_1024
./scripts/evaluate_anchor_linux.sh runs/anchor_train_1024
```

## 数据流

训练加载一个 circuit spec 后，读取原始 BENCH 作为 ATPG 电路，同时读取 AIG graph
和预先计算的 fault embedding。loader 根据 metadata 验证每个 collapsed fault 的
原始 fault key、anchor 和 embedding row，再把排序结果作为原始 fault key 列表传给
PODEM。没有 `.faultmap` 时，PODEM 直接按原始网表 fault catalog 解析这些 key。

因此 Linux 迁移不引入新的映射层：跨平台只改变文件所在路径，不改变
`original fault -> AIG anchor -> embedding row` 的既有映射。

## 错误处理

- 数据缺失或哈希不匹配：在训练开始前失败，并报告 circuit id 和文件；
- metadata 路径是 Windows 绝对路径：忽略该审计字段，继续用当前实际路径与哈希；
- fault key、anchor、row 或数量不一致：严格失败，不允许回退或猜测；
- `cpp_podem` 未编译：setup 明确提示编译要求，训练不静默使用替代实现；
- checkpoint 不存在：evaluate 明确失败，不自动选择其他模型；
- 训练被中断：同一输出目录再次执行时从 `latest.pt` 恢复。

## 验证标准

实现完成需满足：

1. anchor 单元测试和 RL loader 回归测试通过；
2. 把仓库复制到不同绝对路径后，完整 manifest 与 `--check-only` 校验通过；
3. Bash 脚本通过 `bash -n`；
4. setup 不包含 sudo、apt、conda、pip 或网络下载；
5. smoke manifest 至少完成一轮训练、checkpoint 恢复和一次验证；
6. `git status` 证明本次提交未意外包含无关的删除或修改；
7. Git 中没有 LFS pointer，最大文件与总提交数据量在交付说明中明确报告。

Windows 已完成的全量一轮训练结果作为迁移基线：1024 个 episode、64 次更新、
coverage invalid 为 0。Linux 端的目标是复现可运行性，不要求不同硬件上的浮点训练
轨迹逐位一致。
