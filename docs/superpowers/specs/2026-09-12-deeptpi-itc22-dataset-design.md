# DeepTPI ITC22 数据接入 fault-order RL

## 目标

让 `reorderATPG` 的共享 fault 排序模型使用 DeepTPI 仓库中 ITC22 数据集的
前 512 个训练电路进行训练，并使用同一数据集现有的全部 9 个测试电路进行独立
推理。训练集与测试集必须严格隔离，测试结果不得参与 checkpoint 选择。

最终评估除现有 JSON summary 和 ranking NPZ 外，还要输出逐电路 pattern count
对比表，明确展示原始 fault 顺序、模型顺序和减少量。首个实验训练 1 轮，用于验证
完整数据链路和测量实际运行成本。

## 固定数据划分

数据源为 DeepTPI checkout 内的两个本地文件：

```text
DeepTPI-main/DeepTPI-main/data/ITC22_dataset/train/benchmarks_circuits_graphs.npz
DeepTPI-main/DeepTPI-main/data/ITC22_dataset/test/benchmarks_circuits_graphs.npz
```

- 训练集按照 NPZ 中 `circuits` 字典的保存顺序选择前 512 个电路，范围从
  `DMA_aig_000` 到 `s35932_aig_121`。
- 测试集使用 test NPZ 中全部 9 个电路：`b12_C`、`b15_C`、`b17_C`、
  `b20_C`、`b21_C`、`b22_C`、`i2c_aig`、`mem_ctrl_aig`、`max_aig`。
- 导入脚本必须记录源 NPZ SHA256、选择规则和最终电路名列表。输入顺序或内容变化
  时不得静默产生同名数据集。
- 不使用 DeepTPI `rl_train.py` 中跳过前 5 个索引的行为；本实验必须恰好包含前
  512 个训练电路。

## 数据目录与产物

在 `reorderATPG/datasets/deeptpi_itc22/` 下保存数据集：

```text
datasets/deeptpi_itc22/
  raw/
    train/benchmarks_circuits_graphs.npz
    test/benchmarks_circuits_graphs.npz
  train/<circuit>/
  test/<circuit>/
  dataset.json
configs/deeptpi_itc22_train_512.json
configs/deeptpi_itc22_test_9.json
```

`raw` 中保存原始 NPZ 的仓库内副本。每个电路目录包含 source BENCH、binary
BENCH、fault map、fault-map binding、DeepGate2 gate embedding、257 维 fault
embedding 和 metadata。两个 manifest 只引用各自 split 的产物。

生成数据体积较大，尤其测试集共 486,901 个节点。raw NPZ、BENCH、fault map、
binding、embedding 和生成后的 manifest 均保存在 `reorderATPG` 工作区，但加入
`.gitignore`，不提交到 Git。导入脚本和 manifest schema 测试纳入版本控制；
`dataset.json` 随本地数据生成，用于复现和校验，不提交到 Git。

## NPZ 到 BENCH 的转换

新增独立导入脚本，使用 `allow_pickle=True` 读取这两个受信任的本地 NPZ。每个
电路对象必须且只能按已知 schema 提供 `x` 和 `edge_index`，并经过以下校验：

- `x` 为二维数组，节点编号唯一，门类型只允许 0、1、2、3。
- `edge_index` 为合法的有向边列表，所有端点均在节点范围内。
- 图必须无环；每个非输入节点必须有输入边。
- 门类型映射固定为 `0=INPUT`、`1=AND`、`2=NOT`、`3=BUF`。
- 无入边节点写为 `INPUT`，无出边节点写为 `OUTPUT`；两者与图计算结果必须一致。
- 节点统一命名为 `N<原始节点索引>`，避免浮点序列化或原数据名称格式影响解析。

source BENCH 随后进入现有 `convert_binary_bench`，生成 PODEM 使用的 binary BENCH
和折叠 fault map。导入器为新 fault map 生成绑定 sidecar，并使用当前固定的官方
DeepGate2 源码和 checkpoint 导出 fault embedding。已有且校验通过的单电路产物
可以跳过，使 521 个电路的准备过程支持断点续跑；不完整或摘要不匹配的目录必须
明确报错，不能混用旧文件。

## 训练与跨 manifest 评估

训练命令继续使用现有 `Trainer`，输入
`configs/deeptpi_itc22_train_512.json`。一轮仍表示 512 个训练电路各执行一次
episode，best checkpoint 只根据训练 manifest 的确定性评估产生。

扩展 `evaluate` 命令，允许显式传入测试 manifest：

```powershell
python -m fault_order_rl evaluate `
  --checkpoint runs/deeptpi-itc22-1r/best.pt `
  --manifest configs/deeptpi_itc22_test_9.json `
  --output runs/deeptpi-itc22-1r/test-evaluation
```

加载 checkpoint 时仍校验模型 schema 和训练来源，但测试 manifest 不要求与训练
manifest digest 相同。评估器独立校验测试 BENCH、fault map、embedding 和
DeepGate2 provenance，并为测试集重新运行原始 catalog 顺序 baseline。报告同时
记录训练 manifest digest 与评估 manifest digest，防止把测试结果误认为训练集
best 指标。

测试集只执行确定性打分和排序，不更新模型、优化器、reward EMA、训练 baseline
或 checkpoint。若训练没有 coverage 合格的 `best.pt`，沿用现有规则评估明确标记
的 latest checkpoint。

## 逐电路 pattern count 表

每次评估都在 summary 中加入逐电路对比，并额外写出：

```text
evaluation/pattern_reduction_by_circuit.csv
```

CSV 每行对应一个电路，列为：

```text
circuit,native_pattern_count,model_pattern_count,pattern_reduction,
pattern_reduction_percent,native_covered_equivalent_faults,
model_covered_equivalent_faults,coverage_eligible
```

计算规则为：

```text
pattern_reduction = native_pattern_count - model_pattern_count
pattern_reduction_percent = pattern_reduction / native_pattern_count * 100
```

减少量允许为负，负数表示模型排序生成了更多 pattern。只有 coverage 达到该电路
原始顺序水平时，`coverage_eligible` 才为真；汇总总减少量不得掩盖任一电路的
coverage 缺口。

## 执行阶段

1. 先对少量训练和测试电路运行转换、fault map、embedding、manifest 和 PODEM
   的管线测试，确认图语义和门类型映射正确。
2. 准备完整的 512/9 数据集，并运行 `validate`。
3. 对 512 个训练电路执行 1 个训练轮次，产出 checkpoint。
4. 用该 checkpoint 对 9 个测试电路执行确定性评估，生成逐电路 CSV。

第 4 阶段可能是主要耗时。9 个测试电路中 `mem_ctrl_aig` 有 127,353 个节点，
`b17_C`、`b22_C` 也接近十万个节点。完整 fault embedding 和逐 fault PODEM
可能需要较长时间或大量内存。每完成一个测试电路，就原子写入该电路的 ranking
NPZ 和 metrics JSON，并更新 `status.json` 中的完成/未完成列表。恢复评估时可跳过
与 checkpoint、manifest 和产物摘要一致的已完成电路。只有 9 个电路全部完成后，
才发布最终 summary 和 CSV；部分汇总不得标记为完整测试结果。

## 错误处理与可恢复性

- 原始 NPZ schema、门类型、边范围、DAG 性质或固定电路数量不符时立即失败。
- 转换后的 binary BENCH 必须通过现有随机向量等价验证。
- fault catalog、fault map、binding 与 embedding ID 必须完全一致。
- 单电路导入使用临时目录并原子发布，失败不会留下看似完成的目录。
- 数据集 metadata 和 manifest 最后生成；只有全部目标电路准备完成后才标记数据集
  `complete`。
- 跨 manifest 评估不能绕过测试产物校验，也不能修改 checkpoint。
- 单电路 metrics、状态、CSV 和 summary 均使用临时文件原子替换；CSV 和 summary
  只在全部目标电路完成时发布。

## 测试与验收

单元测试覆盖 NPZ 选择顺序、512 上限、9 个测试电路、图到 BENCH 门映射、非法图
拒绝、manifest split 隔离、跨 manifest checkpoint 加载和逐电路减少量计算。CSV
测试必须包含正减少、零减少、负减少和 coverage 不合格四种情况。

集成测试使用少量电路完成 source BENCH、binary BENCH、fault map、embedding、
manifest validation、一次训练和测试 manifest 评估。完整数据验收要求：

1. train manifest 恰好有 512 个唯一电路，test manifest 恰好有 9 个唯一电路，
   两者名称无交集。
2. 521 个电路的 BENCH、fault map、binding、embedding 和 metadata 全部通过校验。
3. 1 轮训练只访问训练 manifest，测试结果不影响 best checkpoint。
4. 测试报告为每个已完成电路给出 native/model pattern count、减少量和 coverage。
5. 完整评估成功时 CSV 恰好包含 9 行电路数据，并与 summary 数值一致。

## 不在本次范围内

- 复现 DeepTPI 的测试点插入 DQN 或其 LBIST/ATPG_PC reward。
- 使用全部 1042 个训练电路。
- 使用测试电路参与训练、调参或 best checkpoint 选择。
- 改变现有 fault collapsing、257 维 embedding 语义或 PODEM 算法。
- 在首轮实验中加入并行或分布式 ATPG。
