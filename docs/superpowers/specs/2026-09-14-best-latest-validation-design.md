# Best 与 Latest Checkpoint 双验证设计

## 目标

训练继续同时保存 `best.pt` 和 `latest.pt`，并在同一套外部验证电路上分别评估两个 checkpoint。验证结果以原始 native fault 顺序为基准，明确报告 fault coverage 的变化和 pattern count 的变化。

## Checkpoint 语义

- `best.pt` 保存训练期间通过训练集覆盖门槛、且按既有 ATPG 指标排序最优的模型。它继续作为历史最优推理模型。
- `latest.pt` 保存最后完整提交轮次的模型、优化器和随机状态。它继续作为断点续训的唯一入口，同时允许只读评估其中的模型权重。
- 评估 `latest.pt` 不改变该文件，不把它伪装或复制成 `best.pt`，也不要求它满足训练集覆盖门槛。报告必须保留 checkpoint 类型和轮次，使两个结果不会混淆。

## 验证流程

`evaluate_anchor_linux.sh` 接收训练目录和可选验证 manifest，依次执行：

1. 读取 `<run-dir>/best.pt`，在验证 manifest 上运行确定性模型排序和 native fault 顺序。
2. 读取 `<run-dir>/latest.pt`，在同一 manifest 上执行相同流程。
3. 将结果写入相互独立的目录：
   - `<run-dir>/evaluation-<manifest>-best/`
   - `<run-dir>/evaluation-<manifest>-latest/`

`best.pt` 不可用或任一 checkpoint 缺失时，脚本以非零状态退出并给出明确错误，避免只完成一半却被误认为双验证已经完成。已有可恢复的逐电路外部验证机制分别作用于两个输出目录。

## Native baseline 与指标

“启发式算法”在本设计中固定指 PODEM 按原始网表 native fault 顺序运行的结果。对于每个验证电路，两个 checkpoint 都各自运行并记录同一语义的 native baseline。结果只按电路比较，不提供或展示将多个电路相加得到的总计变化。

每个电路均报告：

- `native_covered_equivalent_faults`
- `model_covered_equivalent_faults`
- `covered_fault_increase = model - native`
- `native_fault_coverage`
- `model_fault_coverage`
- `fault_coverage_increase`，采用比例值
- `fault_coverage_increase_percentage_points = 100 * (model - native)`
- `native_pattern_count`
- `model_pattern_count`
- `pattern_reduction = native - model`
- `pattern_reduction_percent = 100 * (native - model) / native`

正的 `covered_fault_increase` 和 `fault_coverage_increase_percentage_points` 表示覆盖提升；正的 `pattern_reduction` 表示 pattern 减少。负值原样保留，不能截断为零。
当 native pattern count 为零时，若模型也为零则减少比例为 `0`；若模型产生了
pattern，则绝对减少量保留负数，百分比因除数为零在 JSON 中写 `null`、终端显示
`N/A`，不能用 `0%` 掩盖退化。

## 输出

每个 checkpoint 的验证目录包含：

- `summary.json`：完整逐电路指标、checkpoint 类型、轮次及来源；不写跨电路对比总计。
- `comparison_by_circuit.csv`：上述逐电路 native/model 对比字段。
- 每个电路的排序或断点恢复文件，保持现有外部验证格式。
- `status.json`：记录该 checkpoint 与 manifest 的身份以及已完成电路。

命令行在每个 checkpoint 完成后打印逐电路表格。每一行至少包含电路名、checkpoint 类型和轮次、native/model coverage、coverage 百分点变化、native/model pattern count、pattern 减少数和减少百分比。命令行和对比 CSV 不显示跨电路总计行。

## 代码边界

- checkpoint 加载与评估逻辑接受 `kind=best` 或 `kind=latest`，但只有 `latest.pt` 可以用于恢复训练。
- 对 `best.pt` 保留现有的可用性、陈旧文件和确定性复验检查。
- 对 `latest.pt` 校验 manifest/embedding/PODEM/PyTorch 兼容性，直接使用其中最后提交轮次的模型；不执行仅适用于派生 `best.pt` 的陈旧文件检查。
- 外部 manifest 验证报告必须使用被加载 checkpoint 的真实类型，不能硬编码为 `best`。
- 不修改训练奖励、best 选择规则、DeepGate2 embedding 或 ATPG fault folding。

## 测试与验收

- 单元测试覆盖 `latest.pt` 的外部 manifest 评估及其 checkpoint 类型、轮次和模型权重。
- 单元测试验证逐电路 coverage/pattern 差值，包括提升、下降和零 baseline 边界，并确认对比输出没有跨电路总计行。
- 脚本测试确认同时调用 best/latest，并写入不同目录。
- 现有 best 评估、陈旧 best 检查、断点续训和外部验证恢复测试继续通过。
- Linux shell 语法检查和 fault-order RL 测试集通过。
