# 非折叠已解决故障覆盖率设计

## 目标

故障排序训练的覆盖门槛统一使用非折叠故障口径，并把 PODEM 已证明为
redundant 的故障计入覆盖。pattern count 只能在不降低该覆盖率的候选之间比较。

本次修改后重新开始训练；旧 `latest.pt` 和 `best.pt` 不做迁移。

## 指标定义

每个 collapsed fault 已有 `eqv_fault_num`，表示它代表的 uncollapsed fault 数量。
PODEM 保留现有指标，并新增：

```text
redundant_equivalent_faults
```

当 PODEM 对一个 collapsed fault 返回 `FALSE`、将其证明为 redundant 时，除了把
`redundant_faults` 加一，还把该 fault 的 `eqv_fault_num` 加到
`redundant_equivalent_faults`。

Python 对每次运行派生：

```text
covered_equivalent_faults =
    detected_equivalent_faults + redundant_equivalent_faults

fault_coverage =
    covered_equivalent_faults / uncollapsed_faults
```

`detected_equivalent_faults` 与 `redundant_equivalent_faults` 是互斥集合：被证明为
redundant 的 fault 不参与后续 fault simulation，不能再次计为 detected。

`fault_coverage` 用于报告；训练判断比较整数 `covered_equivalent_faults`，避免浮点
误差。每个电路的 `uncollapsed_faults` 由固定输入 artifact 决定，因此比较覆盖数
与比较覆盖率等价。

## PODEM 接口

`ATPG::AtpgRunResult` 和 Python binding 新增
`redundant_equivalent_faults`。现有的 `redundant_faults` 继续表示 redundant 的
collapsed fault 数量，供诊断使用；不得将它与 equivalent 指标直接相加。

Python `PodemEnvironment` 要求新字段存在，并为每次结果添加
`covered_equivalent_faults` 和 `fault_coverage`。它验证：

```text
0 <= detected_equivalent_faults
0 <= redundant_equivalent_faults
covered_equivalent_faults <= uncollapsed_faults
0 <= fault_coverage <= 1
```

缺少新字段表示 PODEM extension 尚未重新构建，应返回明确错误。

## 训练状态与奖励

每个电路的状态字段从 `required_detected_equivalent_faults` 改为
`required_covered_equivalent_faults`，初始值来自原始 catalog 顺序的
`covered_equivalent_faults`。

episode 有效条件改为：

```text
metrics.covered_equivalent_faults >=
    state.required_covered_equivalent_faults
```

有效 episode 的 pattern reward、无效 episode 的惩罚以及 EMA 算法保持不变。
随机采样 episode 不修改覆盖门槛。

定期确定性评估若得到更高的 `covered_equivalent_faults`，才提高对应电路的门槛。
提高门槛与保存产生该结果的候选模型发生在同一次事务轮次内，保证新门槛始终有
可复现的 checkpoint 支撑。

## Best 选择与报告

best eligibility 要求每个电路的 `covered_equivalent_faults` 都达到当前门槛。
合格候选仍优先最小化总 pattern count；后续排序条件中的覆盖项改为最大化总
`covered_equivalent_faults`，再比较 PODEM calls、backtracks 和 round。

`evaluation/summary.json` 保留 detected、redundant collapsed 和其他原始指标，
并新增每个电路及 totals 的：

```text
redundant_equivalent_faults
covered_equivalent_faults
fault_coverage
```

总覆盖率使用总数相除，不取各电路覆盖率的平均值。

`coverage_shortfall` 及 `coverage_shortfall_by_circuit` 改为以 uncollapsed fault 数量
表示：

```text
max(0, required_covered_equivalent_faults - covered_equivalent_faults)
```

字段名保持不变，但文档明确其单位为 uncollapsed faults。

## Checkpoint 与恢复

checkpoint schema 版本升级。旧 checkpoint 缺少
`redundant_equivalent_faults`、新的训练门槛和新 PODEM digest，恢复时明确拒绝，
不进行猜测或隐式迁移。用户需要重新构建 `cpp_podem` 并从 round 0 重新训练。

新 checkpoint 保存新的 native metrics、state 门槛、每轮 report 和 best report，
从新 `latest.pt` 恢复仍需保持模型、optimizer、RNG、baseline 和 best 的精确一致性。

## 测试

1. C++/binding 测试确认返回 `redundant_equivalent_faults`，并验证 redundant fault
   按 `eqv_fault_num` 而不是 collapsed 条目数累加。
2. environment 测试验证新字段、派生覆盖数、总数上界以及旧 extension 的明确错误。
3. reward 测试覆盖 detected 与 redundant 互相变化但总 uncollapsed 覆盖不变的情况。
4. eligibility、best invalidation 和 coverage shortfall 测试全部使用
   `covered_equivalent_faults`。
5. checkpoint 测试确认旧 schema 被拒绝，新 schema 可以精确恢复。
6. 真实小电路端到端测试确认每个结果满足
   `covered_equivalent_faults <= uncollapsed_faults`，并能训练、恢复和导出 best。

## 不在范围内

- 不改变 PODEM backtrack limit、seed、fault dropping 或 redundant 判定算法。
- 不把 aborted fault 计入覆盖。
- 不迁移旧 checkpoint，也不复用旧 best。
- 不删除 collapsed 指标；它们继续用于诊断，但不参与覆盖门槛。
