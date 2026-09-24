# Stuck-at DTC 统一 Wire Budget 设计

## 背景

stuck-at DTC 当前按 PI 数量为每个 unknown primary output 选择反向 BFS
wire 展开预算：`ncktin <= 32` 时为 15，否则为 100。该规则同时用于
lazy baseline DTC 和 ranked/RL DTC。大电路的 100-wire 预算会使每个
output 收集并尝试更多 secondary faults。

## 目标

- 将所有 stuck-at DTC unknown output 的 wire 展开预算统一为 15。
- lazy baseline 和 ranked/RL 两条 stuck-at DTC 路径使用相同预算。
- 保留现有 protocol config 字段、Python 结果 schema 和训练配置结构。
- 不修改 TDF DTC。

## 方案

保留 `dtc_bfs_small_input_threshold`、`dtc_bfs_small_select_fault_try` 和
`dtc_bfs_default_select_fault_try` 字段，但将 C++ 与 Python 中的 default budget
从 100 改为 15。现有预算选择分支可保留；两个分支都产生 15，因此
不改变公开配置形状，也不引入硬编码与对外报告值不一致的问题。

本次不删除阈值或 small/default 字段；删除它们会改变 protocol identity、
Python 验证和已有配置文件，超出“统一预算值”的需求。

## 数据流与边界

- `find_next_stuck_at_dtc_batch()` 仍从 protocol config 读取预算，用于
  ranked/RL candidate batch 发现。
- `run_stuck_at_lazy_dtc()` 读取同一 protocol config，用于 lazy baseline。
- 每切换到一个 unknown PO，15-wire 预算重新计数。
- 预算仍只统计实际出队并展开的 `U` wire，不改变计数口径。
- `PODEM/src/tdfatpg.cpp` 和 TDF 相关配置、测试、文档不修改。

## 测试与文档

- 将现有 33-input stuck-at fixture 的预期预算从 100 改为 15，并验证
  `visited_wire_count <= 15`。
- 验证 native protocol config 和 Python `PodemSession` 默认值均报告 15。
- 运行 `PODEM/tests` 以覆盖 ranked 和 lazy stuck-at DTC 路径。
- 更新 stuck-at 算法与 RL 文档，将 15/100 规则改为所有电路统一 15。
- 历史设计文档保留当时决策，不回写。

## 验收标准

1. 任意 PI 数量的 stuck-at DTC 都报告 `select_fault_try=15`。
2. 每个 unknown PO 的 `visited_wire_count` 不超过 15。
3. lazy baseline 与 ranked/RL 的预算值一致。
4. TDF DTC 源码和行为不变。

