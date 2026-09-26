# Five-Circuit Validation 设计

## 背景

stuck-at DTC 的每个 unknown PO 反向 BFS wire 预算已经在 C++
`StuckAtProtocolConfig` 和 Python `PROTOCOL_CONFIG` 中统一为 15，因此所有
stuck-at 电路的 lazy 和 ranked/RL DTC 都使用 15-wire 预算。TDF 不在
本次范围内。

默认 fault-order RL validation manifest 当前包含 6 个电路，其中
`b17_C` 运行时间过长，使整轮 validation 无法在实用时间内完成。

## 目标

- 保持所有 stuck-at 电路的 wire 查找预算为 15。
- 将默认 RL validation 改为 5 个电路：`b12_C`、`b15_C`、
  `b20_C`、`b21_C`、`b22_C`。
- 使默认路径、manifest 文件名、生成脚本和文档都明确表示 5 个
  validation 电路。
- 保留 `b17_C` 原始/AIG 数据和 ATPG 诊断脚本。

## 非目标

- 不删除 `datasets/validation/b17_C.bench` 或
  `datasets/validation_AIG/b17_C.*`。
- 不修改 `scripts/diagnose_b17_atpg.py`。
- 不修改 DeepTPI 的 `configs/deeptpi_itc22_test_9.json`。
- 不改变 TDF 的 `select_fault_try`。
- 不改写历史 design/plan/experiment 文档对当时 6 电路数据集的记录。

## 方案

1. 将 `configs/anchor_validation_6.json` 重命名为
   `configs/anchor_validation_5.json`，并删除其中 `b17_C` 条目。
2. 将 `fault_order_rl.trainer.DEFAULT_VALIDATION_MANIFEST` 和当前用户文档更新为
   `anchor_validation_5.json`。
3. 修改 `scripts/generate_anchor_rl_manifests.py`：validation 源目录仍保留 6 个
   bench，但 manifest 生成时明确排除 `b17_C`，并要求最终恢复为
   5 个条目。这避免为了训练配置而删除诊断数据。
4. 删除 `configs/anchor_validation_single/b17_C.json`。生成脚本会在输出
   single manifests 前清理旧 JSON，防止重新生成后残留过期 b17 文件。

## 数据流与兼容性

- 新训练与验证默认读取 `anchor_validation_5.json`。
- 旧 checkpoint 保存了 validation manifest 路径和 digest；它们与新 5-circuit
  protocol 不兼容，恢复时应按现有校验明确拒绝，而不是静默继续。
- `scripts/generate_anchor_aig.py` 仍管理完整 6-circuit validation artifact
  数据集，不改其 `EXPECTED_SPLIT_COUNTS`。RL manifest 是其中的 5-circuit
  运行子集。

## 测试

- 新增 manifest 回归，验证默认 validation 精确包含上述 5 个电路，
  且不包含 `b17_C`。
- 验证 `DEFAULT_VALIDATION_MANIFEST` 指向存在的 `anchor_validation_5.json`。
- 验证单电路 manifest 集合不包含 `b17_C.json`。
- 使用隔离 fixture 测试 manifest 生成器排除 `b17_C` 并清理过期 single
  manifest，不改动真实数据集。
- 运行 fault-order RL 和 manifest 相关回归，并检查当前运行时代码/文档
  不再引用 `anchor_validation_6.json`。

## 验收标准

1. 默认 RL validation 只加载 5 个电路，不加载 `b17_C`。
2. 重新生成 manifest 不会把 `b17_C` 加回。
3. 所有 stuck-at 电路的对外 protocol 仍报告 small/default wire budget
   均为 15。
4. b17 数据和诊断脚本仍可用，TDF 不变。

