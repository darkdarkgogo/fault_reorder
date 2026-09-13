# Anchor-preserving AIG 批量生成设计

日期：2026-09-13

状态：用户已批准方案 1，等待规格文档复核后实施。

## 目标

从以下原始 BENCH 生成供 DeepGate2 使用的结构保持 AIG BENCH：

```text
datasets/train/*.bench
datasets/validation/*.bench
```

输出直接替换当前 ABC 优化版本：

```text
datasets/train_AIG/*.bench
datasets/validation_AIG/*.bench
```

每个 AIG BENCH 配套一个同 stem 的映射文件：

```text
<stem>.aigmap.json
```

原始 `datasets/train` 和 `datasets/validation` 不修改、不删除。AIG 只供 DeepGate2 生成 embedding；fault catalog、fault order 执行、PODEM、fault injection、fault simulation 和覆盖率计算始终使用原始 BENCH。

## 已确认输入约束

- `train` 与 `train_AIG` 各有 1,024 个同名 BENCH。
- `validation` 与 `validation_AIG` 各有 6 个同名 BENCH。
- 所有 1,030 对电路的 PI/PO 名称和声明顺序一致。
- 原始网表只包含 `AND/NAND/OR/NOR/NOT`，最大 fan-in 为 2。
- 当前 AIG 只包含 `AND/NOT`，但经过 ABC 优化后缺少原始内部节点映射，不能用于全量原始 fault embedding。

## 选择的方案

实现一个 Python 脚本：

```text
scripts/generate_anchor_aig.py
```

脚本直接解析原始 BENCH，按拓扑顺序进行局部门型 lowering，不调用 ABC，不运行结构哈希、重写、重定时、常量传播或等价节点合并。转换器在创建每个原始信号的最终 AIG 节点时立即写入 anchor mapping，因而不需要在转换完成后猜测节点对应关系。

未选择的方案：

- ABC protected observation points：工具流程和映射语义更复杂，并可能显著改变 DeepGate2 图结构。
- 对现有优化 AIG 做名字或形式匹配：无法保证已删除或仅存在于 complemented edge 的原始信号都有唯一节点。

## AIG lowering 规则

每个原始 PI 和 gate output 必须对应一个唯一、物化的 AIG node。最终 anchor 节点尽量继续使用原始信号名；helper 使用保留前缀 `__anchor_aig_`。输入中若已使用该前缀，脚本在写文件前拒绝处理。

设 `A(x)` 表示原始信号 `x` 已记录的 anchor：

```text
AND(a,b):
    y = AND(A(a), A(b))
    A(y) = y

NAND(a,b):
    helper = AND(A(a), A(b))
    y = NOT(helper)
    A(y) = y

OR(a,b):
    helper_a = NOT(A(a))
    helper_b = NOT(A(b))
    helper_c = AND(helper_a, helper_b)
    y = NOT(helper_c)
    A(y) = y

NOR(a,b):
    helper_a = NOT(A(a))
    helper_b = NOT(A(b))
    y = AND(helper_a, helper_b)
    A(y) = y

NOT(a):
    y = NOT(A(a))
    A(y) = y
```

当前数据不包含 BUF。解析器可支持 `BUF/BUFF`，但必须用双反相物化独立输出 anchor，不能把 `A(y)` 直接设成 `A(a)`：

```text
helper = NOT(A(a))
y = NOT(helper)
A(y) = y
```

这样即使两个原始信号逻辑等价，也不会把两个 fault site 合并成同一 AIG 节点。

## AIG BENCH 格式

输出保留原始 PI/PO 名称与声明顺序：

```text
INPUT(a)
INPUT(b)
OUTPUT(y)
```

只允许 `AND` 和 `NOT` gate。所有 helper 名字必须唯一，不能与原始 PI、PO 或 gate output 冲突。输出 gate 按可复现的拓扑顺序排列；相同输入文件必须逐字节生成相同 AIG BENCH。

## aigmap schema

每个 `<stem>.aigmap.json` 至少包含：

```json
{
  "schema": "anchor_aig_map_v1",
  "source_bench": ".../datasets/train/example.bench",
  "source_bench_sha256": "...",
  "aig_bench": ".../datasets/train_AIG/example.bench",
  "aig_bench_sha256": "...",
  "converter": "generate_anchor_aig_v1",
  "inputs": ["a", "b"],
  "outputs": ["y"],
  "signals": {
    "a": {
      "anchor_name": "a",
      "anchor_node": 0,
      "origin_kind": "INPUT"
    },
    "y": {
      "anchor_name": "y",
      "anchor_node": 4,
      "origin_kind": "NAND"
    }
  },
  "helpers": [
    {
      "name": "__anchor_aig_0",
      "node": 3,
      "origin": "y"
    }
  ]
}
```

`anchor_node` 使用 DeepGate2 输入图的确定性行号。DeepGate2 图构造器会在所有原始信号 anchor 之后，为每个 PO 确定性地增加一个双反相 boundary anchor；mapping 的 `boundaries` 字段同时记录这些 row。加载时检查 `anchor_name`、node row、source hash 和 AIG hash，防止 mapping 与其他版本的 AIG 混用。

## fault 对齐

fault catalog 由 PODEM 在原始 BENCH 上直接生成，不从 AIG 重新生成。当前单 fanout collapsing 不保留普通 GI row；连接 fault 最终由 upstream 或 downstream GO representative 表示。只有 catalog 中实际保留的 collapsed rows 才生成 embedding。

普通 fault 映射：

```text
GO:
    driver_anchor = signals[fault.node_name]
    feature = [hf(driver_anchor), hf(driver_anchor), sa_value]

GI:
    receiver_anchor = signals[fault.node_name]
    source_anchor = signals[fault.input_wire_name]
    feature = [hf(receiver_anchor), hf(source_anchor), sa_value]
```

边界 fault：

- PI dummy GO 按原始 `INPUT(...)` 顺序绑定到 PI anchor。
- PO dummy GI 按原始 `OUTPUT(...)` 顺序绑定到显式 PO boundary anchor 和原始输出驱动 anchor。

每个 catalog row 保留 `fault_id`、`catalog_row`、GO/GI、SA value、`eqv_fault_num` 和使用的 anchor rows。AIG helper 不创建 fault row。任何 fault 无法映射时，整个电路验证失败。

## 命令行接口

默认命令处理整个数据集并执行安全替换：

```powershell
python scripts/generate_anchor_aig.py
```

提供单电路验证接口：

```powershell
python scripts/generate_anchor_aig.py `
  --source datasets/validation/b12_C.bench `
  --output-dir artifacts/b12-anchor-aig
```

提供 `--check-only`，验证现有 anchor AIG 和 mapping，不写文件。脚本不提供跳过关键验证或允许部分替换的选项。

## 验证

### 静态验证

每个电路必须满足：

1. 原始 BENCH 可以完整解析且无组合环。
2. AIG 只包含二输入 AND 和一输入 NOT。
3. PI/PO 名称与声明顺序逐项相同。
4. 每个原始 PI 和 gate output 恰好映射到一个 anchor。
5. 每个 anchor row 的名称、类型和 AIG graph row 一致。
6. helper 名称唯一且不出现在 `signals` anchor 集合中。
7. source/AIG SHA256 与 mapping 一致。

### 功能验证

转换器使用固定 seed=14 的 bit-parallel random vectors，对每个原始信号而不仅是 PO 比较原始网表值和对应 AIG anchor 值。每个电路至少验证 256 个向量。由于 lowering 是局部等价替换，该检查用于发现解析、极性、拓扑或 anchor row 错误。

单元测试另使用小电路穷举全部输入组合，覆盖 AND、NAND、OR、NOR、NOT 和 BUF。

### fault 验证

使用当前 `cpp_podem` 从原始 BENCH 导出 collapsed catalog，检查：

- 所有普通 GO/GI fault 的必要原始信号都存在于 `signals`；
- 所有 dummy PI/PO fault 都能按端口顺序解析；
- fault ID 无重复，或仅使用 PODEM 已显式生成的 `@dupN`；
- `sum(eqv_fault_num) == uncollapsed_total`；
- 映射后的 fault row 数等于 catalog row 数。

### 端到端 smoke test

批量替换后对 `validation/b12_C.bench` 执行：

1. 从 `validation_AIG/b12_C.bench` 运行固定官方 DeepGate2；
2. 检查 `hs/hf` 为 `[node_count, 128]` 且全部有限；
3. 根据 `b12_C.aigmap.json` 为原始 catalog 生成 257 维 fault embedding；
4. 检查 embedding row 与原始 fault ID 完全对齐；
5. 在原始 `validation/b12_C.bench` 上运行 catalog 原顺序 ordered PODEM；
6. 要求覆盖指标与替换 AIG 前的原始 ATPG baseline 一致。

## 安全替换

批量模式不直接写入现有 `_AIG` 目录。流程为：

1. 在 `datasets` 下创建唯一 staging 目录；
2. 生成全部 1,024 个 train 和 6 个 validation AIG/mapping；
3. 完成全部静态、功能和 fault 验证；
4. 检查 staging 文件集合完整，无额外文件；
5. 删除旧 `train_AIG` 和 `validation_AIG`；
6. 将 staging 目录移动为正式目录；
7. 重新运行正式目录的 `--check-only` 和 b12_C smoke test。

任何生成或验证失败发生在第 5 步之前时，旧目录保持不变，staging 被清理。用户已明确授权在全部 staging 验证成功后删除旧优化 AIG。脚本不保留旧 AIG 备份。

在 Windows 上执行删除或移动前，脚本必须解析并检查所有目标的绝对路径位于仓库的 `datasets` 目录内，并且 basename 严格等于 `train_AIG` 或 `validation_AIG`，避免宽目录误删。

## 测试范围

新增测试覆盖：

- 每种支持门型的 lowering 和逐 anchor 真值等价；
- helper/anchor 唯一性；
- BUF 独立 anchor；
- reserved prefix 冲突；
- undefined wire、重复输出、错误 arity、cycle；
- deterministic byte-for-byte output；
- mapping hash 被篡改时拒绝；
- 单 fanout catalog 只对实际 collapsed representative 生成 row；
- fanout stem 与 GI branch mapping；
- staging 失败不影响正式目录；
- 只允许替换两个精确的 `_AIG` 目标目录。

不在本次实现中修改 RL reward、policy network、PODEM collapsing 规则或 DeepGate2 checkpoint。

## 完成标准

- 1,030 个原始 BENCH 均生成 AIG BENCH 和 mapping。
- 旧优化 AIG 在全部 staging 验证成功后被替换。
- 所有原始信号 anchor 完整且唯一。
- 所有原始 collapsed fault 可映射，AIG helper fault 数为零。
- b12_C DeepGate2 和原始 PODEM smoke test 通过。
- 自动化测试通过，生成日志给出电路数、anchor 数、helper 数、fault 数和 hash 摘要。
