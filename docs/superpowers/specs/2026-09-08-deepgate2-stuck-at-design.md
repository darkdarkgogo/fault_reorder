# DeepGate2 接入 AIG stuck-at：设计草案

状态：根据用户最新要求修订，待确认后实现。

## 本阶段范围

本阶段只完成 benchmark → AIG 特征图、原始 ATPG 故障到 AIG 的映射、stuck-at fault 顺序输入和 DeepGate2 embedding 导出。完全忽略 transition-delay、N-detect、RL 环境、奖励和训练接口。

ATPG 的输入和求解对象始终是原始网表。AIG 只供 DeepGate2 计算特征，不替换 PODEM 的电路，不用于 fault injection 或 fault simulation。

输入以新增的 `.bench` benchmark 为主；时序 benchmark 使用已有 `_scan.bench` 全扫描组合版本。`_binary.bench` 和 `.faultmap` 可用于核对原始节点与现有 collapsed fault 信息，但不会把 `__smartatpg_bin_*` 当作原始节点。

## 两条分离的数据路径

```text
原始网表 ─────────────→ 原 PODEM stuck-at ATPG / fault simulation
    │                              ↑
    └→ AIG 特征图 → DeepGate2 → 原始 fault 的 embedding / 排序分数
```

PODEM 产生和删除的仍是原始网表中的 fault。DeepGate2 输出只用于决定这些原始 fault 的先后顺序。AIG 辅助节点不会进入 ATPG fault list，也不会改变原网表的覆盖率分母。

## 核心语义：每个原始信号映射到一个 AIG anchor

转换器不运行会重写或删除内部信号的 ABC 优化，而是按原始 BENCH 定义确定性地构建 AIG。每个原始信号节点保存一个唯一 `anchor_aig_id`：

- 原始二输入 AND 的最终 AND 节点可以直接作为 anchor。
- 原始多输入门分解出的中间 AND 都标记为 `auxiliary`，只有代表该原始门最终输出的节点是 anchor。
- OR、NOR、NAND、XOR、XNOR、NOT 等逻辑由 AND 和反相构成。必要时用 `AND(x,x)` 或等价的非优化结构物化最终原始输出，使它拥有独立 anchor，避免原始信号只成为一条 complemented edge 或与另一个信号合并。
- 原始 BUF 同样物化独立 anchor，不能与输入共用故障位置。
- AIG 转换生成的其他节点具有 `origin_id = null`，只参与 DeepGate2 的图消息传播。

例如原门 `y = OR(a,b)` 可生成内部 AIG 逻辑，最后得到唯一 `anchor(y)`：

```text
a ─┐
   ├─ AIG helpers ─ anchor(y)   ← 为原网表中 y 的 fault 提供 embedding
b ─┘                helpers     ← 不对应原 fault
```

这里的“插入 fault”只表示为对应的原始 stuck-at fault 产生排序特征。实际 SA0/SA1 始终注入原始网表。DeepGate2 仍为全体 AIG 节点生成 embedding，但导出的排序候选表只引用 anchor。

## AIG 表示与映射文件

输出使用 ASCII AIGER `.aag` 作为可检查的规范表示，并生成 sidecar 映射；需要二进制 `.aig` 时再做无优化格式转换。AIGER 符号表通常只保存 PI/PO 名称，不能独自恢复内部原始节点，因此 sidecar 是必需输入，不能事后按名字猜映射。

每个电路输出：

- `<circuit>.aag`：AND-inverter graph；
- `<circuit>.aigmap.json`：原始节点名、原始类型、anchor AIG literal/node index、是否 PI/PO、辅助节点来源、源文件哈希和转换器版本；
- `<circuit>.embeddings.npz`：所有 AIG/DeepGate2 节点的 `hs`、`hf`、图边和节点类型；
- `<circuit>.faults.json`：从原 ATPG fault list 导出的 stuck-at 候选，字段至少包括稳定 `fault_id`、原始 GO/GI 位置、`origin_id`、`anchor_aig_id`、`sa_value`、embedding 行号和等价故障数。

映射记录示例：

```text
fault_id=42  original_site=y:GO  source_anchor=17  sa_value=0
fault_id=57  original_site=g:GI1 source_anchor=17  receiver_anchor=31 sa_value=1
```

同一原始位置若在 collapsed fault list 中同时保留 SA0/SA1，它们共享对应的 anchor embedding，并由显式极性特征区分。一个 anchor 也可能被多个原始 fanout branch fault 引用。辅助 AIG 节点永远不会单独出现在 `faults.json`，即使它们存在于 embedding 矩阵。

## 原网表 ATPG 语义

stuck-at ATPG、PODEM implication、D-frontier、fault injection、fault dropping 和 fault simulation 全部在原始网表上运行。AIG 节点不会进入这些算法。排序器给出原始 `fault_id` 顺序，C++ 按该顺序选择原 fault pointer。

当前 C++ 入口存在两个需要先修复的问题：`main.cpp` 无条件创建 transition fault list，SAF 分支成功后调用了 transition fault simulator。实现时恢复 `generate_fault_list()` 和 `fault_sim_a_vector()`，继续使用原来的 stuck-at fault collapsing 规则。AIG 映射不会重新生成或替换这份 fault list。

本阶段保留一个简单排序输入：可选地读取 `fault_id` 顺序文件，依次决定哪个 anchor 先做 SA0 或 SA1；没有顺序文件时使用稳定默认顺序。它用于确认映射和排序控制有效，不包含 RL。

## 原 fault 到 AIG embedding 的映射

原始 ATPG 先按照现有规则生成 collapsed stuck-at fault list。每个 fault 保留原始网表身份：驱动线、GO/GI、接收门和引脚、SA0/SA1。映射器再为它寻找对应的原始信号 anchor：

- GO fault 使用该原始输出信号的 anchor embedding；
- GI branch fault 使用该原始输入信号的 anchor embedding，并附加接收门 anchor embedding、引脚编号和 GI 标志，以区分同一 fanout stem 的不同分支；
- PI 和 PO 的处理严格跟随原 ATPG fault list，不额外添加 AIG fault；
- 如果某个原始 fault 无法映射到 anchor，转换立即报错，不静默使用辅助节点或零向量。

因此原始 collapsed fault 数量、等价故障数和 coverage 定义保持不变。现有 `_binary.faultmap` 可作为映射核对资料，但 `__smartatpg_bin_*` 不能成为新的故障动作。

## DeepGate2

DeepGate2 对完整 AIG 计算结构 embedding `hs` 和功能 embedding `hf`。fault feature 默认保存：

```text
[hs(source_anchor), hf(source_anchor),
 hs(receiver_anchor), hf(receiver_anchor),
 one_hot(SA0/SA1), one_hot(GO/GI), pin_features]
```

先导出原始向量，不在本阶段做降维、策略网络或微调。checkpoint 必须与代码和 embedding 维度严格匹配；当前工作区没有找到预训练权重，当前 Python 环境也尚未安装 torch/torch_geometric。权重与依赖未就绪时，转换器和映射仍可完成并验证，但不能把随机权重输出称为 DeepGate2 embedding。

## 验收标准

1. 对 c17、c432 以及一个 scan benchmark，检查每个被原 ATPG fault list 引用的原始信号恰好映射到一个 anchor。
2. `faults.json` 的 fault 数、SA 极性、GO/GI、等价故障数与原 ATPG 导出的 collapsed fault list 完全一致；所有 `auxiliary` AIG 节点均不会创建额外 fault。
3. 小电路穷举输入，比较原 BENCH 与 AIG 在每个原始 anchor 上的逻辑值，不只比较 PO；覆盖 AND/NAND/OR/NOR/NOT/BUF/XOR/XNOR 和多输入门。
4. fault 只在原始网表注入；用独立串行 stuck-at 仿真核对 PODEM 输出 pattern 的检测结果，并确认执行过程中没有读取 AIG 作为 ATPG 电路。
5. 默认 fault 顺序和外部顺序文件分别运行，确认只改变目标选择次序，候选集合不变。
6. DeepGate2 checkpoint 到位后，检查 embedding 行数、维度、有限值、映射完整性和重复运行一致性。

## 明确不做

- 不用 AIG 替换原 ATPG 网表；
- 不在 AIG 辅助节点插入 fault；
- 不根据 AIG 节点数改变 coverage 分母；
- 不在本阶段实现 RL。
