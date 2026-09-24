# X-path 重汇去重优化设计

## 背景

`ATPG::trace_unknown_path()` 递归遍历值为 `U` 的 fanout cone，用于判断从
当前 wire 到 primary output 是否仍存在 X-path。当组合电路有大量 reconvergence
且这次查询最终失败时，同一个后级子图会被多条路径重复展开，使时间
复杂度接近可达路径数，而不是可达节点和边的数量。

`b17_C` 中出现慢 secondary fault 的 `P2_U79xx`/`P2_U80xx` 区域每个只有约
6,000–8,000 个唯一后继节点，但拥有数百亿条到 PO 的拓扑路径。

## 目标

- 保持现有 X-path 存在性语义和 fanout 搜索顺序。
- 使每次顶层 X-path 查询中每个 wire 最多展开一次，复杂度为
  `O(V + E)`。
- 不在 PODEM 迭代或回溯之间复用过期结果。

## 方案选择

采用每次顶层调用新建的局部 `unordered_set<wptr>`：

1. 公开的 `trace_unknown_path(wptr)` 创建 visited set。
2. 新增私有递归 helper，接收 visited set 引用。
3. helper 进入 wire 时尝试插入；已访问则直接返回 `false`。
4. 其余 PO 判定、`U` 过滤和 fanout 顺序保持不变。

在单次查询中电路状态不变，因此一个已经完整证明无路径的 wire
从另一条重汇边再次到达时仍然无路径，该剪枝是正确的。visited 不能持久化
到下一次调用，因为 PODEM 赋值或回溯会改变 `U` 子图。

不采用 wire generation stamp：它能减少 hash 和分配开销，但需要修改全局
`WIRE` 状态并管理 generation 溢出。不采用整个电路的反向 DP：它的改动面
更大，而当前只需修复局部查询的重复展开。

## 错误处理与边界

- 空 fanout 且非 PO 的 wire 返回 `false`。
- PO 保持返回 `true`。
- visited 仅用于单次存在性查询，不更改 wire value 或其他 ATPG 状态。
- levelization 保证输入为组合 DAG；visited 主要用于重汇去重，同时也使
  helper 对意外环更稳健。

## 测试与验收

- 添加源码结构回归，确认 visited set 在顶层调用内创建并传入递归 helper，
  而不是全局或跨调用缓存。
- 重建 `cpp_podem` 原生扩展。
- 运行 `PODEM/tests` 及相关 ATPG 回归，确认 fault 状态、vector 和计数不变。
- 如运行 b17 诊断，原先 `iterations=1, decisions=0, backtracks=0` 的慢失败
  secondary 应从数百秒降到与数千节点线性遍历相符的时间。

