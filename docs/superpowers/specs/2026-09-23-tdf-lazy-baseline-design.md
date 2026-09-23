# TDF Lazy Heuristic Baseline 设计规范

## 背景

2026-09-22 版 Ranked-DTC 设计让 heuristic baseline 与 RL 共用“在
`select_fault_try` 预算内收集完整 BFS candidate batch”的候选发现过程。
这使 baseline 不再具有原 TDF `tdf_podemx_bt()` 的 lazy 控制流：原 TDF
只在当前 fault 队列为空时继续展开 BFS wire，一旦发现候选就立即
尝试。

本修订把 heuristic/native baseline 恢复为 TDF lazy DTC，同时保留 RL
的完整 batch 排序语义。

## 目标

- `StuckAtSession.step(primary_fault_id)` 使用 TDF lazy DTC。
- `run_stuck_at_ordered(..., ordered_fault_ids)` 使用同一套 TDF lazy DTC。
- `begin_step(primary_fault_id)` + `rank_dtc_candidates(permutation)` 保持
  Ranked-DTC：收集完整 BFS batch，再执行模型给出的完整 permutation。
- 两种模式在每次 secondary 尝试后都重新建立 good-circuit cube 并检查
  当前 unknown PO；PO 一旦不再为 `U`，立即停止当前 PO 的后续尝试。

## 非目标

- 不改变 Primary fault 的选择策略。
- 不改变 RL actor 的打分、完整 permutation 约束或 executed-prefix
  概率语义。
- 不把原 TDF 中缺少重汇去重、过期 fault 过滤等旧实现缺陷
  重新引入 stuck-at 路径。
- 不改变 PODEMX、accepted PI cube 回滚、preserved-fault 检查、fault
  simulation 或 STC。

## 备选方案

### 方案 A：独立 lazy baseline 调度器（采用）

保留现有完整 batch 发现函数专供 RL，新增一个 baseline-only lazy
调度器。两条路径复用同一个“尝试一个 secondary”原子操作。

优点是 RL 协议不需改动，baseline 的 lazy 状态机也能直接对照
`tdf_podemx_bt()`；两种候选发现语义不会因为过度抽象而混淆。

### 方案 B：单一参数化 BFS 状态机

用 `lazy`/`collect_all` 模式参数控制同一个状态机。代码表面更集中，
但两种模式在“何时返回、何时展开 wire、调用方是否必须提交
permutation”上差异较大，容易产生难以验证的交叉分支。

### 方案 C：继续先收集完整 batch，仅改 baseline 的执行循环

这只能模拟“按 BFS 顺序尝试”，不是 TDF lazy，因为后层 wire 仍会在
第一个 secondary 尝试之前被展开。不采用。

## 运行时设计

### 共享的 secondary 尝试操作

把现有 ranking 循环中对单个 fault 的操作抽成一个 C++ 内部 helper。
它必须按以下顺序执行：

1. 使用全局 attempted-ID 集合保证当前 Primary DTC 中只尝试一次。
2. 记录 attempted ID 和 DTC 调用计数。
3. 运行 `stuck_at_podemx_secondary()` 并累计 backtracks。
4. 如果 PODEMX 成功，检查 proposed cube 是否仍检出所有 preserved faults。
5. 只有通过 preserved-fault 检查时才接受 cube 并记录 embedded ID。
6. 恢复 accepted good-circuit cube 并运行 simulation。
7. 返回当前目标 PO 是否已经不再为 `U`。

baseline 和 RL 必须共用这个 helper，避免求解与回滚语义漂移。

### Heuristic/native baseline：TDF lazy

对每个按 `cktout` 顺序找到的 unknown PO：

1. 创建以该 PO 为起点的 FIFO wire 队列和空 fault 队列。
2. 只当 fault 队列为空且目标 PO 仍为 `U` 时，展开下一个未访问
   的 `U` wire。
3. 展开 wire 时，按稳定 `udflist` 顺序把当前 eligible faults 放入
   fault 队列，并按 gate fan-in 顺序把未访问的 `U` inputs 放入
   wire 队列。
4. 对 fault 队列队首执行共享 secondary 尝试操作。
5. 如果目标 PO 已不是 `U`，立即丢弃当前 fault/wire 队列尾部并转入
   下一个 unknown PO。
6. 如果 PO 仍为 `U` 且 fault 队列已空，才回到步骤 2 继续 BFS。
7. 当已展开 wire 数达到 `select_fault_try` 或 wire/fault 队列都为空时，
   停止该 PO。

`select_fault_try` 保持现有配置：`ncktin <= 32` 时为 15，否则为 100。
预算统计实际首次出队并展开的唯一 `U` wire。

候选继续使用现有资格过滤：排除 active Primary、`test_tried`、
`detect == TRUE`、`detect == REDUNDANT`、本次 Primary DTC 已尝试 ID，并按 fault
ID 保留第一次出现。

`StuckAtSession.step()` 直接调用该 lazy baseline 路径。
`run_stuck_at_ordered()` 继续通过 `step_stuck_at()` 执行，因而自动获得同样语义。

### RL：完整 batch 排序

RL 的分阶段公共协议保持不变：

1. `begin_step(primary)` 在 Primary 成功后，从第一个 unknown PO 开始反向
   BFS。
2. 在该 PO 的 `select_fault_try` 预算内收集全部 eligible faults，形成
   完整 candidate batch。
3. Python 必须向 `rank_dtc_candidates()` 提交该 batch 的完整 permutation。
4. C++ 按 permutation 逐个执行共享 secondary 尝试操作。
5. 每尝试一个 fault 都检查当前 PO。PO 一旦不是 `U`，立即停止，
   未执行的 permutation 尾部不记为 attempted。
6. 若整个 permutation 耗尽后 PO 仍为 `U`，跳过该 PO 并从下一个
   unknown PO 重新收集完整 batch。

因此，模型对完整 batch 排序，但训练仍只对真正执行的 prefix
计算 log-probability 和 entropy。

## 状态与接口

- 公开 Python API 签名不变。
- `begin_step()` 始终表示 RL/显式 ranking 路径；它不得因 baseline
  的修改而返回 lazy 小批次。
- `step()` 始终表示 heuristic convenience 路径；它不再内部将完整
  batch 原样提交给 ranking API。
- `run_stuck_at_ordered()` 的 `ordered_fault_ids` 只决定 Primary 顺序，不解释为
  secondary 顺序。
- 一个 Primary DTC 内的 attempted-ID 集合、accepted cube 和 preserved-fault
  集合在 baseline/RL 两条路径中具有相同生命周期。

## 日志与可观测性

- RL 继续记录完整 batch 的 PO、candidate count、budget、visited wire
  count、attempted prefix 和 embedded prefix。
- baseline 日志不得声称已构建完整 candidate batch。它应记录 lazy PO
  开始/结束、实际展开 wire 数、实际 attempted/embedded 数和结束原因
  (`po_resolved`/`budget_exhausted`/`cone_exhausted`)。
- 现有 step result 中的 `dtc_attempted_fault_ids`、`dtc_embedded_fault_ids` 和
  DTC 计数保持真实执行语义。

## 测试设计

1. **RL 完整 batch**：构造多层 unknown cone，验证 `begin_step()` 在返回前
   已包含预算内近层和深层候选，且只接受完整 permutation。
2. **RL 逐 fault PO-stop**：将能解析 PO 的 fault 排在第一位，验证只执行
   第一个 fault，排序尾部不出现在 attempted IDs 中。
3. **baseline lazy 展开**：构造近层候选能解析 PO、深层仍有候选的
   cone，验证 baseline 不展开深层 wire。
4. **baseline lazy 续搜**：近层候选失败后 PO 仍为 `U`，验证只在近层
   fault 队列耗尽后展开下一个 wire。
5. **入口一致性**：对同一固定 Primary 顺序，`StuckAtSession.step()` 与
   `run_stuck_at_ordered()` 得到相同的模式、fault 状态、coverage 和 DTC 计数。
6. **过滤与重汇**：验证 detected/redundant/already-attempted faults 不会进入
   lazy fault 队列，重汇 wire/fault 不被重复执行。
7. **15/100 预算**：分别验证小电路和大输入电路的 lazy wire 展开上限。
8. **回归**：运行 C++ fault-mapping 测试、Python environment/policy/trainer 测试
   和运行时日志测试。

## 验收标准

1. 所有 heuristic/native baseline 入口均在 fault 队列为空时才继续展开
   BFS wire。
2. baseline 发现候选后立即尝试，不预先构建预算内的完整 batch。
3. RL 仍获得当前 PO 在预算内的完整 eligible candidate batch，并必须
   返回完整 permutation。
4. baseline 和 RL 每执行一个 secondary 后都重新 simulation 并检查当前
   PO；PO 不再为 `U` 时立即停止。
5. RL 未执行的 permutation 尾部不计入 attempted prefix 或训练概率。
6. baseline 和 RL 使用相同的 secondary 求解、cube 接受、回滚和
   preserved-fault 规则。
7. `StuckAtSession.step()` 与 `run_stuck_at_ordered()` 对同一 Primary 顺序的
   heuristic 结果一致。
