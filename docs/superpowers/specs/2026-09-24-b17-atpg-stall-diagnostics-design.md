# b17_C ATPG 卡点诊断设计

## 背景

基线 commit `35027ea` 上，b17_C 的 native baseline 在 step 3 Primary PODEM
成功后长时间没有外部输出。当前控制流为：

```text
PodemSession.step(primary)
  -> cpp_podem.StuckAtSession.step(primary)
  -> ATPG::step_stuck_at(primary)
  -> ATPG::begin_stuck_at_step_impl(primary, false)
  -> ATPG::run_stuck_at_lazy_dtc()
  -> ATPG::attempt_stuck_at_dtc_secondary()
  -> ATPG::stuck_at_podemx_secondary()
```

Python 只有在 `step()` 返回后才能打印完成信息，因此必须在 C++ lazy DTC
和单个 secondary PODEMX 内增加可刷新的观测点。

## 目标

- 定量判断耗时发生在 lazy DTC 的 PO/wire/secondary 总工作量，还是单个
  secondary PODEMX 搜索内部。
- 判断 `podemx_backtrack_limit=50` 是否实际约束了搜索，或大量 forward
  decision 在 backtrack 很少时消耗了主要时间。
- 提供 b17_C 专用 native baseline 驱动，默认运行全部 Primary，同时支持限制
  Primary 数量以快速复现。
- 保持 ATPG 算法、fault 顺序、候选规则、预算和结果不变。

## 非目标

- 不加入 decision、iteration、wall-clock 或 Primary 级停止预算。
- 不调整 lazy DTC wire budget、Primary backtrack limit 或 secondary
  backtrack limit。
- 不改变 PI cube monotonic invariant、fault dropping、DTC eligibility 或
  BFS/FIFO 顺序。
- 不加载 RL 模型，不经过 `begin_step()`/`rank_dtc_candidates()` 的完整 batch
  排序路径。
- 诊断驱动不调用 `finish()`，因此不进入 STC。

## 方案选择

采用运行时环境变量 `PODEM_DTC_DIAGNOSTICS=1` 控制新增日志。

- 不采用始终开启：正常训练和验证会收到大量 DTC/PODEMX 输出。
- 不采用单独诊断扩展：会增加构建产物和复现差异。
- 环境变量不改变公开 Python/C++ API；专用驱动会在创建 native session 前设置
  该变量。

只有值严格为 `1` 时启用。未设置或其他值均关闭。`run_stuck_at_lazy_dtc()`
和每次 `stuck_at_podemx_secondary()` 调用各自在入口读取一次当前进程环境，
避免在搜索内循环反复调用 `getenv()`；该开关只影响日志和诊断计数，不参与
控制流决策。

## C++ 诊断日志

所有新增日志写入 `stderr`，每条记录以单行 `key=value` 形式输出，并在每条后
立即 `fflush(stderr)`。fault 和 wire 名称来自已有 identifier/name，不做二次
解释。

### Lazy DTC start

`run_stuck_at_lazy_dtc()` 入口记录：

```text
[ATPG][DTC-LAZY] start primary=<id> outputs=<count> wire_budget=<budget> secondary_backtrack_limit=50
```

`primary` 使用 `stuck_at_active_step.selected_fault_id`，`outputs` 使用
`cktout.size()`，`wire_budget` 使用本次实际选择的 small/default budget，
`secondary_backtrack_limit` 使用 `podemx_backtrack_limit` 的实际值。

### Lazy DTC 计数口径

- `po_processed`：进入时为 `U`、实际开始执行 lazy DTC 扫描的 PO 数。每个此类
  PO 只计一次；进入时非 `U` 的 PO 不计。
- `current_po`：当前被扫描的 `unknown_po->name`。
- `expanded_wires`：本 Primary 所有已处理 PO 的累计 wire 展开数。
- `current_po_wires`：当前 PO 的 `expanded_wires` 局部值。
- `attempted`：本 Primary 已登记到 `dtc_attempted_fault_ids` 的 secondary 数。
- `embedded`：本 Primary 已登记到 `dtc_embedded_fault_ids` 的 secondary 数。
- `backtracks`：本 Primary 已完成 secondary 调用累计到
  `stuck_at_active_dtc_backtracks` 的 backtrack 数。
- `elapsed_s`：从 `run_stuck_at_lazy_dtc()` 入口开始的 steady-clock 秒数。

全局 `expanded_wires` 在某个 wire 真正通过去重和 `U` 过滤、即当前实现递增
局部 `expanded_wires` 的同一位置递增。它不是 visited set 的大小，也不包含
被跳过的 wire。

### Lazy DTC progress

```text
[ATPG][DTC-LAZY] progress primary=<id> po_processed=<N> current_po=<name> expanded_wires=<N> current_po_wires=<N> attempted=<N> embedded=<N> backtracks=<N> elapsed_s=<seconds>
```

在以下任一阈值自上次 progress 后首次满足时打印，并同步更新各阈值基线：

- 新处理 100 个 unknown PO；
- 新登记 500 次 secondary attempt；
- 距上次 progress 至少 10 秒。

检查点放在：开始一个 unknown PO 后、每次 wire 展开后、每个 secondary 调用
返回后，以及结束一个 PO 后。外层时间条件无法在一个尚未返回的 secondary
内部触发；该情况由 PODEMX 内层日志覆盖。

### Lazy DTC done

正常离开 `run_stuck_at_lazy_dtc()` 时记录：

```text
[ATPG][DTC-LAZY] done primary=<id> po_processed=<N> expanded_wires=<N> attempted=<N> embedded=<N> backtracks=<N> elapsed_s=<seconds>
```

由于函数没有诊断新增的早退或异常处理，`done` 只表示现有 lazy DTC 控制流正常
完成。

### PODEMX progress

`stuck_at_podemx_secondary()` 为当前调用维护局部诊断计数：

- `iterations`：每次进入 `while (true)` 循环体时加一；
- `decisions`：`valid_decision` 为真并将新 decision 压入 stack 时加一；
- `depth`：打印时的 `decisions.size()`，表示当前 stack 深度；
- `backtracks`：既有 `backtracks` 输出参数的当前值；
- `elapsed_s`：从该 secondary 函数入口开始的 steady-clock 秒数。

每个 secondary 最多每 10 秒打印一次：

```text
[ATPG][PODEMX] progress primary=<primary> secondary=<secondary> iterations=<N> decisions=<N> depth=<N> backtracks=<N> elapsed_s=<seconds>
```

定时检查在每次主循环开始执行。日志计数不改变现有 backtrack limit 判断位置或
任何求解分支。

### PODEMX slow-done

secondary 返回前，若实际耗时至少 2 秒，记录：

```text
[ATPG][PODEMX] slow-done primary=<primary> secondary=<secondary> status=<detected|failed|aborted> iterations=<N> decisions=<N> backtracks=<N> elapsed_s=<seconds>
```

状态映射严格对应现有返回值：`TRUE=detected`、`FALSE=failed`、
`MAYBE=aborted`。日志在恢复现有清理流程后、函数返回前输出，不增加新的返回
状态。

## b17_C 专用驱动

新增一个仓库脚本，直接导入已构建的 `cpp_podem` 并创建：

```python
cpp_podem.StuckAtSession(
    bench_path,
    faultmap_path,
    backtrack_limit=100,
    seed=14,
    dtc_enabled=True,
    stc_enabled=False,
)
```

驱动在创建 session 前设置 `PODEM_DTC_DIAGNOSTICS=1`，使用
`datasets/validation/b17_C.bench`。该 validation bench 直接由 native ATPG
生成 collapsed fault catalog，因此 `faultmap_path` 传空字符串。它不构建
embedding、不加载 checkpoint、不实例化 RL trainer。

循环每次读取 `remaining_fault_ids()`，选择返回列表中的第一个 ID，再调用
`session.step(primary)`。这保持 native baseline 的现有 Primary 顺序，而不是
把 PDF 中观察到的三个 ID硬编码为运行顺序。

默认持续运行，直到 `remaining_fault_ids()` 为空。可选参数 `--max-steps N`
要求 `N >= 1`，用于只运行前 N 个 Primary；达到上限后正常退出。两种模式都不
调用 `session.result()`/Python wrapper `finish()`，避免进入 STC。

每步对齐输出：

```text
[B17-DEBUG] STEP START step=<N> selectable=<N> primary=<id>
[B17-DEBUG] STEP DONE step=<N> remaining=<N> dtc_attempted=<N> dtc_embedded=<N> step_dtc_calls=<N> step_dtc_backtracks=<N> elapsed_s=<seconds>
```

`step_dtc_calls` 和 `step_dtc_backtracks` 通过相邻 step 返回的累计
`current_dtc_secondary_calls`/`current_dtc_backtracks` 作差得到；首步以零为前值。
输出到 `stderr` 并立即刷新，保证重定向日志时可见。

## 错误处理

- bench 或 native extension 缺失时，驱动在创建 session 前给出明确错误并以
  非零状态退出。
- `--max-steps` 非正整数由参数解析拒绝。
- C++ 诊断日志只读取已有对象和局部计数；不捕获或吞掉现有异常。
- 正常模式关闭诊断时，不执行格式化和输出；计数器只存在于函数栈上，不写入
  session 结果。

## 测试与验证

### 自动化测试

使用现有小型 DTC fixture 覆盖：

1. 未设置环境变量时，不出现新增 `[ATPG][DTC-LAZY]` 或
   `[ATPG][PODEMX]` 日志。
2. 开启后，lazy DTC 至少输出成对的 `start`/`done`，字段使用实际 Primary、
   output 数、budget 和 backtrack limit。
3. 日志开启与关闭两次运行返回的 target status、generated vector、remaining
   IDs、attempted/embedded IDs 和 backtrack 计数完全相同。
4. 将 PODEMX progress/slow 阈值实现为文件内常量；单元测试验证日志格式和
   状态映射，不通过降低生产阈值改变求解行为。
5. 驱动参数测试覆盖默认无限步、`--max-steps` 截断、顺序取 remaining 首项，
   以及不调用 finalize。

### 构建与回归

- 重建 `cpp_podem` 原生扩展。
- 运行 `PODEM/tests` 与受影响的 Python 测试。
- 对现有 fixture 比较诊断开/关结果，确认诊断为只读观测。

### b17_C 诊断运行

先以 `--max-steps 3` 快速确认 step 3 可持续输出两层日志，再执行默认完整运行。
完整运行可能耗时很长；只要进程仍在执行，日志应能区分：

- PO、wire、attempt 与 backtrack 总工作量持续增长；
- 少数 PO 候选密集；
- 单个 secondary 的 PODEMX 长时间搜索；
- forward decisions 很大但 backtracks 很小；
- backtracks 快速达到 50。

诊断阶段不依据观察结果自动修改任何预算。性能优化作为后续独立设计处理。

## 验收标准

1. 正常运行默认没有新增诊断日志，ATPG 结果与现有回归一致。
2. 开启诊断后，能持续看到 lazy DTC 外层和 PODEMX 内层观测点。
3. 对任意 Primary，可读出处理的 unknown PO、展开 wire、attempt、embedded、
   DTC backtrack 和耗时。
4. 对任意慢 secondary，可读出 iteration、decision、当前 depth、backtrack、
   状态和耗时。
5. b17_C 驱动默认运行全部 Primary，`--max-steps N` 只限制 Primary 数，不改变
   其选择顺序。
6. 驱动不加载 RL、不走 ranked DTC、不调用 STC。
7. 日志足以判断 `podemx_backtrack_limit=50` 是否构成实际限制因素。
