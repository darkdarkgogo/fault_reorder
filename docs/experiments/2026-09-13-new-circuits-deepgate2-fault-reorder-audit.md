# 新电路 DeepGate2 + fault reorder RL 可运行性审计

日期：2026-09-13

## 结论

当前状态是：**数据转换与 fault embedding 对齐已经完成。** 原 ABC AIG 已替换为 anchor-preserving AIG；DeepGate2 读取新 AIG，fault catalog 与 ATPG 继续读取原始 BENCH。全部 1,030 个电路的原始 collapsed fault 都有严格对齐的 257 维 embedding。

本文前半部分保留改造前的审计数据，用来解释为什么不能直接使用旧 ABC AIG；“全量改造结果”记录实施后的最终状态。

## 全量改造结果

生成脚本为 `scripts/generate_anchor_aig.py`。正式输出直接位于 `datasets/train_AIG` 和 `datasets/validation_AIG`，每个电路包含五个同 stem 文件：

```text
<stem>.bench                  # anchor-preserving AND/NOT AIG
<stem>.aigmap.json            # 原始信号、helper、boundary 到图节点的映射
<stem>.faults.json            # 原始 BENCH 的 collapsed fault catalog 与 anchor rows
<stem>.gates.npz              # DeepGate2 hs/hf，维度 [node_count, 128]
<stem>.fault_embeddings.npz   # fault 特征，维度 [fault_count, 257]
```

| split | 电路 | collapsed fault embeddings | uncollapsed faults | feature nodes | helpers | 文件 |
|---|---:|---:|---:|---:|---:|---:|
| train_AIG | 1,024 | 408,690 | 692,436 | 366,579 | 97,470 | 5,120 |
| validation_AIG | 6 | 298,222 | 554,854 | 228,246 | 88,182 | 30 |
| 合计 | 1,030 | 706,912 | 1,247,290 | 594,825 | 185,652 | 5,150 |

全量生成先写 staging，再逐电路验证 AIG 确定性 lowering、所有原始信号 anchor、fault 身份、receiver pin、SA、equivalence count、NPZ 行绑定、DeepGate2 provenance 和所有摘要 hash。验证全部成功后才原子交换正式目录。安装后的独立 `--check-only` 再次通过：训练集 408,690 行、验证集 298,222 行，缺失 fault 为 0。

`b12_C` 安装前后均在原始 `datasets/validation/b12_C.bench` 上运行 ordered PODEM，结果保持：3,444 个 collapsed faults、5,906 个 uncollapsed faults、218 条 pattern、覆盖 5,906/5,906、0 aborted。完整数据集摘要和逐文件 SHA256 保存在 `datasets/anchor_aig_dataset.json`。

### 强化学习 smoke test

训练 loader 已支持双路径 manifest：`bench` 指向原始网表，`aig_bench` 与 `aigmap` 绑定 embedding 来源，新数据不要求 `.faultmap`。使用 `configs/anchor_smoke_train.json` 对 `train_0001_iscas89_s13207_g7920` 进行了两轮真实 REINFORCE 训练：

```text
collapsed faults                 = 96
uncollapsed faults               = 248
原始 catalog order patterns      = 15
round 1 sampled order patterns   = 20, reward = -5
round 2 sampled order patterns   = 19, reward = +1
best deterministic patterns      = 17
best coverage                    = 248 / 248
latest checkpoint round          = 2
exported ranking rows            = 96（rank 1..96，全部唯一）
```

baseline、采样 permutation、policy loss、有限梯度、Adam 更新、`latest.pt`、`best.pt`、round-2 断点恢复和独立 `evaluate` 均成功。该结果证明完整 RL → ordered PODEM 控制链可以运行；两轮 smoke 的 best 比原始顺序多 2 条 pattern，因此只证明工程链路正确，不表示模型已经收敛或优于 baseline。

主要阻塞不在 DeepGate2 或 PODEM，而在两条数据路径之间缺少完整且可信的映射：

```text
datasets/*_AIG/*.bench ──→ DeepGate2 ──→ AIG node embedding
                                              │
                                              │ 缺少 100% anchor mapping
                                              ▼
datasets/*/*.bench ──→ original fault catalog ──→ order ──→ original PODEM
```

新数据中的 ABC AIG 保留了相同的 PI/PO，但大量内部节点被重命名、重写或删除。验证集按名称只能完整定位约 4.1%–14.4% 的原始 collapsed faults，六个电路合计为 15,924 / 298,222，即 5.34%。因此不能把剩余 fault 静默映射到相近节点、零向量或图级向量；这样程序虽然可能启动，训练目标却已经不是原始网表 fault reorder。

采用的做法是从每份原始 BENCH 生成一份**保留每个原始信号 anchor 的非优化 AIG**，同时写出 sidecar mapping。DeepGate2 仍然只读取 AIG，ATPG 仍然只读取原始 BENCH。

## 新数据盘点

目录配对情况：

| split | 原始 BENCH | AIG BENCH | 同名配对 | PI/PO 不一致 |
|---|---:|---:|---:|---:|
| train / train_AIG | 1,024 | 1,024 | 1,024 | 0 |
| validation / validation_AIG | 6 | 6 | 6 | 0 |

训练集一共包含 141,873 个原始门，配对 AIG 包含 164,858 个门。验证集一共包含 128,316 个原始门，配对 AIG 包含 141,092 个门。

原始 BENCH 的门类型只有 `AND/NAND/OR/NOR/NOT`，最大 fan-in 为 2；配对 AIG 只有 `AND/NOT`。因此这批原始 BENCH 不需要为了 PODEM 再做 binary gate 展开，也不会触发当前 ordered PODEM 对物理 `XOR/EQV` 的拒绝逻辑。

训练集内部原始门名在 AIG 中的保留比例中位数为 30.25%，范围为 2.01%–64.62%；验证集更低，中位数为 3.07%，范围为 2.22%–9.67%。名字相同只是可检查的直接映射下界，不能代替正式 sidecar，也不能证明名字不同的节点在功能上对应。

## 已完成的实测

### 1. DeepGate2 可以读取新 AIG 并生成 embedding

使用固定的官方 `python-deepgate` 源码和 checkpoint，在 `validation_AIG/b12_C.bench` 上实测：

```text
AIG BENCH gates       = 1,772
feature graph nodes   = 2,148
DeepGate2 hs shape    = (2148, 128)
DeepGate2 hf shape    = (2148, 128)
all values finite     = true
backend               = official-python-deepgate
checkpoint verified   = true
```

checkpoint SHA256 为：

```text
9bc4a0c1f8fc57cc3aa0498dd8737af561ca71c26ca5332288d12a91a308f4d5
```

本机应使用 `C:\Users\acer\.conda\envs\d2l\python.exe`。仓库 `.venv` 中没有 PyTorch，不能用于 DeepGate2 或 RL scorer。

### 2. PODEM 可以直接在原始网表上生成 fault catalog

`validation/b12_C.bench` 的原始网表实测结果：

```text
collapsed faults      = 3,444
uncollapsed faults    = 5,906
```

这份 catalog 是后续 embedding、RL action 和 ATPG 的唯一 fault source of truth。AIG 不生成 fault list。

### 3. 原始网表 ordered ATPG 可以运行，而且 order 确实有效

在相同原始网表、相同 fault 集、seed=14、backtrack limit=5000 下：

| order | patterns | covered equivalent faults | coverage | PODEM calls | backtracks |
|---|---:|---:|---:|---:|---:|
| catalog 原顺序 | 218 | 5,906 | 100% | 218 | 169 |
| 完全逆序 | 208 | 5,906 | 100% | 208 | 214 |

逆序少 10 个 pattern，说明 C++ 入口已经能接受一份完整 fault ID permutation，并且 fault order 会影响 test compaction 结果。这里没有证明逆序普遍更好，只证明 reorder 控制链有效。

### 4. 现有新数据 manifest 不能通过校验

当前 `configs/deeptpi_itc22_test_9.json` 和 `configs/deeptpi_itc22_train_512.json` 仍指向 `datasets/deeptpi_itc22/...`，但该目录不存在。当前校验首个错误为：

```text
missing circuit artifact: datasets/deeptpi_itc22/test/b12_C/b12_C_binary.bench
```

即使改掉路径，当前 manifest/schema 仍把一个 `bench` 同时当作 embedding 图来源和 ATPG 输入，不支持“配对 AIG 生成 embedding、原始 BENCH 跑 ATPG”的明确分流。

## 验证集 fault 到现有 ABC AIG 的直接映射率

映射判定采用当前 257 维 fault feature 所需的最小信息：

- GO fault：必须找到对应原始驱动信号的 AIG 节点；
- GI fault：必须同时找到接收门输出和被故障输入信号的 AIG 节点；
- PI/PO dummy boundary 按端口顺序处理；
- 只接受明确同名节点，不猜测 `new_n*` 的来源。

| circuit | 原始门 | AIG 门 | collapsed faults | 可直接映射 | 比例 |
|---|---:|---:|---:|---:|---:|
| b12_C | 1,231 | 1,772 | 3,444 | 497 | 14.43% |
| b15_C | 10,480 | 14,513 | 25,910 | 2,040 | 7.87% |
| b17_C | 37,379 | 47,580 | 88,918 | 5,990 | 6.74% |
| b20_C | 22,557 | 21,749 | 51,208 | 2,139 | 4.18% |
| b21_C | 23,100 | 22,631 | 52,362 | 2,139 | 4.09% |
| b22_C | 33,569 | 32,847 | 76,380 | 3,119 | 4.08% |

这些比例还包含较容易映射的 PI/PO boundary fault，因此不能据此认为内部 fault mapping 足够。现有 AIG 只能证明组合电路在 PO 层面配对，不能证明每个原始内部 fault site 都有一个唯一 AIG node embedding。

## fault 应该怎么定义和传递

### 1. fault catalog 的来源

先让 PODEM 在**原始 BENCH**上运行 `generate_fault_list()`，得到 collapsed stuck-at catalog。每个 catalog row 至少保留：

```text
fault_id
node_name
input_wire_name
io                 # GO 或 GI
input_index
input_occurrence
fault_type         # SA0 或 SA1
eqv_fault_num
catalog_row
```

稳定 ID 使用当前格式：

```text
<gate>:GO:sa0
<gate>:GO:sa1
<receiver>:GI<input_index>:sa0
<receiver>:GI<input_index>:sa1
```

相同物理位置产生重复 ID 时保留 `@dupN`。RL 不使用临时指针或 PODEM 的运行时 `fault_no` 作为跨 artifact 身份。

### 2. collapsed 和 uncollapsed 的区别

RL 排序的是 collapsed catalog rows。每个 row 的 `eqv_fault_num` 表示它代表多少个 uncollapsed faults。

```text
uncollapsed_total = sum(eqv_fault_num)
```

报告覆盖率时使用：

```text
covered_equivalent_faults =
    detected_equivalent_faults + redundant_equivalent_faults

fault_coverage =
    covered_equivalent_faults / uncollapsed_total
```

aborted fault 不计入覆盖。训练模型不能通过丢 fault、改 fault collapsing 或降低覆盖率来减少 pattern 数。

### 3. 原始 fault 到 AIG anchor 的规则

DeepGate2 为 AIG 节点生成 `hf[128]`。当前 scorer 输入可以继续保持 257 维：

```text
GO: [driver_anchor_hf, driver_anchor_hf, sa_value]
GI: [receiver_anchor_hf, connected_signal_anchor_hf, sa_value]
```

具体映射：

- 普通 GO：原始 gate output → 该原始信号的唯一 AIG anchor；
- 普通 GI：原始 receiver output → receiver anchor，同时原始 input wire → source anchor；
- PI dummy GO：按原始 `INPUT(...)` 顺序映射到 PI anchor；
- PO dummy GI：按原始 `OUTPUT(...)` 顺序映射到显式 PO boundary anchor，同时保留驱动信号 anchor；
- `sa_value` 明确保存 0/1；
- `eqv_fault_num` 只用于覆盖统计，不混入 fault ID；
- AIG helper node 永远不能创建新的 fault action。

任何一个 catalog row 找不到完整 anchor 时，整个电路导入失败。不能跳过、补零或退化为随机 embedding。

### 4. RL 输出与 ATPG 输入

scorer 对每个 collapsed fault embedding 输出一个 scalar score。训练时按 Plackett–Luce/Gumbel 采样完整 permutation；评估时按 score 降序，使用原 catalog row 稳定破同分。

传给 C++ 的必须是**所有原始 catalog fault IDs 的恰好一次排列**：

```text
len(order) == fault_count
set(order) == set(catalog_fault_ids)
no duplicate IDs
no AIG helper IDs
```

C++ 重新从同一份原始 BENCH 生成 catalog，按 ID 找回原始 fault pointer，然后只改变 `flist_undetect` 的遍历次序。PODEM implication、fault injection、fault simulation、fault dropping 和 pattern 统计全部继续使用原始网表。

## 为什么当前 ABC AIG 不能直接补一个名字映射

ABC 优化会进行结构哈希、逻辑重写、反相边合并和冗余节点删除。一个原始信号可能出现以下情况：

1. 名字保留且功能保留，可以直接映射；
2. 功能保留但节点改名为 `new_n*`，需要 synthesis-time mapping 或形式等价证明；
3. 只剩 complemented edge，没有独立 node；
4. 被优化进其他逻辑，不再存在等价单节点；
5. 多个原始信号被合并到一个 AIG node。

对第 3–5 类，事后按名字或拓扑距离都无法得到当前 fault feature 所要求的唯一 anchor。形式等价工具最多能恢复一部分对应关系，也不能保证每个原始信号都有物理 node。因此应该在生成 AIG 时保留 anchor，而不是在优化后的 AIG 上猜。

## 推荐改造

### A. 数据 schema 分离

把 manifest 从当前单 `bench` 改成至少以下字段：

```json
{
  "name": "b12_C",
  "atpg_bench": "../datasets/validation/b12_C.bench",
  "aig_bench": "../generated/validation/b12_C.anchor_aig.bench",
  "aigmap": "../generated/validation/b12_C.aigmap.json",
  "fault_catalog": "../generated/validation/b12_C.faults.json",
  "embeddings": "../generated/validation/b12_C.fault_embeddings.npz"
}
```

所有文件保存 SHA256，并在训练、恢复、评估前重新核对。

### B. 生成 anchor-preserving AIG

推荐直接复用并完善 `fault_embedding.circuit.build_graph` 的结构化转换：

- 不运行会删除内部信号的逻辑优化；
- 每个原始 PI 和 gate output 都有一个唯一 anchor；
- NAND/OR/NOR/NOT 用 AND/NOT helper 展开；
- 必要时物化 BUF/双反相，避免原始 fault site 与上游信号合并；
- helper 记录 `origin=null`，anchor 记录原始 signal name；
- 对所有原始 anchor 做逻辑等价检查，不只检查 PO；
- 输出 `aigmap.json`，不能只把映射留在内存。

这样仍符合“DeepGate2 读取 AIG 得到 embedding”，同时保证 embedding 能回到原始 fault。

如果必须继续使用当前 `train_AIG` / `validation_AIG`，则需要重新运行其生成流程：把每个原始内部信号作为受保护 observation point 或额外输出，并在 ABC 运行时导出 original signal → AIG literal sidecar。仅凭当前两个 BENCH 文件无法安全重建全量映射。

### C. 原始网表 ATPG 不再强制 faultmap

当前 `PodemEnvironment` 和 manifest 强制要求 `.faultmap`，这是以前“ATPG 也跑 binary-transformed BENCH”留下的约束。新数据应支持：

```text
catalog_stuck_at(original_bench, "")
run_stuck_at_ordered(original_bench, "", ordered_fault_ids, 5000, 14)
```

`fault_catalog` JSON 是冻结身份和 embedding 对齐的 artifact，不应被当成把 fault 搬到 AIG 或 binary BENCH 的 `.faultmap`。

### D. 分阶段 smoke test

不要一开始直接训练 1,024 个电路。建议顺序：

1. 选一个小训练电路，验证原始 catalog → anchor map → DeepGate2 → 257 维 embedding 全量对齐；
2. 用 `b12_C` 做原顺序、逆序、模型顺序 ATPG，要求 fault 集和覆盖完全一致；
3. 选 8–16 个小训练电路跑一轮 RL，检查 checkpoint、reward、恢复和 ranking 导出；
4. 扩到 1,024 个训练电路；
5. 最后对 6 个大型 validation 电路做独立评估。

训练集电路较小，但每轮仍要对每个电路运行一次 sampled ATPG，并周期性运行 deterministic evaluation；验证电路最大有 88,918 个 collapsed faults。应先记录逐电路 wall time，再决定 batch size 和 evaluation frequency。

## 验收条件

完整链路只有同时满足以下条件才算“可以跑动”：

1. 1,024 个 train pair 和 6 个 validation pair 全部通过 PI/PO 配对校验；
2. 每个原始 catalog row 都有完整、唯一、可验证的 AIG anchor mapping；
3. embedding 行数、fault IDs、SA 极性和 `eqv_fault_num` 与原始 PODEM catalog 完全一致；
4. embedding 维度为 `[fault_count, 257]`，数值有限，provenance 是固定官方 DeepGate2；
5. AIG helper node 不出现在 fault IDs 中；
6. RL 输出是原始 catalog 的完整 permutation；
7. ATPG 运行时打开的是 `atpg_bench`，不是 `aig_bench`；
8. 原顺序和模型顺序使用相同 seed、backtrack limit 和 fault set；
9. 模型顺序的 `covered_equivalent_faults` 不低于原顺序；
10. checkpoint 绑定 manifest、原始 BENCH、AIG、aigmap、catalog、embedding、DeepGate2 和 PODEM 二进制 digest。

## 最终判断

- **DeepGate2：可运行。** 新 AIG 格式和官方 checkpoint 已实测通过。
- **原始网表 stuck-at PODEM：可运行。** 新数据门型受支持，b12_C 已完整跑通。
- **fault reorder 接口：可运行。** b12_C 原序与逆序产生不同 pattern 数且覆盖一致。
- **现有 RL 直接读取新数据：不可运行。** manifest 指向旧的缺失 artifact，schema 没有分离 AIG 和 ATPG BENCH。
- **现有 ABC AIG 用于逐 fault embedding：不合格。** 缺失全量 anchor sidecar，验证集直接映射率只有 5.34%。
- **anchor-preserving AIG 与原始 catalog 对齐：已完成。** 全部 706,912 个 collapsed fault embedding 已生成并通过安装后独立校验。
- **RL/PODEM 双路径原则：保持不变。** RL 使用生成的 fault embedding 排序，C++ PODEM 仍接收原始 BENCH catalog 的完整 fault ID permutation。
