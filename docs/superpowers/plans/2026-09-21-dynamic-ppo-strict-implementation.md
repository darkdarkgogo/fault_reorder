# 动态 Fault Ranking PPO 严格实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有动态 Primary REINFORCE 训练器改成严格 coverage-first 的 Actor-Critic PPO，并让策略生成的 DTC ranking 真实进入 C++ 求解流程。

**Architecture:** C++ session 接收 Primary 与完整 secondary permutation，Python 校验实际 attempted 连续前缀。Actor-Critic 对每个动态 remaining set 产生 ranking logits 与 state value；每个完整 circuit rollout 后执行 GAE 和 4 次 PPO，独立 validation 选择 best，checkpoint 在每个 circuit 后提交。

**Tech Stack:** C++17、pybind11、Python 3、PyTorch、NumPy、pytest。

**Spec:** `docs/superpowers/specs/2026-09-20-dynamic-ppo-strict-design.md`

## 全局约束

- Primary backtrack limit 固定为 100；DTC secondary backtrack limit 固定为 50。
- Training rounds 固定为 5；每条 trajectory 做 4 次 PPO epoch。
- `alpha=0.1`、`beta=10.0`、`gamma=1.0`、`gae_lambda=0.95`、`clip=0.2`。
- Training 与 `configs/anchor_validation_6.json` 完全隔离。
- `dtc_attempted_fault_ids` 必须是 requested secondary ranking 的连续前缀。
- Checkpoint schema 升级为 4，version 1-3 必须拒绝。
- 只修改与本功能直接相关的代码和文档。

---

### Task 1: Native ranked-DTC 接口

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/atpg.cpp`
- Modify: `PODEM/src/saf_compaction.cpp`
- Modify: `PODEM/src/python_bindings.cpp`
- Test: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Produces: `ATPG::step_stuck_at(const string&, const vector<string>&)`。
- Produces: `ATPG::run_stuck_at_dtc(fptr, const vector<string>&)`。
- Produces: Python `StuckAtSession.step(primary_fault_id, ranked_secondary_fault_ids)`。

- [ ] 新增失败测试：传入逆序 secondary ranking 后，attempted 顺序必须与输入一致；重复、遗漏、未知 ID 必须失败。
- [ ] 运行对应 native 测试，确认旧实现失败。
- [ ] 修改 C++ API，验证 secondary 参数恰好是其余 selectable faults 的 permutation，DTC 严格按输入遍历。
- [ ] 将 Primary 默认与 protocol 固定值统一改为 100，保留 DTC=50。
- [ ] 运行 `python -m pytest PODEM/tests/test_fault_mapping.py -q`。
- [ ] 提交 `feat: pass policy ranking into native DTC`。

### Task 2: Python 环境连续前缀契约

**Files:**
- Modify: `fault_order_rl/environment.py`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- Produces: `PodemSession.step(fault_id, ranked_secondary_fault_ids)`。
- Guarantees: attempted 是 requested secondaries 的连续前缀，embedded 是 attempted 的保序子序列。

- [ ] 新增环境测试，覆盖完整 permutation、连续前缀、换序、中间跳过、重复和遗漏。
- [ ] 将 `PROTOCOL_CONFIG.primary_backtrack_limit` 与构造参数改为 100。
- [ ] 调用新 binding，并在使用结果前完成 prefix/subsequence 校验。
- [ ] 更新 `PodemEnvironment.run()` 的默认 limit 与协议校验。
- [ ] 运行环境相关 pytest selection。
- [ ] 提交 `feat: validate ranked DTC execution protocol`。

### Task 3: Actor-Critic 与 executed-prefix policy

**Files:**
- Modify: `fault_order_rl/model.py`
- Modify: `fault_order_rl/policy.py`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- Produces: `FaultActorCritic.forward(features) -> (scores, value)`。
- Produces: `sample_ranking(scores, temperature, stochastic, generator=None) -> rows`。
- Produces: `executed_prefix_stats(model, embeddings, remaining_rows, executed_rows, temperature) -> (log_prob, entropy, value)`。

- [ ] 新增模型 shape/gradient、稳定 tie break、executed-prefix log-prob 与 mean entropy 测试。
- [ ] 把 scorer 拆为共享 encoder、actor head、critic head。
- [ ] 实现随机/确定性完整 ranking 与 executed-prefix 逐项 masked probability。
- [ ] 删除训练路径对单 Primary categorical log-prob 的依赖。
- [ ] 运行模型与 policy 相关测试。
- [ ] 提交 `feat: add actor critic ranked fault policy`。

### Task 4: Reward、GAE 与 PPO 数学组件

**Files:**
- Create: `fault_order_rl/reward.py`
- Modify: `fault_order_rl/trainer.py`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- Produces: `step_reward(pattern_increment, newly_detected_eqv, initial_eqv, alpha)`。
- Produces: `target_return(patterns_after_stc, initial_eqv, shortfall, beta)`。
- Produces: `terminal_correction(step_rewards, target)`。
- Produces: `compute_gae(rewards, values, gamma, gae_lambda)`。
- Produces: `normalize_advantages(values, eps)`。

- [ ] 新增 redundant/aborted zero reward、valid/invalid exact return、GAE、单样本 normalization 测试。
- [ ] 实现纯函数与有限值/范围校验。
- [ ] 把 terminal correction 加到最后 transition 后再计算 GAE。
- [ ] 实现 PPO clipped actor loss、critic MSE、executed entropy bonus 与 4 epoch 更新。
- [ ] 运行 reward/GAE/PPO 单测。
- [ ] 提交 `feat: train ranked policy with PPO and GAE`。

### Task 5: 独立 Validation 与 per-circuit checkpoint

**Files:**
- Modify: `fault_order_rl/trainer.py`
- Modify: `fault_order_rl/checkpoint.py`
- Modify: `fault_order_rl/__main__.py`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- Trainer create 接收 training manifest 与 validation manifest。
- `latest.pt` schema 4 保存 active round、next circuit index、RNG 与两个 split provenance。
- `best.pt` 使用 `(validation_shortfall, validation_patterns_after_stc)`。
- `final.pt` 在 Round 5 后无条件保存。

- [ ] 新增 split overlap、validation no-update、best key、schema 4、per-circuit resume 与 final checkpoint 测试。
- [ ] 分离 `train_circuits`、`validation_circuits` 和各自 native baseline。
- [ ] 每个 circuit 的 4 次 PPO 成功后原子提交 latest 与训练记录。
- [ ] 每轮结束只跑 validation，并更新 best；第五轮发布 final。
- [ ] 让 evaluation 支持 best/final 和显式 external manifest。
- [ ] 运行 checkpoint/resume/validation 测试。
- [ ] 提交 `feat: isolate validation and checkpoint each circuit`。

### Task 6: CLI、文档与必要回归

**Files:**
- Modify: `README.md`
- Modify: `docs/fault-order-rl.md`
- Modify: `scripts/run_anchor_linux.sh`
- Modify: `scripts/evaluate_anchor_linux.sh`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- CLI train 支持 `--validation-manifest`，默认 `configs/anchor_validation_6.json`。
- CLI 明确固定 5 rounds、Primary=100、DTC=50 与 schema 4 restart requirement。

- [ ] 更新 CLI/script 测试和用户文档。
- [ ] 构建 native extension。
- [ ] 运行 `python -m pytest tests/test_fault_order_rl.py PODEM/tests/test_fault_mapping.py -q`。
- [ ] 使用 smoke manifest 完成一次 create、一个 circuit PPO update、resume、validation 与 best/final evaluation。
- [ ] 提交 `docs: document strict dynamic PPO workflow`。
