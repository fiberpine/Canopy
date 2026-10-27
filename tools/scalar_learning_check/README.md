# L1实测：安全停止，未完成训练预算

实际结果：24 actor零优化通过；正式28,513条后S3停止，14 actor爆窗。110冻结评测NOT_RUN，不能把工具存在当成这些评测已通过。详细结果见[最新研究交接](../../docs/RESEARCH_REVIEW_BRIEF.md)。

`runtime/`保存实际有限执行器、guard、冻结评测候选、actor导出工具与已执行TF后验检查源码。以`CANOPY_L1_ROOT`读取独立运行目录，框架须放在`$CANOPY_L1_ROOT/framework`，显式提供注册/主机授权与learner端私有SSH文件；本仓库不带个人部署器或凭证。历史deadline已过，不得复用本次注册启动。`tf_postmortem.py`是本次step28513的特定审计快照，不是通用训练CLI。导出/冻结评测路径在本轮没有执行，只通过语法检查。

首个S3停止后仍按原账本排空更新并保存最终模型，故该checkpoint不保留首次危险策略。不得用终点函数查询重写在线历史。所有源/证据的执行SHA与公开字节SHA区别见[发布来源](../../evidence/scalar-limited-learning-20261027/publication-provenance.json)。新增质量判定`tail_guard`为opt-in；实际注册已打开，要求P95/P99均<=1.2*CUBIC+1ms，缺失不能通过。由于安全提前停止，本轮并未应用网络性能判定。

下面是原提案工具说明；`make_learning_registration.py`生成的是原模板，实际登记的执行补充和失败结果以[registration.json](../../evidence/scalar-limited-learning-20261027/registration.json)为准。没有自动长训授权。

---

# Limited learning check (L1) tooling

Read-only helpers for the first finite single-action Orca learning run after the zero-update fixes
(verified IPC v2, small_uniform output head, zero_sum_offset exploration, 64 MiB host buffers).
The three top-level helpers below are read-only/offline. The `runtime/` historical executor snapshot CAN start TF/TCP, change temporary sysctl values and coordinate SSH; do not run it to read the evidence.

| Script | Purpose |
|---|---|
| `make_learning_registration.py` | Writes the registration: training config (historical moving template + validated components + 1.2M budget), host buffer receipts, monitor rules, 110 frozen evaluation jobs, predictions, decision rules. Asserts that only those keys differ from the template. |
| `learning_monitor.py` | Per-actor 30 s windows from `logs/<run>-actor<i>/states.csv`: deterministic action, saturation, cwnd, utilization against goodput capacity, max_bw anchoring, own and privileged reward, alias flag, queue, loss, cumulative exploration. Registered stop rules S1 saturation, S2 low window, S3 explosion. Reads `logs/<run>/training.csv` when present. |
| `adjudicate_learning.py` | Applies the registered decision rules to the frozen evaluation rows (trim10): COLLAPSE_REPRODUCED > POLICY_LIMITED > ANCHORING_LIMITED > LONG_RTT_STATIC > GO_LONG_TRAINING > PARTIAL. Compares with the historical moving checkpoint without using it in a decision. |

```sh
python3 tools/scalar_learning_check/make_learning_registration.py --code-commit <sha> --out learning-registration.json
python3 tools/scalar_learning_check/learning_monitor.py --root <framework> --run learncheck-L1-orca_moving-s3 --out monitor.json
python3 tools/scalar_learning_check/adjudicate_learning.py --evaluations evaluations.csv --registration learning-registration.json --monitor monitor.json
python3 tools/scalar_ipc_diagnostic/window_binding.py <run folders>   # binding_rtt and ss_mode per run
```

Self-tests: `--self-test <tmpdir>` on each script, or `python3 -m unittest tools/test_learning_check.py`.
