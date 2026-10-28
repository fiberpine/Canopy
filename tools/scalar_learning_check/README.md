# 最新（Claude提案，未执行）：critic视野与无模型方向检查

分析见[离健康基座还有多远](../../docs/SCALAR_BASE_ROADMAP.md)。下面三个是离线工具，均不启动TCP，不写checkpoint。

| 文件 | 作用 | 依赖 |
|---|---|---|
| `value_horizon.py` | 理想固定策略传播模型：以Pr[Poisson(τt)≥m]加权；不作为神经critic视野定理；拟合已发布R3/R5记录的Q斜率与R20翻号时间 | NumPy；读已发布证据 |
| `exploration_response.py` | 核查zero_sum偏移重建，按RTT×动作前延迟类报告创新响应及HAC误差；不直接识别持续动作效应 | NumPy；states.csv或E1R的npz |
| `runtime/critic_horizon_replay.py` | H1（τ=0.001，4万次critic-only）、H2（τ=0.005，1万次）记录Q与模型、λ、按RTT×延迟类的dQ/da翻号；E1R导出replay的偏移序列 | TF1，与`first_updates_replay.py`相同的输入和环境变量；单文件可独立复制运行 |
| `runtime/first_updates_replay.py` v3 | 训练过程不变；放行点与终点额外记录逐状态Δπ、dQ1/da和按延迟类的拆分 | 同v2 |

```sh
python3 tools/scalar_learning_check/value_horizon.py --evidence evidence/scalar-first-updates-20261027 --reward-mean 0.706
CANOPY_FIRST_SOURCE=... CANOPY_FIRST_CHECKPOINT_DIR=... CANOPY_FIRST_FRAMEWORK=... \
python3 tools/scalar_learning_check/runtime/critic_horizon_replay.py --job H1 --output /new/output/H1.json   # 同样运行H2、E1R
python3 tools/scalar_learning_check/exploration_response.py --series-npz /new/output/E1R.npz
CANOPY_FIRST_SOURCE=... python3 tools/scalar_learning_check/runtime/first_updates_replay.py --arm R5 --seed 3 --output /new/output/R5-v3-s3.json
```

---

# 真实TF首批更新重放与速度对照

已完成16个离线case，结果见[交接](../../docs/RESEARCH_REVIEW_BRIEF.md)与[实测修正](../../docs/SCALAR_FIRST_UPDATES_DIAGNOSIS.md)。`actor_speed_sim.py`只是同结构/合成输入的数量级演示，不是与真实TF逐值相同的证明。

`runtime/first_updates_replay.py`每次运行一个独立TF1进程，R0原设置、R1仅重抽critics、R2冻结BN统计、R3校准3000后开放、R4 lr1e-5、R5校准后lr1e-6、R6仅lr1e-6。它在内存训练模型，会用CPU；不启动TCP环境/TF Server，不写checkpoint。读取显式输入目录的registration.json、first-update-coverage.json、step-0.index/data和真实replay.npz；不含这些新输入资产。`CANOPY_FIRST_FRAMEWORK`须指向具有完整`code/orca`/依赖的scalar框架。

```sh
CANOPY_FIRST_SOURCE=/path/to/inputs CANOPY_FIRST_CHECKPOINT_DIR=/path/to/inputs CANOPY_FIRST_FRAMEWORK=/path/to/scalar-framework \
python3 tools/scalar_learning_check/runtime/first_updates_replay.py --arm R0 --seed 3 --output /new/output/R0.json
python3 tools/scalar_learning_check/analyze_first_updates.py --out diagnostics/first-updates
```

第二条复算无需TF/服务器，读取已公布逐次JSON，重建16组CSV和分析（原值完全匹配）。原脚本每次诊断会消耗随机平滑噪声；实际版本改用无平滑TD残差作诊断，不作为真实价值标签。低学习率只feed已有Adam的lr，slot/global_step路径不变。R3/R5允许BN变化，冻结的是actor trainables而非完整forward。R1核对仅critic/target发生重抽。

实际v1/v2源在证据`executed-sources`；v2只增加R6的选择/参数，不重写前轮结果。保留了R0密集/稀疏诊断的非逐位等价；不能称精确重建在线轨迹。历史summary字段`updates_to_abs_mean_pi_point9`实际测的是均值偏移，绝对policy均值阈值应读新CSV对应列。

没有在本轮重新在线训练，下面是前一轮L1的历史工具说明。

---

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
