# Scalar Orca and Canopy training harness

**2026-10-26 status: research only, performance acceptance NOT PASSED.**
The archived `baseline` and healthy V31 models are separate identities. The
latest scalar control/moving runs each completed3600s/actor and72 frozen runs;
only19/36 and16/36 conditions respectively reached80% utilization. The Canopy
artifact arm was paused after about2192s/actor and has no formal frozen results.
Do not overwrite original models or advertise these new runs as strong baselines.

This independent harness trains **70-state, one-window-action** controllers.
Both use a fixed 20ms observation/control period and CUBIC per-ACK inner control.
There is no MTP action, 149-state migration, runtime teacher or fallback.

`policy_family=orca` uses the official Orca `agent.py` and `envwrapper.py`.
`policy_family=canopy` uses the Canopy Sonnet/IBP actor, with shallow properties
12, five partitions, k=3 and reward `0.75 * raw + 0.25 * QC`.

Preserved `frameworks/baseline` and `frameworks/mtpjoint` are not modified.
For an independent installation, copy their shared `code` and `configs/traces`
assets into this framework (or create read-only links to them). Use unique
output directories and run IDs. The controller is `cluster_control.execute(cfg,
duration=seconds)`; it requires the existing five-node authorization/SSH
environment. `timed_distributed.py` remains the learner/actor entrypoint.

Engineering differences from unmodified upstream artifacts are explicit:

- Independent twin-critic Adam clocks on both algorithms.
- Target BN uses initialized and softly updated moving statistics on both.
- Canopy actor interval bounds use the deployed moving-statistics function.
- Training copies each parameter tensor once per query and shares these values
  between the concrete action and interval bounds. This prevents concurrent
  optimizer/BN writes from producing reversed intervals. It is not an atomic
  snapshot of a complete optimizer step. Frozen deployment runs no QC queries.
- QC loss-center modes are explicit through `canopy_qc_modes.py`. `artifact`
  retains the vendored updater's actual loss centers; `initializer` uses good
  loss=0 and bad loss=[0.25,1] as a separate interpretation/ablation. The latest
  Canopy arm used **artifact**, not initializer. Both retain the disclosed
  snapshot/moving-BN engineering. Do not call either exact paper reproduction.
- The bounded development experiment uses a common 2BDP constant-link grid.
  This differs from the current Canopy shallow launcher, which uses 0.5BDP.

These are source-based engineering baselines, not a claim of exact paper
reproduction. The two algorithms have distinct TF implementations and random
initial weights; their comparison is not a reward-only causal ablation.
Runtime frozen evaluation executes the scalar actor alone, without QC/fallback.

An additional **disabled-by-default training ablation** is available through
`orca_actor_loss_moving_stats: true`. It computes the Orca actor gradient through
the same moving-BN function used in deployment, without adding checkpoint
variables or changing the deployed action, critic target, reward, replay, or
window mapping. Normal BN statistics updates retain the existing schedule.
This option addresses a measured training/deployment function difference;
neither performance improvement nor the cause of an existing weak checkpoint
has been established. Do not relabel old checkpoints as trained with this flag.

## Explicit native startup contract

`native/src/orca-server-normal-start.cc` removes the pre-policy helper which
repeatedly wrote1.1*cwnd while waiting for its slow-start condition. Historical
first policy windows could reach roughly2^31. The new native publishes its first
valid real TCP feedback without auxiliary writes; the evaluator requires its
`NORMAL_CUBIC_FIRST_FEEDBACK` receipt. This is an explicitly different startup
contract and is not proof that subsequent policy actions cannot create huge
windows. Some training summaries still contain extreme cwnd values; their
origin remains unresolved. No silent window cap, multiplier scaling or fallback
has been added. The preserved baseline native and model files are untouched.

## Prepare an independent research run

Use the compatible Linux/runtime/kernel setup in the root README. All nodes must
have the repository/runtime and the same absolute isolated framework path.
`prepare` creates source/configs/symlinks only. It does not start training, actors
or TCP. The preparation CLI is a new packaging adapter: its generated deployment
requires a smoke check and was not itself used for the historical scores.

```sh
python3 tools/scalar_research.py plan --variant orca-control --directory /home/ubuntu/scalar-control
runtime/venv/bin/python tools/scalar_research.py prepare --variant orca-control --directory /home/ubuntu/scalar-control
runtime/venv/bin/python tools/scalar_research.py build --framework /home/ubuntu/scalar-control
# Repeat prepare/build on selected actor hosts at the same absolute path.
# Copy configs/cluster.example.json to your own ignored config and fill endpoints/key paths.
# This next command DOES start a finite real-TCP smoke, only when explicitly requested:
runtime/venv/bin/python tools/scalar_research.py train --framework /home/ubuntu/scalar-control --cluster configs/my-cluster.json --actors 1 --seconds 60 --smoke
```

Other choices are `orca-moving` and `canopy-artifact`. Each must use its own new
directory. Formal runs use an explicit `--seconds` (maximum3600), have no
automatic continuation, and do not automatically run evaluation or more arms.
Same-seed paired initialization must be verified; different online trajectories
under a fixed exposure budget will produce different transition/update counts.

```sh
# Explicit frozen evaluation of your own70-state/1-action checkpoint:
runtime/venv/bin/python tools/scalar_research.py evaluate --framework /home/ubuntu/scalar-control --checkpoint /home/ubuntu/scalar-control/checkpoints/EXACT_RUN/step-EXACT_STEP --scenario 0 --seconds 60
# Read-only model/gradient queries, no TCP or optimizer updates:
runtime/venv/bin/python tools/scalar_audit/checkpoint_direction_audit.py --framework /home/ubuntu/scalar-control --run EXACT_RUN --output /home/ubuntu/diagnostics/EXACT_NEW_RESULT.json
```

The read-only diagnostic uses each run's retained final replay. Older checkpoint
queries are distribution-shift sensitivity, not historical on-policy states.
The768-row query gradients are not a particular512-row training update.
Decision-weighted reward summaries are not physical-time reward integrals.
CPU metrics include process ownership defined in `resource_cost.py`; raw calls
and lower total throughput cannot establish a controller CPU advantage.


## 2026-10-27接管诊断补充

同一moving模型的E2冻结干预支持接管条件影响弱场景，但固定10秒预启动未通过正常条件守门，不能作为修复默认值。参见[当前交接](../../docs/RESEARCH_REVIEW_BRIEF.md)与[E2工具](../../tools/e2_handover/README.md)。

## 2026-10-27双向IPC与初始化诊断（opt-in）

`verified_ipc_v2: true`要求独立native和SHA/schema sidecar。两个方向均用整帧flock、头及载荷CRC32、递增64位序号，动作绑定反馈序号/CRC/state ID；错误立即拒绝，不退回旧协议。默认旧ASCII路径保留，没有修改原native、模型、奖励、乘窗或观测周期。这是完整性修复，未证明CPU改善。

```sh
# Linux / 原自定义TCP内核环境；新路径不得覆盖现有二进制。
python3 tools/build_scalar_verified_ipc.py --framework frameworks/canopy_single --output /absolute/new-directory/orca-server-verified-ipc
```

在自有运行JSON显式指定`server_binary`为该绝对路径，并启用`verified_ipc_v2`和`require_normal_cubic_start`。此构建包装只做本机CLI/静态检查；云端已实际编译并运行的是其等价编译命令，见证据中的deployment。部署仍需原有完整runtime/trace/内核。

`scalar_window_initialization: small_uniform`仅允许fresh Orca，把actor输出层kernel/bias设为U[-.003,.003]，随后原`init_target`复制target；不允许改已恢复checkpoint。默认`upstream`保持原初始化。`diagnostic_no_optimizer: true`关闭全部learner优化，实际training actor/RPC/探索/FIFO仍运行。该诊断开关不可用于声称完成训练。

实际完成2次smoke和32次60秒零优化TCP检查，32次全部96个TF变量前后不变；双向159166个正式接收帧一致、CRC/序号失败0。原初始化16次中6次百万窗口、8次超过预注册持续时间门槛；small_uniform仍有2次和7次失败。因此小幅初始化不足，长训NO-GO。持续时间来自边界actual_dt加权近似；两种初始化种子不是独立训练seed。两条small_uniform残余失败的直接乘窗命令与原生区间增长已能解释爆窗，不能归因为critic更新（本轮0更新）。

本轮没有运行E2b、窗口cap或后续训练。IPC日志与锁会增加成本，不能用于低开销创新宣称。原始模型/replay/逐包日志保留云端；精简机器结果见`evidence/scalar-ipc-init-20261027`。新native的verified动作等待依赖外部有限运行监督，本轮不将其当成经过故障注入验证的独立部署watchdog。
