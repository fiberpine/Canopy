# Claude 离线接手结果：连续双动作 v4（2026-11-17）

接手 `59cf69a`（`innovation/joint-continuous-conditioned-v4`）。本轮没有 SSH、云端 TCP 或训练；时间头保持连续；原健康基座 main/`c67717d` 与负面证据都未改动；没有训练独立固定周期组。**这仍是离线候选，不是训练就绪，也没有任何 TCP 收益结论。**

## 1 执行环境与范围

- Claude 在 Anthropic 云端的 Linux 沙盒里工作，**访问不到用户本机 `D:` 盘**，看不到本机 stderr 和 `D:/Canopy/experiments` 下的数据。
- 沙盒只有 Python 3.13、NumPy 2.5、SciPy；**没有 TensorFlow，也装不了**（包安装被网络策略拒绝）。
- 因此：
  - **所有 TF 图相关的改动都没有执行过**，只做了 `py_compile` 语法检查和逐行复查；
  - critic 能力问题用 NumPy 独立重建 critic（直接从 TF checkpoint 读迁移权重）复现和诊断；
  - 回放、门槛、覆盖逻辑是纯 Python，用仓库真实代码执行。

## 2 结论

1. **发现一个会卡住真实训练的确定性缺陷，已修复。** v4 校准期探测改成了对数周期上的均匀分层连续值，但覆盖门仍按"离 20/40/80/120/160 最近"分桶，160 ms 桶只得到约 6.9% 的探测。用真实 `replay.py` 和正式配置（24 actor、30 万回放）模拟：回放填满后，160 ms 桶每 actor 稳定在 **57–69 行，永远到不了 128 行**，所以校准在期望上永远不放行。修复后（覆盖桶改为探测本身的 5 个等宽对数分层）每桶 200–205 行，在每 actor 约 240 秒物理时间时达到 128 行。门槛未改。
2. **流水线 fixture 停在 phase0 有两个确定原因，已修复（TF 部分未执行）。** 每个 actor 只用一个周期，不可能满足"每 actor × 每桶"的覆盖门；每 actor 100 行时，5 个 actor 中 3 个的 holdout 块落在第 100 行之后。这两条都是 fixture 自身的问题，不是 learner 或网络失败。
3. **critic 能力任务的失败主因是 Adam 抖动，不是表达能力。** 周期残差输出层约 69 个权重，Adam 每步让每个权重移动约一个学习率，lr=1e-3 时优势估计的抖动约 √69×1e-3 ≈ 0.008，远大于 0.0025 的门。实测后半段误差在 0.002–0.009 之间来回摆，1 万步终点基本是在抽签。
4. **最小算法修复：给新周期残差单独一个 Adam，lr=1e-4；迁移主干保持 1e-3。** 不设该键时图与原来逐位相同。NumPy 复现中：
   - 去掉任务里任意的 43 单位水平偏移后：6/6 过门，后半段 100% 稳定（原设置 5/6，62%–95%）；
   - **原登记任务仍是 FAIL（4/6）**：两个负优势格为 0.00260/0.00266，门为 0.00249，单调下降但 1 万步内没收完；2 万步的诊断运行为 0.00179。门槛和预算我都没改。
5. **不采用的方案（有证据）：**
   - Φ+A：去掉水平偏移的实验等价于完全准确的 Φ，lr=1e-3 下仍有 24%–38% 的记录点超门，抖动不受 Φ 影响；
   - 残差改成"相对同窗口 20 ms 的差"：比原结构更差（0.00306）；
   - 用 τ=0.001 的参数滑动平均读优势：0.0030/0.0034，不能消除摆动。
6. **原任务的 43 单位偏移不典型。** 迁移 critic 在 1,059 个真实输入上的中位值是 163.5，而任务挑的两个状态（minRTT 最小、最大）上分别是 148.7 和 99.4，标签 168/169 是任意给的。原任务作为压力测试保留并仍是判定依据；"迁移初值水平"版本作为诊断另报。

## 3 改动清单（相对 `59cf69a`）

| 文件 | 改动 | 是否执行过 |
|---|---|---|
| `frameworks/joint_healthy/core.py` | 新增 `period_bucket(cfg, ms)`：旧档案仍按最近锚点分桶，v4 按 5 个等宽对数分层 | 是（单测、模拟） |
| `frameworks/joint_healthy/replay.py` | 两处分桶改用 `period_bucket` | 是 |
| `frameworks/joint_healthy/model.py` | 设置 `critic_period_residual_lr` 时，每个 critic 拆成主干 Adam（lr_c）和残差 Adam；梯度一次算出并经 `tf.tuple` 同步，两组都用更新前的同一 critic；恢复时检查该键一致 | **否（仅语法）** |
| `frameworks/joint_healthy/continuous.py` | 校验该键：仅 v4 可设，范围 (0, lr_c] | 是 |
| `frameworks/joint_healthy/configuration.py`、`templates/joint-healthy/joint-continuous-v4.json` | v4 档案设 `critic_period_residual_lr=1e-4`（模板保持 CRLF，只加一行） | 是（配置层） |
| `frameworks/joint_healthy/identity.py` | 准入契约绑定该键 | 是 |
| `tools/verify_joint_healthy_pipeline.py` | 每 actor 轮转 5 个分层周期、300 行、按旋转公式选相位使 5 个短回报块各落一桶并在启动前断言；FIFO 容量按总行数；失败时保留日志目录与门槛原因 | 排程部分是，TF 部分**否** |
| `tools/check_joint_continuous_capability.py` | 轨迹改用与门槛同口径的 3 列 MAE（原 4 列含恒为 0 的 20 ms 列，例如原 5000 步的 0.00225 按门槛口径约 0.0030）；后半段每 250 步记录；另报"迁移初值水平"诊断任务；判定仍只看原任务 1 万步终点 | **否** |
| `tools/verify_joint_continuous.py` | 断言残差变量只由残差 Adam 持有槽位、主干只由主干 Adam 持有 | **否** |
| `tools/simulate_joint_continuous_coverage.py` | 新增：用真实回放代码模拟正式配置的校准覆盖，`--bucketing nearest` 复现修复前 | 是 |
| `tools/joint_numpy_reproduction/` | 新增：无 TF 的 checkpoint 读取器、v4 critic 的 NumPy 重建（手写反向、TF 版 Adam）、能力矩阵单格脚本 | 是 |
| `tests/joint_healthy/test_continuous_coverage.py`、`test_numpy_reproduction.py` | 新增 9 项 | 是 |

## 4 实际执行与结果

- `python3 -m unittest discover -s tests/joint_healthy`：**78 项通过**（原 69 + 新 9）。
- `python3 -m unittest discover -s tools -p 'test_*.py'`：**200 项通过**（含 scalar 相关测试）。
- `tools/verify_repository.py`：`59cf69a` 原样为 PASS（5,165 文件）；本次按改动更新清单后重新校验，结果见提交。
- 覆盖模拟：`evidence/joint-continuous-claude-20261117/coverage-nearest-prefix.json`（修复前，`COVERAGE_UNREACHABLE`）、`coverage-strata-fixed.json`（修复后，`COVERAGE_REACHABLE`）。行为分布中"约 2% 空 RTT 帧"和 minRTT 四档是假设，160 ms 桶在修复前只由探测供给，不受这个假设影响。
- critic 矩阵：`capability-matrix-numpy.json`（24 格，两种残差学习率 × 2 critic × 3 种优势 × 2 种水平）。
- 诊断过程：`critic-development-diagnostics-numpy.json`，含不同预算和被否决的变体，**不是预登记结果**。
- NumPy 复现本身的校验：checkpoint 读取器前向与 530 个由基座窗口产生的探针动作误差 < 1e-4；手写梯度与有限差分相对误差 ≤ 4e-4；残差层初始化用 NumPy 种子，与 TF 种子 3 不逐位相同，结果是统计意义上的复现。TF 工具在 `59cf69a` 的结果（正 0.00268/0.00269，负 0.00197/0.00267）落在 NumPy 同设置的范围内（0.00211–0.00582）。

## 5 未执行、需在本机 TF1.14 上跑的项目

按顺序，全部离线：

```sh
PY=D:/Canopy/experiments/joint-healthy-offline-20261113/python36/python.exe
CFG=templates/joint-healthy/joint-continuous-v4.json
STATES=evidence/joint-cost-repair-20261116/real-frozen-probes.npz
$PY -m unittest discover -s tests/joint_healthy -p "test_*.py"
$PY -m unittest discover -s tools -p "test_*.py"
$PY tools/verify_joint_continuous.py --config $CFG --states $STATES --output graph.json
$PY tools/verify_joint_healthy_pipeline.py --config $CFG --output pipeline.json
$PY tools/check_joint_continuous_capability.py --config $CFG --states $STATES --output capability.json
$PY tools/verify_repository.py
```

预期与判读：
- 图检查：应 PASS，新增 `separate_period_residual_adam=true`。若槽位断言失败，说明 `tf.tuple` 拆分写法在 TF1.14 下不成立，需要修，不是算法问题。
- 流水线：应自然经过 phase 0→1→2，actor 更新 > 0，欠额 0，≥3 个 checkpoint 清单。失败时 `pipeline-failure-scene/` 会保留现场，`last_gate_reasons` 给出原因。
- 能力：按 NumPy 复现，**原登记任务预计仍 FAIL**（负优势两格略超门），`isolated_level_status` 预计 PASS。若 TF 结果与此明显不同（例如原任务 PASS 或诊断任务 FAIL），以 TF 为准，并说明 NumPy 复现与 TF 图有差异，需要查。

## 6 下一轮云端最短门

在第 5 节全部离线项通过（能力原任务的 FAIL 如实带着）之前，不开云。之后：

1. **机制检查（冻结、不学习，约 30 分钟）**
   - 读已有 raw15 或新跑少量冻结流：各 RTT 档稳态的 cwnd 与 ssthresh 关系、实际倍率分布。只记录事实，不裁剪 T；T≤2RTT 只是持续慢启动下的条件近似。
   - 连续周期实际执行：同权重用 `--override-period-ms` 取 3 个非锚点值（如 27、63、133 ms）在 2–3 个条件上，核对 native 实际 dt 与计划周期、每次决策只计一次成本、无隐藏 20 ms 读取。
   - v4 初始策略的 24 格 × 1 冻结评测，作为 G4 证据到 v4 源码与 native 的桥接，补齐准入回执里的 `initial_frozen_grid_health_pass` 和 `short_tcp_mechanism_pass`。
2. **可学习性短准入（600 秒/actor，事先登记）**
   - 硬门：校准**自然放行**（记录放行时的覆盖、TD 与短回报）；两头参数都更新；BN 统计逐位不变；欠额有界并最终为 0。
   - 观察：残差优势在固定探针上的符号与幅度轨迹、时间头 logit 是否移动、窗口相对 teacher 的偏离按周期分层。
   - 结束后同权重冻结 24 格 × 1：健康门不变；调用变化作为可学习性信号报告，不作最终验收。
3. **正式训练（3600 秒/actor）只在第 2 步通过后。** 最终门不变：调用降 ≥20%，逐格吞吐、mean、P95 门全过；不训练独立固定周期组。

## 7 仍然存在的风险

- **价值分辨率：** 即使修复后，原登记的能力任务也没在 1 万步内过门；真实 TD 学习里还有自举目标和小批量噪声，比这个受控任务更难。短准入里如果时间头不动，第一嫌疑仍是 critic 对 0.01 量级差的分辨。
- **条件窗口学习率：** 条件支路沿用窗口 lr=1e-5。解析 actor 任务里 5,000 步能达到目标窗口，但真实短测中长周期窗口的适配可能偏慢。建议在短准入里看按周期分层的窗口偏离，不预先改。
- **`tf.tuple` 拆分优化器未执行。** 这是本轮唯一没跑过的算法路径改动。
- **单 seed：** 基座只有一个训练 seed，所有双动作结论都以"这个基座"为条件。
