# 最新补充：IPC修复通过，初始化仍未过关（2026-10-27）

**仍为PARTIAL、NO-GO长训；尚无新的合格Orca/Canopy基座，也没有双动作性能—CPU突破。** 本节是E2之后完成的单动作固定20ms诊断。原baseline、V14、健康149维V31及51个选定模型文件未改，V31的观测/执行真共周期语义未变。

## 已完成的修复与真实检查

- 新增opt-in `verified_ipc_v2`：双向整帧flock、头/载荷CRC32、64位递增序号，动作绑定state ID/反馈序号/CRC；异常即拒绝，默认旧ASCII保留。没有额外TCP采样或改乘窗、奖励。实际Linux编译、Python/C++ mmap 5000次交换通过；后者不是TCP实验。
- 经实际training actor/RPC/探索/FIFO路径完成2次12s smoke、32次60s正式检查。8条件×初始化种子3/4×原初始化/小幅初始化，均2BDP、Gaussian std=.2，learner优化锁0；不是24actor训练，不是多训练seed。
- 正式159166个接收帧与发送payload/CRC一致，CRC/序号失败0；每次全部96个TF变量（含BN、optimizer和计数）前后不变。IPC正确不代表学习或网络性能合格，也未证明新锁/日志省CPU。

| 初始化 | 正式次数 | 百万窗口门槛失败 | 持续时间门槛失败 |
|---|---:|---:|---:|
| upstream | 16 | 6 | 8 |
| 输出kernel/bias U[-.003,.003] | 16 | 2 | 7 |

持续时间门槛为边界actual_dt加权的cwnd>4×(BDP+queue)比例<1%，不是区间内部精确占用。完整full/trim10利用率、吞吐、mean/P95/P99、丢包、低窗口及动作保留，不能把中性初始化的高吞吐单独当成功。

两个小初始化残余失败均为seed4/R200：G48/G72最大窗口37202245/52372832，超界比例61.68%/58.30%。无探索输出均值约-.0034/-.0031；到首次百万窗口前，探索+clip理想log冲量约8.660/8.303、确定性项-.465/-.393、整数取整项-2.443/-2.185、命令之间native增长5.810/5.207，代数重建误差<4e-15。已校验的连续乘窗命令和native增长足以解释这两个局部案例，不需要假设发生critic更新或撕裂读；不能因此排除其他连接中的PRR恢复效应。

## 反证、限制和下一步

旧264条连接的6714230个cwnd配对中36不一致；percent配对0不一致，9个percent=24两端一致。旧raw拼接重建支持撕裂读风险，不能唯一证明每条异常的源头，更不能把24都归因于动作通道损坏。新协议不将未校验半帧交给策略。

小幅初始化减少爆窗但不足，不能立即长训。下一步只应单变量检验乘性探索的累计冲量，先用零优化短测通过，再决定有限训练；尚未实现或启动。未运行E2b、窗口cap或后续双动作训练。额外配对初始化/producer账本后验收集先有字段错误，修正后受SSH阻断；该额外结果明确BLOCKED，不能冒称完整PASS。各次96变量不变的原证明已保存。

结果收尾确认learner归属进程/IPC为0、51保护文件SHA未变。大模型、replay和逐包留实验存储；下面发布的均为源码和小证据，不需要开启服务器来阅读。

- [全部32次正式检查](../evidence/scalar-ipc-init-20261027/formal-results.json)、[64个full/trim10窗口CSV](../evidence/scalar-ipc-init-20261027/evaluation-runs.csv)、[汇总判定](../evidence/scalar-ipc-init-20261027/final-summary.json)、[旧echo审计](../evidence/scalar-ipc-init-20261027/legacy-echo-summary.json)。
- [残余失败分解](../evidence/scalar-ipc-init-20261027/small-failure-decomposition.json)、[实际注册计划](../evidence/scalar-ipc-init-20261027/registration.json)、[收尾](../evidence/scalar-ipc-init-20261027/closure.json)、[验证状态](../evidence/scalar-ipc-init-20261027/validation.json)。
- [执行脚本与复算入口](../tools/scalar_ipc_diagnostic/README.md)、[发布来源与改写边界](../evidence/scalar-ipc-init-20261027/publication-provenance.json)。

---

# 当前研究审阅交接（2026-10-27）

当前仍为PARTIAL。原健康V31、baseline、V14及51个选定模型未改。以下是新完成的单动作Orca冻结诊断；没有新增训练或已通过性能验收的模型。此前10-26结论保留在后文，日期和运行状态不得混用。

## E2：接管方式有重要影响，固定10秒预启动未通过验收

完整72次正式60秒评测为6条件×A0/A1/A2/A4×3冻结重复，另5次真实pilot，0网络进程失败、0探索/优化。只一个训练seed。18个condition/repeat块跨五机独占并行，四臂在同一块同机顺序运行，块内随机顺序；最长节点约1011秒。没有实施A3、灰区加测或新训练。

A0立即执行冻结moving策略；A1先10秒中性窗口动作，状态、max_bw及历史照常更新，然后交权；A2交权边界清零max_bw并将历史置为零63+当前7；A4使用配置容量参考下限，仅特权诊断。网络和窗口动作定义未变。full含全部启动成本，trim10预先固定，没有为尾部问题换统计窗口。

| 条件 | A0利用率 | A1利用率 | A2利用率 | A4诊断利用率 |
|---|---:|---:|---:|---:|
| G48-R20 /2BDP | 41.23% | 94.58% | 93.27% | 95.01% |
| G36-R20 /2BDP | 63.72% | 94.13% | 92.67% | 89.60% |
| 恒定24-R20 /5BDP | 38.81% | 94.44% | 93.08% | 98.58% |
| G72-R100 /2BDP | 77.92% | 98.23% | 98.90% | 98.72% |
| G12-R50 /2BDP | 98.56% | 97.63% | 97.60% | 100.00% |
| G48-R50 /2BDP | 98.59% | 94.69% | 96.20% | 99.15% |

表为trim10、3次冻结重复均值，非多训练seed。四弱A1-A0平均+39.9228pp，12/12配对为正，通过预注册机制支持门槛。但正常G48-R50利用率退化3.90pp、mean排队3.18→8.74ms，违反util和mean守门；G72-R100 P99为45→178.33ms。三个A1重复中，trim10之后>100ms的尾部均集中在packet-clock第10秒的交权区间。这是必须保留的启动成本，不以新trim窗口掩盖。

**判定：接管条件是重要影响因素；统一固定10秒预启动方案拒绝采用，当前NO-GO长训。** A2在清零后由当前满速反馈立即重建约容量95%—96%的参考，因此A1≈A2不能证明max_bw不重要。A2保留完整内核和连接状态，不能解释为只保留cwnd。A4改变策略归一化输入，不能当成可部署性能结果或容量oracle方案。

## 训练前窗口异常和遥测限制

既存48条训练连接有45条出现>=1e6窗口，43条在第一次完整critic优化之前（control23/24，moving20/24）；另moving两条在首次优化后达到阈值。一个moving例子是在首次优化前17.954秒将558688按337%写为1882778，前面连续多次正乘窗，附近无探索输出也为正。源control.csv记录的是实际RL写窗，不能只归因为旧启动辅助或后期训练。初始化、探索和乘窗累积需在下一轮训练前处理，不能靠加长训练或静默加cap掩盖。尚未证明这些异常是全部性能问题的唯一原因。

199482条正式raw反馈中发现1条packets_out=2790000000、同刻cwnd476。内核采集/非原子scalar IPC来源未确认；该字段本身未进当前7维actor状态，但不能证明其他字段免疫。该run的occupancy均值无效，原始记录完整保留。72次网络进程完成不等于遥测质量完全PASS。旧cnt是RTT采样次数，并非确认包数；不能据此精确判定慢启动或cwnd-limited，处于ssthresh以下也不等于病理锁死。

## 周期价值检验更正

存在两个不同的约0.86 p值：V37 physical4s校准留出检验p=0.859285；V38窗口方向探针p=0.855036。不能混为一谈。V37父V31周期价值与有限运行回报的两轮稀疏关联检验为p=0.0139965和0.0102474，支持正向关联。未显著不等于证明无效；这些有限回报/尾项代理也不是最优真实Q或部署网络质量证明。原始数值见[prior-value-analysis.json](../evidence/e2-handover-20261027/prior-value-analysis.json)。

## 审阅、复算和后续门槛

- [全部144窗口和CPU指标](../evidence/e2-handover-20261027/e2-results.csv)、[48组均值](../evidence/e2-handover-20261027/e2-summary.csv)、[配对差](../evidence/e2-handover-20261027/e2-paired-differences.csv)。CPU是同run25..50s归属进程值，full/trim行重复同一测量；新增诊断日志不代表部署成本，不能拿少调用直接换算CPU。
- [注册计划](../evidence/e2-handover-20261027/registration.json)、[最终判定](../evidence/e2-handover-20261027/e2-decision.json)、[逐机完整小结果](../evidence/e2-handover-20261027/runs)、[训练异常窗口证据](../evidence/e2-handover-20261027/history)、[字段异常](../evidence/e2-handover-20261027/raw-field-sanity.json)。
- [工具与运行边界](../tools/e2_handover/README.md)、[发布来源](../evidence/e2-handover-20261027/publication-provenance.json)。路径参数化和准备包装未新增TF/TCP验收，不能当作原Release已覆盖的新运行认证。
- 下一步先定位scalar遥测完整性，并在零优化短检查中验证初始窗口策略不会指数推窗；研究中性输出初始化时保持架构与乘窗定义，独立记录变量。通过后再决定公平接管协议及有限训练，不自动重启旧pipeline。

五机已核查无旧双线/本轮残留进程和已记录IPC，保护模型与源checkpoint未变。原始raw/逐包/model/replay留实验存储，仅同步小证据。新增研究模型未上传Git。Canopy部分训练仍暂停、无正式冻结；双动作V31未改，尚无稳定网络质量—CPU优势。

---

# 2026-10-26交接（历史状态）

本次同步后续研究源码与小证据。原始快照仍为 `85baf09b6f43a9f3223f7a8eaa1a4321b1a11b3b` / `consolidated-20261018`，51个原模型文件未变。新增代码不代表新增合格模型；实际训练入口、事后归档与发布commit必须区分，参见[来源说明](../evidence/20261026-publication-provenance.json)。本文后半保留原2026-10-20交接，其服务器状态仅属当时。

## 当前结论与身份

**仍为PARTIAL，没有论文级性能—实际CPU突破。** 健康V31保持149状态、两个连续动作、原乘窗和20—160ms真共周期。长MTP没有隐藏20ms RL观测，CUBIC逐ACK执行；无提前唤醒、QC或fallback。单个RL控制耦合动作没有被证明不可行，当前证据也不支持仅换时间头就能解决。

| 对象 | 实际工作 | 结论与限制 |
|---|---|---|
| V31兼容校准 | 完整actor/critic/target/BN迁移，fresh优化器/replay/计数，独立actor gate | 迁移与冻结验证通过；未产生可替换成功策略，不是连续resume |
| 独立critic Adam | 显式opt-in、A/A与中心化短数值对照 | 数值修正可重复；不是网络性能收益证明 |
| 五个fixed-MTP149模型 | 各3600s/actor独立随机长训 | 仍有弱项，不能把短周期失败包装成动态MTP优势；新模型未上传 |
| Orca control / moving | 正常启动，同一完整随机step0，24actor各3600s，各72冻结 | util>=80%条件为19/36、16/36，均未全面验收 |
| Canopy单动作artifact | 70状态/1动作，约2192s/actor后用户暂停 | 1872730入库、186044完整critic；正式冻结0次，性能未知 |
| 区间RTT奖励shadow | 24条旧连接重建149状态，两奖励各2000完整critic，actor/BN冻结 | Q1留出Bellman RMSE .4660/.4702，无决定性优势，actor未开放 |

两组Orca完整critic为255642/354516，优化器操作511284/709032。统一每入库transition .1完整更新，因此实际经验/更新数不同，是行为差异的下游结果，不是额外配置处理。均只一个训练seed；两次冻结不等于多训练seed。原始11组模型仍在Git，新研究checkpoint/replay留原实验存储，来源路径不是已交付模型。

## 优先阅读的新增证据

1. scalar旧native在第一条策略前辅助放大窗口，可能达到约2^31。新分支去掉辅助写窗并审计首反馈，原baseline未覆盖。启动交接因此不同，旧模型比较必须匹配启动。
2. 原Orca公式以连接自身observed_peak归一。真实原函数的合成fixture可让3.33%与83.33%利用率对应相同状态和reward=1。这是数学性质，不是TCP实验。144冻结中有5个观察配对（4 full、1 trim10）网络五指标均不更好却reward更高，非5个独立场景。保留[逐run数据](../evidence/scalar-20261025/evaluation_runs.csv)和[逐条件均值](../evidence/scalar-20261025/evaluation_summary.csv)。
3. 动作=-1、Gaussian std.2时，非减窗概率约2.87e-7，46calls/s下每流小时预期.0475次；说明边界附近支持稀少，不证明永久不可恢复。
4. [16个checkpoint只读方向审计](../evidence/scalar-direction-20261026/learner-outcome.json)表明moving最终actor梯度与部署函数一致，但弱低RTT条件actor接近critic预测的负动作优选。Q网格/梯度不是正确反事实；每臂使用自身最终replay，早期CP查询为分布偏移敏感性，768诊断batch不等于某次512历史更新。
5. 部分整段训练摘要平均cwnd仍达1e8—6e8级。首反馈正常不保证之后正常；具体形成时点与来源尚未定位，不能静默加窗口上限或改动作定义掩盖。
6. scalar对上游的回放容量/采样、warmup、更新节奏、target BN和Adam有工程调整，不能将其失败直接当作官方Orca上限。边界sRTT、ACK样本均RTT与精确效用积分/尾延迟不是同一口径。

## 代码、复现与下一步

- [单动作研究入口](../frameworks/canopy_single/README.md)：artifact/initializer QC模式分离、正常启动native源码、隔离目录。新包装入口须另做环境smoke，本次发布未新增TCP验收。
- [V31兼容初始化与冻结门控](V31_COMPATIBLE_CRITIC_CALIBRATION.md)：新特性显式启用，未替换健康V31模型。
- `tools/scalar_audit/audit.py`：本机NumPy执行原公式与证据检查；`checkpoint_direction_audit.py`在兼容TF1主机只读现有网络/replay。
- `tools/interval_reward/`：从既有真边界导出两奖励，149状态不改，校准结果actor_release=false。ACK均RTT不等于精确区间效用。

CPU沿用resource_cost.py进程归属口径，包含运行时和数据路径，不能当纯NN成本。每run只有一次25..50s CPU测量，CSV的full/trim10行重复该值，不能当两个独立CPU样本。证据中的私有端点和绝对路径改为占位符，不能直接当部署配置。旧SHA回执对应原始文件；公开文件以repository-files.json校验。

下一步先用既有控制日志定位异常大窗，再选择一个可证伪修复；不默认扩网络、扩集群或同时更改奖励/探索/结构。原单动作baseline与V14/V31角色均保留，新的scalar研究不冒充已认证的论文复现。

---

## 原2026-10-20交接（历史记录）

# V31研究审阅交接（2026-10-20）

源码仓库：https://github.com/fiberpine/Canopy 。原始三实现收敛快照commit为`85baf09b6f43a9f3223f7a8eaa1a4321b1a11b3b`，tag `consolidated-20261018`。当前仅公开仓库并补充研究交接，未做新训练/评测，云服务器没有启动。

## 目标、底座与不可混用的身份

目标是在保住逐场景拥塞性能的前提下降低实际系统CPU，而非只减少调用。V31为149维状态、双动作：窗口倍率沿用原定义，MTP范围20—160ms，同时决定观测和执行周期；长MTP期间没有隐藏20ms RL观测，CUBIC继续按ACK控制。保留此架构与窗口动作定义，当前不加QC/fallback/提前唤醒。

主模型为健康V31 step157704；另外保留冻结70维单动作baseline和V14固定20ms观测/动态执行模型。V14不能作为真正减少观测唤醒的实现；V31仍为PARTIAL。Git中有11组模型/必要先验及原配置。五个固定20/40/80/120/160ms模型确为独立训练，但训练覆盖/谱系并非完全一致，不能虚称统一同预算多seed强对照。V31原训练从V28先验迁移、新建优化器/replay/计数；复跑模板不是从最终V31无缝resume。

已有1learner＋24actor训练工程，每台actor服务器6个独立环境，重连轮换环境；不是56actor同时运行。打包另验1actor和跨四机4actor短训练，不能当作新24actor容量/收敛实验。

## 已确认事实

1. 修复并审计过target BN、IPC快照、真实时钟/dt、SMDP折扣、物理入库经验更新额度、无新RTT时调度边界等问题。修bug有效，不代表每次修复都带来性能收益。
2. V28吞吐从ACK delivery-rate样本均值改为相邻真实边界累计bytes_acked差/真实dt。该修复已纳入V31，带schema保护，没有增加中间20ms观测。它不等于应用goodput，也没解决边界sRTT/非线性效用漏尖峰等全部测量问题。
3. V31增加执行动作及RTT新鲜度/年龄上下界、TCP快照反馈。V36审计未发现MTP执行失效，学习周期8/8次动作必要门槛通过；动作有方差不等于选择有益。
4. V34确认并修复双critic共享Adam时钟，但候选未过综合性能/动作门槛；冻结健康V31保留原实现，不静默改成V34后继续声称同一模型。
5. V35从健康V31 actor/BN出发、fresh随机critic/target与独立Adam，两种奖励分母各600秒/actor，候选动作均0/30，父模型30/30；所以只换奖励分母不足以解决退化。价值冷启动/target传播是待检验机制，不能称唯一已证明根因。

## V36匹配诊断数据

四个开发条件×8变体×2重复=64次正式冻结，优化/探索关闭，网络运行失败0。下表为预定trim10结果；单位为利用率% / 平均排队ms。baseline和官方Orca使用正常CUBIC启动参照，不是原baseline历史启动均值。

| 场景 | 自训baseline正常启动 | 官方Orca正常启动 | V31学习MTP |
|---|---:|---:|---:|
| 浅队列公交R20 | 90.40 / 13.05 | 87.81 / 11.52 | 76.86 / 7.28 |
| 公交R100 | 91.91 / 67.97 | 95.93 / 101.18 | 87.04 / 33.14 |
| 突降R100 | 86.93 / 29.19 | 87.21 / 44.14 | 98.83 / 31.49 |
| 混合时间尺度R20 | 98.49 / 8.40 | 92.36 / 3.58 | 98.85 / 9.68 |

同V31权重强制80ms：浅公交76.48%/7.30ms，与学习周期相近，指向窗口/状态/目标仍弱；公交R100强制20ms仅22.49%，缩短周期不是通用修复。突降强制80ms为97.99%/18.69ms、P99 310.50ms，学习为98.83%/31.49ms、P99 404.33ms，说明周期选择仍有改进空间。混合学习9.68ms对强制80的17.13ms较低，但调用18.56/s对12.24/s更高，不能称统一省开销。

这些override改变未来观测、状态、窗口输出及累积乘窗强度，不是独立训练fixed baseline，更不是保持窗口轨迹不变的纯频率反事实。CPU来自冻结单流25—50秒归属进程user+system，部分口径含Mahimahi；不能解释成纯NN CPU或训练集群开销。完整/P95/P99/丢包/CPU见原始CSV，不只看该表。

原始CSV：`evidence/causal/metrics/v36-causal-final2/evaluation_summary.csv`及`evaluation_runs.csv`。全部开发多算法比较：`evidence/multialgorithm/metrics/v34-all12-final2-r3/`，经2026-10-20重解析为10组算法×12条件×2=240次；full/trim10是分析窗口，不是额外重复。另有Aurora预训练参考，其协议/覆盖不完全匹配，尚缺统一公平重训的纯学习强对照。现有重复不是多个独立训练seed，发现/调优场景不等于盲测。

## 第一性原理下的优先判断（建议，不是新实验结论）

**把目标写成“网络质量约束下最小CPU”，并画性能—成本Pareto边界。** 每个场景必须同时报告吞吐、平均/P95/P99排队、丢包和实际CPU；可预先定义第一梯队为相对匹配强对照的非劣容限，不要求不可能的所有指标处处第一，不允许牺牲弱场景后只报平均。

**MTP既决定下一次获得信息的时间，也改变窗口干预强度。** 忽略CUBIC与整数效应时，仅RL乘窗贡献可描述为每次`Δlog(cwnd)≈a_cwnd*log(4)`，单位时间约与`a_cwnd/MTP`有关。这是解释耦合的近似诊断式，不是立即按MTP缩放窗口动作的理由。必须保留动作定义，训练时让窗口政策在周期覆盖中学会合适脉冲。

**先查有效训练链与目标，再增加训练。** 跟随run.py→distributed.setup→动态install/初始化→实际agent/native追踪，不能只读一个未生效agent方法。先检查critic对真实周期收益的排序/尺度、actor开闸时的校准与target传播、旧状态的时间统计、奖励排序。若改奖励尺度，旧Q不能未经校准直接使用；若重建随机critic，也不能直接放开健康actor照搬V35失败。

**把“分布更多”改为“信息更有用”。** 在相同覆盖/物理预算下比较多个固定周期，先寻找同一连接不同阶段出现周期折中变化的场景，并问决策时是否有可观测先兆。对完全不可预知的突变，睡眠中的策略不可能提前得知；不能把预知trace的oracle作为可部署策略。固定周期跨场景排名交叉本身也不足以证明动态在线选择有效。

**独立测CPU瓶颈。** 将ACK侧CUBIC、IPC/读状态、特征、NN推理、等待轮询、诊断日志和仿真器分开。若日志/轮询占主导，先做可验证的工程单变量改动，复核动作轨迹和网络结果再比较CPU。禁止靠改变日志范围/多流负载给新方法有利口径；新增ACK侧统计若需要必须单独计成本，不能称零开销或隐藏20ms观测。

## 最小后续程序

1. 离线先做有效调用链/目标/数据口径审查，列出已证bug、机制假设及反证，不再盲目添加模块。
2. 若需新TCP数据，先短冻结：浅公交、高RTT公交、突降/混合，统一预注册窗口、启动、重复及指标；actor冻结时收集有明确预算的周期覆盖数据。重新建连不能声称还原同一TCP起点。
3. 校准critic并验周期排序，再选择一个变量有限训练（优先评估保留健康actor+兼容critic/target/BN的可控warm-start，或actor冻结下价值校准），开actor后监测双动作与逐场景退化；根据证据决定，而不是同时改奖励、探索、结构与数据集。
4. 局部门槛通过再增加物理训练量/环境覆盖，保持匹配固定周期强对照；最终至少多独立seed及未用于选择的测试trace/相位/队列/RTT，报告区间与所有失败。
5. HyVar已研究动态MTP混合拥塞控制，TaCoS已研究联合控制与持续时间；因此“加一个时间动作”不能单独作为充分创新。需要证明特有的TCP反馈延迟/CUBIC乘窗耦合机制及对应最小设计，并有性能—真实开销收益。本文未完成全面新颖性检索，不能预先保证可发表。

## 审阅时应读的有效实现

先读根README、`docs/MTPJOINT_V31.md`、`EXPLORATION_HISTORY.md`、`manifest.json`及三个train/eval模板。重点读`run.py`、`frameworks/mtpjoint/distributed.py`、`decision_clock_v21.py`、`joint_mtp.py`、`feedback_context_v31.py`、`trainable_timing.py`、`staged_timing.py`、`consistent_target_bn.py`、`moving_stats_backend.py`、`period_coverage.py`、`update_credit.py`、`resource_cost.py`、`code/canopy/rl-module/agent_v2.py`、`envwrapper_v2.py`和`code/runtime/src/orca-server-joint-mtp.cc`。名字中的历史版本号不说明该代码未被使用；需要沿实际安装链核实。

研究依据优先官方原文：https://conferences.sigcomm.org/events/apnet2025/slides/Session3/3-4.pdf （HyVar官方slides，不能假装已读不可访问全文）；https://papers.nips.cc/paper_files/paper/2024/file/746b0a1e6e3cff8d968ba6d2e6fff049-Paper-Conference.pdf （TaCoS）。TaCoS假设获得区间累计奖励，而本实现从边界统计近似效用，必须审查适用边界；并且CUBIC在间隔内继续反馈控制，不等于把整个网络控制输入恒定保持。
