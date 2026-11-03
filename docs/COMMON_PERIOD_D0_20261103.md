# 最新：Claude审阅复核与零TCP小核查（2026-11-03）

**已完成0新TCP、0 Canopy优化的有限核查；A′和D-Q未启动，原FAIL_A不变。** 同意先小查、后单变量；不把梯度相关性直接当成队列因果或容量不可观测的证明。

- [完整复核、探针与下一候选](../evidence/capture48-zeroTCP-20261103/report/review-and-zeroTCP.md) · [结果](../evidence/capture48-zeroTCP-20261103/results.json) · [容量探针](../evidence/capture48-zeroTCP-20261103/capacity-probes.csv) · [高排队状态梯度](../evidence/capture48-zeroTCP-20261103/gradient-comparison.json)

63,156状态的监督容量探针按整连接分组，独立重复间能读出部分信息，但留一条件泛化很差，不能证明无信息或已稳健可估。全711,285训练决策中，A20高SRTT/minRTT≥2有39,953/331,147条（约12%）；最终回放仍有7,912/71,635条，不能归结为训练没见过此粗分层。同一A20高队列状态、固定窗口而切换critic周期输入时，Q1梯度会改变，仍是离策略只读查询。

复现48run均值Q校准斜率0.12271、相关0.31403；但“四扫描80ms都最好”不成立：原时间加权效用最高均值为40/40/80/80ms。原16个超差格的降级保留。容量归一化奖励可作为优先单变量候选，但原约定明写“真实容量只用于离线审计”，训练期改用它仍需修订协议/奖励版本，并区分固定条件容量与瞬时容量的目标；不自动启动奖励重训。

云端11.33分钟/0.2073机时（<1机时），模型/回放/51保护文件/sysctl不变，五机本轮进程0，完整状态经VPC留云端，本机只下载578,694字节小证据。服务器可关闭；没有追加任务。

---

# D0 与48次冻结补采（2026-11-03）

**最新：48/48次冻结采证完成，0网络失败、0优化；原FAIL_A不改判。** 用户将38次提案修订为24格×2次共48次，四个条件补齐五周期扫描。采得63,156个完整149维状态/动作/后继区间。按采前登记容差，24格中8格两次全过，16格的旧A解释降级；不能把新连接变化直接归因插桩或当作旧A失败复现。

- [完整补采报告和逐格表](../evidence/common-period-capture48-20261103/report/report.md) · [结果](../evidence/common-period-capture48-20261103/results.json) · [周期图](../evidence/common-period-capture48-20261103/report/period-scans.png)
- [两次重复](../evidence/common-period-capture48-20261103/run-summary.csv) · [扰动逐项核对](../evidence/common-period-capture48-20261103/instrumentation-checks.csv) · [有限回报](../evidence/common-period-capture48-20261103/finite-return-labels.csv) · [公开谱系](../evidence/common-period-capture48-20261103/publication-provenance.json)

新轨迹无float32精确±1饱和，但最高边界比例97.57%；部署状态上21/48个固定16点cohort出现共享fc1/fc2跨周期负梯度余弦，与训练状态样本不同。这是冻结critic下的离策略局部诊断，不证明真实环境梯度或五头方案有效。384个H≥20秒同连接后缀按run聚合，每格仍只有2个独立连接；Q既有高估也有低估，不用重叠后缀制造窄置信区间。

预算含小包交付21.69分钟、保守1.808机时，均低于45分钟/4机时；五机进程/IPC清零，51保护模型/选定CP/native/系统参数未变，实例未自动关机。原始NPZ/大日志/模型仍云端。D-Q、λ扫描、选择器TCP、新训练和B0/C均未执行；价值优化仍另行审批。

---

## 历史：原D0阶段

**PARTIAL：冻结部署字段不足。原FAIL_A不改判，B0/C未运行。** 本轮0新拥塞控制TCP、0优化、0λ扫描，模型和BN/target/原checkpoint不变。

- [完整报告](../evidence/common-period-D0-20261103/report/report.md) · [机器可读结果](../evidence/common-period-D0-20261103/results.json)
- [264个run的字段覆盖](../evidence/common-period-D0-20261103/frozen-field-coverage.csv) · [累计actor×周期账本](../evidence/common-period-D0-20261103/actor-period-ledger.csv)
- [分配与CPU图](../evidence/common-period-D0-20261103/report/figures/ledger-and-cpu.png) · [固定训练状态上的梯度方向图](../evidence/common-period-D0-20261103/report/figures/gradient-directions.png)
- [仅待审批的下一提案](../evidence/common-period-D0-20261103/next-proposal.json) · [交付与时长](../evidence/common-period-D0-20261103/delivery.json) · [执行/公开字节谱系](../evidence/common-period-D0-20261103/publication-provenance.json)

确认全程711,285条入库的周期/物理暴露分布；最终回放采样期望20/160ms比例约7.21，但未记录全程按周期采样或梯度贡献。8个固定batch的同状态查询，shared-fc1/fc2跨周期梯度cosine均为正，不能据period_w因正负t编码出现反向就认定需要五头。实际actor与诊断moving前向一致，改用batch BN的差异仅为敏感性证据。

120个A冻结run均缺原ACK区间RTT字段，264个run均没有完整149维状态，因此无法精确重建其区间奖励或H≥20秒同策略价值标签。216训练episode奖励复算可过，但不能替代冻结失败轨迹。48训练启动状态中24个可对保留replay逐位核对，不能冒充正式部署状态支持。

模型段进程CPU：actor约0.08825ms/调用，五候选actor+双Q+N的两次Session取值结构约0.23804ms，约2.70倍。只有微基准与固定调用率投影；不推断闭环CPU收益或一般不可能性。整个D0首次连接至最终小交付约17.56分钟，任务占用约0.325机时；原模型/51保护文件/系统参数未变，归属进程与相关IPC检查通过。用户实例未自动关闭，机时口径不是厂商账单。

建议另批38次60秒冻结采证以补足原始边界和状态；这是GPT/Codex依据本轮数据提出的草案，未与Claude另行核定或由用户批准，**本轮未执行**。D-Q、动作干预、A′和B0/C仍分别审批。完整旧A失败证据保留在[前次交接](COMMON_PERIOD_20261102.md)。
