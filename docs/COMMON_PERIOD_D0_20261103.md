# D0冻结诊断（2026-11-03）

**PARTIAL：冻结部署字段不足。原FAIL_A不改判，B0/C未运行。** 本轮0新拥塞控制TCP、0优化、0λ扫描，模型和BN/target/原checkpoint不变。

- [完整报告](../evidence/common-period-D0-20261103/report/report.md) · [机器可读结果](../evidence/common-period-D0-20261103/results.json)
- [264个run的字段覆盖](../evidence/common-period-D0-20261103/frozen-field-coverage.csv) · [累计actor×周期账本](../evidence/common-period-D0-20261103/actor-period-ledger.csv)
- [分配与CPU图](../evidence/common-period-D0-20261103/report/figures/ledger-and-cpu.png) · [固定训练状态上的梯度方向图](../evidence/common-period-D0-20261103/report/figures/gradient-directions.png)
- [仅待审批的下一提案](../evidence/common-period-D0-20261103/next-proposal.json) · [交付与时长](../evidence/common-period-D0-20261103/delivery.json) · [执行/公开字节谱系](../evidence/common-period-D0-20261103/publication-provenance.json)

确认全程711,285条入库的周期/物理暴露分布；最终回放采样期望20/160ms比例约7.21，但未记录全程按周期采样或梯度贡献。8个固定batch的同状态查询，shared-fc1/fc2跨周期梯度cosine均为正，不能据period_w因正负t编码出现反向就认定需要五头。实际actor与诊断moving前向一致，改用batch BN的差异仅为敏感性证据。

120个A冻结run均缺原ACK区间RTT字段，264个run均没有完整149维状态，因此无法精确重建其区间奖励或H≥20秒同策略价值标签。216训练episode奖励复算可过，但不能替代冻结失败轨迹。48训练启动状态中24个可对保留replay逐位核对，不能冒充正式部署状态支持。

模型段进程CPU：actor约0.08825ms/调用，五候选actor+双Q+N的两次Session取值结构约0.23804ms，约2.70倍。只有微基准与固定调用率投影；不推断闭环CPU收益或一般不可能性。整个D0首次连接至最终小交付约17.56分钟，任务占用约0.325机时；原模型/51保护文件/系统参数未变，归属进程与相关IPC检查通过。用户实例未自动关闭，机时口径不是厂商账单。

建议另批38次60秒冻结采证以补足原始边界和状态；这是GPT/Codex依据本轮数据提出的草案，未与Claude另行核定或由用户批准，**本轮未执行**。D-Q、动作干预、A′和B0/C仍分别审批。完整旧A失败证据保留在[前次交接](COMMON_PERIOD_20261102.md)。
