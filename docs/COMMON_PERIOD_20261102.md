# 共周期条件窗口执行（2026-11-02）

本轮按双方确认方案执行 A0→24actor 短准入→A。A 实际 seed3 从 V31 step157704 初始化，711,285 条训练经验、143,919 完整 critic/N/target 更新、64,839 actor 更新；264 次固定周期冻结评测全部完成，运行错误 0。研究阶段结论：**FAIL_A**。完整数值与后续位置以报告为准，不把工程运行通过称为双动作创新成功。

- [报告、失败门槛与限制](../evidence/common-period-20261102/cluster/report/summary.md)
- [全部门槛](../evidence/common-period-20261102/cluster/A-gates.json) · [逐条件 full/startup/trim10 CSV](../evidence/common-period-20261102/cluster/A-condition-metrics.csv)
- [A0 最终核验](../evidence/common-period-20261102/a0/report/summary.md) · [保留的首次 PARTIAL](../evidence/common-period-20261102/a0-first-attempt/report/summary.md)
- [原约定方案](../evidence/common-period-20261102/agreed-plan.md) · [实际运行字节与公开脱敏字节的谱系](../evidence/common-period-20261102/publication-provenance.json)

新实现是 `frameworks/mtpjoint/common_period*.py`；本机契约 `python -m unittest discover -s tools -p "test_common_period*.py" -q` 为24项。真实TF/网络/原生时钟检查另在证据中。旧 `run.py train` 仍是旧路线，不可当本方案启动器；训练需要显式角色、条件、保护资产、A0与容量回执及有限阶段配置。公开 runtime-source 是运行快照；字符串已脱敏，原始/公开 SHA 区分，不冒充原始云端运行字节。

新模型、replay、原始 packet/state/control 日志未下载或上传Git。Git 中的 baseline/V31/51保护模型与原始tag保持不变。最终只保存可恢复权重/优化器/回放，不支持TCP/history/探索现场无缝恢复。B0/C 是否执行以最终报告及门槛为准，不能因已有B0算子就说已完成周期价值选择或多seed盲测。
