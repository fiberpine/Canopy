# 2026-11-11 基座筛查与重连准入

[报告](summary.md) · [逐格数据](checkpoint-screen.csv) · [原冻结登记](registration-frozen.json) · [结构化结果](result.json) · [来源与发布变换](publication-provenance.json)

**无健康候选。24 次冻结执行成功；两个真实重连准入均失败，不能把整轮称为零失败。** 72 次候选确认、完整正式重训和双动作训练均未启动。

`screen-latest.json` 和 `positive-latest.json` 是新 18+6 次结果；终点复用 11 月 10 日的公开证据，明确标注历史来源。六格刻意包含弱条件，不能冒充完整 24 格均值。复算使用 `python evidence/baseline-screen-20261111/verify_results.py`。

`capacity-attempt1-latest.json` 是关闭计时文件导致失败的首次尝试；`capacity-latest.json` 是修复计时后因 bind 冲突失败的第二次。二者未打开覆盖热身门，日志记录的 critic 更新为零；入库最大记录只是下界，最终账本缺失，满回放保存均未到达。最新独立端口及权限清理修复为事后版本，只通过离线测试。

原始执行 SHA 与公开文件 SHA 分列。主机、地址、个人路径仅在字符串中替换；数值、布尔和空值保真。旧时钟审计未保留延迟列表时产生的 12 个未使用 NaN 分位数占位转为 null，位置见发布清单，不能把它们当作新评测的分位数。原始私有证据保持原字节。

`runtime-source` 只记录本轮冻结采集辅助，依赖原框架与已登记现场，不是新的通用部署入口。checkpoint、replay、raw15 和逐包日志留在云端；路径是来源记录，不表示 GitHub 已包含这些大文件。

本轮全部网络实验已结束，五机归属进程及记录 IPC 均为零。后续实验需新预算。正常训练入口与仍待真实验证的边界见 [重连候选说明](../../docs/BASELINE_RECOVERY_20261111.md)。
