# 连续双动作离线分析证据

对应[问题与方案](../../docs/JOINT_CONTINUOUS_REPAIR_PLAN_20261116.md)。这些是已结束第三短的只读复核，不是新训练、TCP实验或已实现的连续修复。

- `period-value-decomposition.json`：本机最终完整CP在既有960分层SMDP转移和175真实冻结输入的只读结果；模型变量前后逐项不变。确定性target不额外采样目标窗口平滑。工具为`tools/audit_joint_period_value.py`。
- `continuous-parameterization-math.json`：原softsign+log映射与候选物理周期sigmoid的解析局部导数，不能解释为实际训练速度或TCP收益。
- `claude-review.json`：三轮实际讨论、读取范围和已撤回推断。用户明确要求连续时间头，离散方案不再推荐。

复算需要原TF1环境及本机已备份的选定CP/小probe；本目录不重复上传完整CP、replay或原始日志。输入SHA在分解JSON中。没有为本次离线分析连接云端或再下载数据。

审计辅助程序最初草稿曾把3个物理重连标记直接当作Bellman终点，代码核对时发现并在形成结论前改为实际`model.feed`的bootstrap语义；原草稿输出仅在本机标为`scratch-wrong-physical-terminal-mask.json`保留。正式JSON中`physical_end_rows=3`、`effective_bellman_terminal_rows=0`。该修正只影响新审计工具，未修改训练代码或历史实验结果。
