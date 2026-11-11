# 目标BN完整重训后的复核交接

最新：[与Claude的实测后复核及保留边界](../evidence/baseline-target-retrain-20261111/claude-postrun-review.md)已完成。双方同意更新图与学习信号优先于重连；“后半训练无信息”、相关性可证明critic正确及新的饱和早停均未采纳为既定事实或当前门槛。未新增云端实验。

[完整结果与24格表](../evidence/baseline-target-retrain-20261111/summary.md) · [机器可读结论](../evidence/baseline-target-retrain-20261111/result.json) · [首批目标BN证据和失败](../evidence/baseline-target-audit-20261111/summary.md)

目标点策略依赖batch的问题已用同权重/同focal输入隔离，修复后上下文影响为0；但BN-only最终健康失败。不能将“找到真实缺陷”与“已充分解释训练失败”画等号。只保留一个seed且没有新健康终点，双动作暂不接入。

下一次长训前先完成三个可短测的问题：

1. **执行层**：3秒等待通过短准入却在长连接结束失败；报告实际看到sudo fd3仍持有listener。核查native socket的close-on-exec、子进程继承和PID归属，独立短测启动/停止/并发，修复后不靠长训验清理。
2. **学习图**：现有证据已列出critic参数集合重叠、target2重复赋值和共享Adam。逐图核对实际非None梯度、每个时钟推进和唯一target配对；集合重叠不自动意味着重复优化，重复无依赖赋值不自动等于2tau。先建立并测试精确更新契约，再做单变量有限学习。
3. **状态/目标**：用已有最终replay和冻结轨迹对照Q、原效用、边界动作及新连接状态覆盖。单连接与max_bw归一化的机制仍待区分；重连会同时改变覆盖和峰值分母，不称纯覆盖干预，也不同时悄改reward或网络。

如果代码契约修复仍不能在短有限学习中保留动作能力，再评估训练配方或奖励。短准入应包括学习行为观察而不仅是队列不溢出，但不能用早期坏点断言永久不可恢复；最终健康仍需事前固定样本预算、唯一终点和完整原门槛。

健康终点通过后做第二独立训练seed，再开展cwnd/MTP联合学习。双动作的固定周期强对照、同物理经验/更新额度、尾延迟与实际CPU必须先登记，不能把降低策略调用数直接等同CPU收益。

此前与Claude的协议讨论原文保留在[复核目录](../evidence/baseline-review-20261111/README.md)。本文件是本轮实测后的新交接，不冒称Claude已经审阅了新结果。复核时先看result、health-assessment、training-complete和原/补测两套run清单，避免只看平均吞吐或误将FAILED集群判为COMPLETED训练。
