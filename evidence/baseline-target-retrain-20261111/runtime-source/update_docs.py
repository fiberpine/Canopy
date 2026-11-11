"""Refresh current entry points while preserving earlier proposals and failures."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
R=H.parents[1]/'innovations/baseline-target-audit-20261111'
z=json.loads((H/'result.json').read_bytes());o=z['overall']
entry=f'''[目标BN单变量完整重训与最终评测](evidence/baseline-target-retrain-20261111/summary.md)：短准入通过，fresh seed3完成{z['transitions']:,}条/{z['counters']['critic_iterations']:,}完整更新；最终健康 **{z['health']}**（利用率{o['utilization']*100:.2f}%、mean queue {o['queue_mean_ms']:.2f}ms）。原72项70完成、2项首次策略动作前启动失败，2项同权重补测单列。正式训练因清理超时保留FAILED，完整终点已持久化且独立验账；五机最终收尾通过。没有新健康基座、第二训练seed或双动作收益结论。[下一步复核方向](docs/BASELINE_TARGET_RETRAIN_20261111.md)。

'''
p=R/'README.md';text=p.read_text(encoding='utf-8');title,rest=text.split('\n',1)
if '[目标BN单变量完整重训与最终评测]' not in text:p.write_text(title+'\n\n'+entry+rest.lstrip(),encoding='utf-8')
brief=entry.replace('(evidence/','(../evidence/').replace('(docs/BASELINE_TARGET_RETRAIN_20261111.md)','(BASELINE_TARGET_RETRAIN_20261111.md)')
p=R/'docs/RESEARCH_REVIEW_BRIEF.md';old=p.read_text(encoding='utf-8')
if not old.startswith('# 最新：目标BN完整重训'):
 p.write_text('# 最新：目标BN完整重训仍未得到健康基座（2026-11-12）\n\n'+brief+'下文“等待开机/正式NO-GO”是前一阶段历史。现已完成授权任务，剩余费用预算不自动续跑。完整训练不足以支持target BN唯一因果；先修执行契约和审计实际更新图，不直接复跑长训或启用双动作。\n\n---\n\n'+old,encoding='utf-8')
p=R/'docs/BASELINE_NEXT_TRAIN_20261111.md';old=p.read_text(encoding='utf-8')
if not old.startswith('# 最新执行结果'):
 p.write_text('# 最新执行结果：协议已实施，终点健康失败\n\n目标BN只读检查、真实准入、完整随机训练和冻结确认均已执行。详见[最终报告](../evidence/baseline-target-retrain-20261111/summary.md)与[后续复核方向](BASELINE_TARGET_RETRAIN_20261111.md)。原始训练运行的清理失败保留；补测与原执行错误分别报告。本轮已关闭，下一轮真实训练需新的费用窗口。下文是实际执行前冻结的历史协议，保留不追溯改门槛。\n\n---\n\n'+old,encoding='utf-8')
doc='''# 目标BN完整重训后的复核交接

[完整结果与24格表](../evidence/baseline-target-retrain-20261111/summary.md) · [机器可读结论](../evidence/baseline-target-retrain-20261111/result.json) · [首批目标BN证据和失败](../evidence/baseline-target-audit-20261111/summary.md)

目标点策略依赖batch的问题已用同权重/同focal输入隔离，修复后上下文影响为0；但BN-only最终健康失败。不能将“找到真实缺陷”与“已充分解释训练失败”画等号。只保留一个seed且没有新健康终点，双动作暂不接入。

下一次长训前先完成三个可短测的问题：

1. **执行层**：3秒等待通过短准入却在长连接结束失败；报告实际看到sudo fd3仍持有listener。核查native socket的close-on-exec、子进程继承和PID归属，独立短测启动/停止/并发，修复后不靠长训验清理。
2. **学习图**：现有证据已列出critic参数集合重叠、target2重复赋值和共享Adam。逐图核对实际非None梯度、每个时钟推进和唯一target配对；集合重叠不自动意味着重复优化，重复无依赖赋值不自动等于2tau。先建立并测试精确更新契约，再做单变量有限学习。
3. **状态/目标**：用已有最终replay和冻结轨迹对照Q、原效用、边界动作及新连接状态覆盖。单连接与max_bw归一化的机制仍待区分；重连会同时改变覆盖和峰值分母，不称纯覆盖干预，也不同时悄改reward或网络。

如果代码契约修复仍不能在短有限学习中保留动作能力，再评估训练配方或奖励。短准入应包括学习行为观察而不仅是队列不溢出，但不能用早期坏点断言永久不可恢复；最终健康仍需事前固定样本预算、唯一终点和完整原门槛。

健康终点通过后做第二独立训练seed，再开展cwnd/MTP联合学习。双动作的固定周期强对照、同物理经验/更新额度、尾延迟与实际CPU必须先登记，不能把降低策略调用数直接等同CPU收益。

此前与Claude的协议讨论原文保留在[复核目录](../evidence/baseline-review-20261111/README.md)。本文件是本轮实测后的新交接，不冒称Claude已经审阅了新结果。复核时先看result、health-assessment、training-complete和原/补测两套run清单，避免只看平均吞吐或误将FAILED集群判为COMPLETED训练。
'''
(R/'docs/BASELINE_TARGET_RETRAIN_20261111.md').write_text(doc,encoding='utf-8')
print('Updated current documentation')
