# 执行契约、晚期恢复与开源RL参照

2026-11-12，当前6M训练仍在运行，最终性能待冻结评测。文献用于提出检查，不替代本项目实测。

## 已证实的BN重复写，不等于已证明性能根因

实际运行环境为TF1.14 / Sonnet1.34。正式step-0保存图中，4个在线BN统计各有两个`AssignSub`来源，8个操作的`use_locking`均为False。固定旧CP1500000、同一512条真实状态、同一优化器现场，重复24次：

|执行方式|不同参数结果|最大统计/参数差|固定状态部署动作最大范围|
|---|---:|---:|---:|
|当前actor与全部BN同一次执行|5|0.0184021|0.00319269|
|先执行全部BN，再actor|9|0.1289215|0.00276330|
|每个BN统计只取一个原有写入，再actor|1|0|0|

当前并行方式的参数差主要在BN，actor可训练权重在这24次中相同。原表的绝对统计差不可直接当性能差，固定输入动作影响约0.0032也不是TCP因果效应。单统计写入加顺序控制消除了本次一步不确定性，但同时改变EMA更新次数和actor读取统计的时点；**尚未隔离去重与排序两者的必要性，未在当前训练中启用此候选。**

[完整重复实验](../evidence/baseline-capacity-repair-20261112/actor-bn-contract-r3.json) · [独立进程重复](../evidence/baseline-capacity-repair-20261112/actor-bn-contract-r2.json) · [正式保存图与依赖源码SHA](../evidence/baseline-capacity-repair-20261112/BN-runtime-source-inspection.json) · [实际执行的审计工具](../tools/audit_scalar_update_contract.py)

这与依赖源码及API契约一致：Sonnet把统计更新加入更新集合；TensorFlow对未加锁的减法更新不提供并发确定性保证。实测与实际图属性是本项目的直接证据，不能只凭库函数名字推断。[Sonnet1.34源码](https://github.com/google-deepmind/sonnet/blob/v1.34/sonnet/python/modules/batch_norm.py)、[TF1.14 moving averages源码](https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/training/moving_averages.py)、[AssignSub官方契约](https://www.tensorflow.org/api_docs/python/tf/compat/v1/assign_sub)。

附带核查也排除一个过强猜测：同一个真实保存点上，快速moving路径与原零宽IBP在`is_training=True/False`两种情况下，输出、actor损失和8项参数梯度最大误差均为0。这里是固定状态和保存点的函数检查，仍不声称所有长期随机轨迹相同。

这些短检查在独立本地TF进程内执行，读取旧模型，未连接在线TF master、修改在线模型/队列/随机现场或保存旧模型。它们与当前训练有短时间资源重叠，区间已登记，因此本次训练轨迹也不能当作无额外资源扰动的因果对照。首次直接取TF1条件内部操作失败，原失败和修正均保留在[诊断溯源](../evidence/baseline-capacity-repair-20261112/BN-diagnostic-provenance.json)。

## 420万时实际恢复，反对过早判永久坍塌

既有监测在约420.5万入库时记录actor梯度0.07005、最小动作边界actor为0；此前很长阶段为17个最小边界、7个平均动作接近最大边界，梯度常为0。原始监测行见[晚期恢复记录](../evidence/baseline-capacity-repair-20261112/late-recovery-observation.json)。这说明该次训练的早期饱和可恢复；不证明最终性能健康，也不能把恢复唯一归因训练量。当前仍按原最终排空端点参加24格×3评测，不挑中间点。

## 从原始论文和作者代码得到的具体检查

|原始参照|相关事实|本项目如何使用，及边界|
|---|---|---|
|[作者TD3实现](https://github.com/sfujim/TD3/blob/master/TD3.py)|双Q总损失用一次optimizer step；actor/target延迟更新；其Actor没有BN|“共享optimizer”本身不是错。本项目的问题是一个TF Adam对象派生两次可并行minimize、重复推进同一时钟。独立Adam是已测修复；总损失单次minimize也是可研究方案，不能与“两次共享时钟”混称|
|[Spinning Up TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)|区分早期全范围随机动作、开始优化的热身和后续加噪；默认延迟actor更新|当前覆盖热身不是全范围探索，当前actor:critic为1:1。若BN契约修复后仍不稳，再独立检验优化节奏或探索覆盖；不同时照搬其学习率/噪声/回放默认值|
|[CrossQ论文](https://arxiv.org/html/1902.05605v4)与[作者代码](https://github.com/adityab/CrossQ)|研究off-policy分布差异下的BN，采用特定统计设计及BRN|它支持认真检查统计分布，不能证明当前重复写就是性能唯一原因。移植CrossQ或换归一化属于新算法实验，不是当前执行修复|
|[DDPG失败机制论文](https://arxiv.org/html/1911.11679v1)|在稀疏确定性任务中分析策略饱和和价值梯度失效，更多奖励数据也不自动保证恢复|Orca为连续密集效用，不能直接套用其定理。借鉴的是检查actor tanh导数、策略处Q动作梯度及实际探索支持；本次420万恢复更不能写成永久死锁|

本项目实际行为噪声是`noise_type=3`、不衰减Gaussian、sigma=.2；不是因为`noise_exp=50000`就会在5万步停探索。若无噪动作恰为±1，单次噪声把动作反向需超过5个标准差，概率约2.87e-7。这是源码和高斯分布的计算，不是已测的TCP逃逸概率；内核调整、状态变化、共享网络参数和BN仍可改变轨迹。已准备逐actor完整日志计数，区分实际反向探索与native窗口上限的动作别名，不凭图形猜原因。

## 后续最小顺序

1. 完成当前6M固定配方及预登记冻结评测；容量24已通过，48未准入，不为追CPU利用率改并发。
2. 用固定真实状态比较保存点的权重、BN、动作与Q梯度，保留训练到部署的覆盖缺口；函数交换不等同在线因果。
3. 若需要下一轮，先完成“只去重BN、保持原执行形式”与“去重并排序”的隔离短检，再决定最小实现修复；先做有限准入，再按表现决定学习预算。
4. 重连会同时重置history/max_bw/TCP，延迟actor更新也改变优化配方，均应分别登记。研究SAC/CrossQ或移除BN不能替代这个次序。

本轮新增Claude窗口交互被用户Esc中止，未发送这些新增证据，不能声称已经取得新一轮Claude共识。此前完成的五点复核仍见[原讨论记录](../evidence/baseline-capacity-repair-20261112/claude-review.md)。
