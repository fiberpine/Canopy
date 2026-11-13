# 当前模型健康门通过，按用户要求停止重训并归档清理

**最终更新：96/96完成、0执行错误。** 当前seed3在24条件×3次的原健康门全部PASS：trim10条件等权利用率99.745330%、mean queue12.738410ms、条件P95均值18.069444ms、P9919.791667ms、零丢包；最差单次利用率97.463494%。源/模型、接管≤10秒、动作映射、优化/探索0与五机清理均通过。[逐run](../evidence/baseline-health-continuation-20261113/full-seed3-latest.json)、[原评分](../evidence/baseline-health-continuation-20261113/full-seed3-score/health-assessment.json)、[源/接管审计](../evidence/baseline-health-continuation-20261113/full-seed3-source-and-handoff-audit.json)可复算。

用户随后要求暂不重训，先采用当前版本为健康基座并清理三地。新消息前seed7已于约16:29启动，收到后立即安全停止：165396真实经验、15310完整critic/actor/target更新、30620critic优化器操作，24路generated/enqueued/inserted一致且整数信用排空。整个计划标USER_STOPPED_PARTIAL_NOT_6M_COMPLETION；底层干净COMPLETED不等于完成6M，无seed7性能结论，预登记78次没有执行。[停止账本](../evidence/baseline-health-continuation-20261113/seed7-user-stop-final-ledger.json)与[五机终检](../evidence/baseline-health-continuation-20261113/closure-user-stop.json)保留。

**当前仅确认这一权重在明确静态网格和自然部署协议下健康；训练配方的重复成功尚未验证。** 用户的新决定取代“两个完整seed后才采用/清理”的原阶段条件。归档入口见[当前基座](CURRENT_BASELINE.md)，旧失败不覆盖。

训练前仅对结束容量测试的42个指定文件无损gzip，原SHA与解压SHA逐一相同，累计节省1224715861个payload字节；没有改动正式训练日志、模型或源码。最初部署检查误要求actor具有仅learner需要的SSH文件，已修正检查范围、保留辅助记录，未改变算法，也未传私钥内容。

下文保留此前事前计划和中途状态；“进行中/尚未训练”仅为当时记录。

北京时间2026-11-13新窗口15:54:39—19:54:39，用户已明确同意240分钟上限和按门槛继续；末10分钟收尾。用户另确认上轮是助手额度耗尽后自行关闭服务器，属于外部中断，不计为算法或实验执行失败。[明确原因记录](../evidence/baseline-health-continuation-20261113/interruption-cause-user-confirmed.json)补充先前只知道重启、尚不知原因的历史报告，原部分日志仍保留。

原4个完整自然接管run保留，只按完成性补齐8个缺失槽，两份被截断的原尝试用新ID重做。补测8次全部完成、0执行错误；[合并筛查](../evidence/baseline-health-continuation-20261113/recovery-screen-assessment.json)通过原逐格、任一次低于50%的灾难门、首10秒利用率保护及接管/动作/模型/计数/清理审计。

|条件|完整次数|trim10利用率范围|
|---|---:|---:|
|G12-R200|6|99.4145%—99.5448%|
|G48-R200|4|99.9158%—100.0000%|
|G72-R20|2|99.9532%—99.9860%|

这12项仍为探索筛查，原4项和新8项分列，跨重启的混合结果不当作新一批独立确认。原40项首反馈FAIL、原seed3旧启动FAIL均不变；自然接管通过不证明小窗口恢复、动态网络或留出条件已经修好。

用户已确认actor2—4开启。五机身份、原51模型、旧core/sysctl、CUBIC/HyStart/no_metrics_save/缓冲、实际自然ELF及helper核对通过。**独立96次已于16:05左右启动**：当前同一seed3终点24条件×3，加上同一自然接管ELF的H0全24×1；五机各同时一个冻结流。顺序/主机按事前Random20261114固定，12项筛查不复用进完整验收。当前还没有完整96次结论。

完整确认沿用原绝对健康门，另检查每run实际`ssthresh<cwnd`接管≤10秒、辅助写窗0、完整trim10策略闭环、模型与优化/探索计数、动作映射和清理。H0是同协议参照，不把不同启动链的历史均值混作本轮因果对照。

只有完整确认PASS且剩余≥150分钟，才最多一次seed7 fresh6M：1learner＋24actor，学习核心与seed3运行字节一致，精确克隆实际配置，仅改变seed及run/路径/端口/截止/元数据；仍用legacy训练native、自然协议冻结部署。终点预先固定72次加6个H0哨兵的78个槽，当前只登记顺序，尚未训练、未绑定或挑选seed7模型。不扩actor，不启双learner，不自动追加第二次长训。

清理仍须两seed端点、训练稳定性及可恢复归档校验都通过，当前未删除旧基座代码或覆盖H0。[上轮事实、机制限制与Claude实际复核范围](BASELINE_STARTUP_VALIDATION_RESULT_20261113.md)保留；不会因筛查通过就提前宣布健康配方。
