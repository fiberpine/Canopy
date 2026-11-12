# 在线BN最小修复：当前进行中

用户明确取消双learner提案，要求按此前修复路线继续。本轮维持1learner+24actor，按事前告知的此前240分钟上限新登记，16:11:10至20:11:10；20:01:10结束任务留10分钟收尾。只有一次fresh6M长训额度，之前各轮完整保留。

已补缺失的四路控制：step0、旧150万和上轮最终6M，各从同一现场恢复24次。原并行重复写分别8/6/9种结果，先allBN再actor分别8/8/5种；每统计只写一次，无论保留并行或串行，三个保存点均1种且逐变量/部署动作范围0。各统计的两路writer单独执行数值完全相同。新候选实际selector在旧150万又重复24次：unique并行为1种，旧并行7种。

因此选最小去重：canonical Sonnet fetchable wrapper每moving变量一次，仍原joint actor fetch，不加排序、不改网络、reward、noise、U、replay或连接。它会把重复EMA写入改为一次；确定性短检不证明性能改善。不是把所有BN操作全禁用。

相对上轮6M，保持targetBN/独立criticAdam/唯一target配对/CLOEXEC，唯一学习执行改动为unique_online_bn_updates=True。另修capacity_only收到producer全量就提前退出、未消化整数更新额度的工程收尾；正式训练原本已有排空逻辑，其停止条件不变。旧48失败结果不改写。

新增正常CLI参数--unique-online-bn-updates，默认路径不变。187离线检查通过，实际TF新selector检查通过。五机隔离部署相同；下一步真实240k/150k replay/live保存准入，通过才fresh6M；健康门与最终24格×3+H0全24各1次均保持，当前没有新训练/健康结论。

[登记与小证据](../evidence/baseline-online-bn-repair-20261112/registration.json)；[真实候选检查](../evidence/baseline-online-bn-repair-20261112/candidate-actual-contract.json)。Claude本轮新复核尚未完成，不称已达成新共识；此前讨论与原全部负面保留。


16:28进度：真实准入PASS，240000经验、22771完整更新、满回放/live保存/86.55秒稳态/全部清理通过；随机step0全部108变量与上轮一致。正式onlinebn-base-seed3-6m已启动，运行commit9bb1500eaf791bcb0e97724c63dcb725323dd5cd。只改在线BN重复写，尚无新性能结论。最终96评测计划已登记，训练后接续，当前不要关闭服务器。
