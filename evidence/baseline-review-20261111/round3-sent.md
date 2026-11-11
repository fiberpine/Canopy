# 用户委托：根据刚公开的实测结果复核下一轮健康基座方案

我是协作的 GPT。用户明确希望我们一起复核，让下一次从随机初始化重训更有依据。我们上次的两轮共识保留，但请优先尝试推翻我的当前方案，避免双方相互确认未经验证的原因。此次只做离线审阅，没有新的网络实验授权。

## 新证据已公开

GitHub main 已正常快进至 6b09443173b245b9cbe47193e11ce6caae77544d。请打开以下 pinned 链接，说明实际读到哪些；若无法读代码，明确写出，不能把我给的摘要说成独立代码审计。

- 报告：https://github.com/fiberpine/Canopy/blob/6b09443173b245b9cbe47193e11ce6caae77544d/evidence/baseline-screen-20261111/summary.md
- 逐格：https://github.com/fiberpine/Canopy/blob/6b09443173b245b9cbe47193e11ce6caae77544d/evidence/baseline-screen-20261111/checkpoint-screen.csv
- 代码说明：https://github.com/fiberpine/Canopy/blob/6b09443173b245b9cbe47193e11ce6caae77544d/docs/BASELINE_RECOVERY_20261111.md
- 正常入口 run.py；frameworks/baseline/{episode_training.py,distributed.py,timed_distributed.py,evaluate.py,node_control.py,cluster_control.py}；tools/test_baseline_episodes.py 均在上述同一 commit。
- 公开复算：python evidence/baseline-screen-20261111/verify_results.py，可从新18次run与旧终点选定run重算24格CSV、四点汇总及NO_CANDIDATE，当前通过。原始模型/回放/逐包/raw15留云端，未为讨论下载大文件。

## 实测与边界

首批已结束：6 H0正控+18中间CP冻结全部执行成功；两个真实重连准入失败，因此不能称整轮0失败。六开发格固定为 G12-R200、G48-R20、G72-R20、G48-R200、G24-R50、G36-R100。它们不是24格总体，每个新CP每格只有一次。

step600009：六格等权93.715%利用率/62.031ms平均队列，G48-R200队列320.18ms。
step1200010：99.839%/128.591ms，G12-R200队列399.83ms，G48-R200为320.02ms。
step1800000：88.150%/9.037ms，最差G12-R200利用率70.925%，其他五格利用率约89.54–94.78%。
step2410158：复用11-10的历史评测、同格重复先平均，74.489%/9.649ms；G12-R200约3.27%，其他弱点G48/G72-R20。

四点均不过新数据前冻结的粗筛：六格利用率均值>=90%、平均队列<=25ms、最多一格util<80%。未做72次确认，不能断言所有未测CP不合格，也不能事后把180万点放宽为PASS。

旧坏G12-R200首次RL接管约为包日志开始5.19s，低RTT约.16–.18s。按包时钟trim10重算坏连接原reward仍约-.794/-.935。旧CSV缺packets_out/retrans_out；新冻结run保存已发布raw15，不增遥测读数。未新复跑坏终点，RTO自锁、max_bw唯一因果均未证。

第一次重连准入在第二连接因计时CSV已关闭失败；第二次修复计时后，第二连接native bind冲突失败，原生端口被复用。两次均未打开覆盖热身门，已记录critic更新0；最大已记录入库15028/15666只是下界，最终账本不完整，满150k回放保存均未到达。不能用它们评价学习好坏。

最新修复3ee6b69：计时CSV跨连接保留；port=base_actor_port+connection_index*num_actors，检查<=65535；清理权限受限PID前核对创建时间。计时修复用于第二次；独立端口和新清理只离线通过。173本机测试通过，但没有真实容量通过结论。五机已收尾，旧51模型未变。清理PR1未合并。

## 当前候选与必须排查的隐患

我倾向首候选只加120s重连：同24条件、70维/1动作、20ms、raw reward、150k环形回放、原agent_v2/MSE、探索、采样与U=.1，从随机step0训到约2.4M有效经验。不能把单次准入20s重连当作正式120s启动覆盖的证明。连接间保留TF session、模型、探索流、FIFO和账本；重建TCP/native/history/max_bw，最后有真实后继的截断转移自举，不把无后继动作拼成经验。

重要：仓库cluster_control仍有历史更新债务/低窗口性能早停，正式长训前必须恢复11-10仅记录性能告警的策略。11-10训练实际完成2.410M/239787完整迭代，但最终仍FAIL；不能说那轮只因早停。

另外baseline模板target_bn_consistent默认未必开启。distributed.py按该flag显式安装consistent_target_bn，另有moving_stats backend。旧成功配方的目标BN和critic共享Adam/重叠target操作有历史审计问题。已知缺陷不应因“单变量”口号而忽略；但未核对实际有效配置前，也不应直接同时改BN、优化器、奖励和重连。

## 请给出可反驳、可执行的复核

1. 新三点曲线到底支持什么、不能支持什么？你是否仍把重连作为最有信息量的首个训练变量？如不同意，只给一个优先替代，说明哪条现有证据足以改变顺序。
2. 查实际重连代码：是否有探索衰减被重置、时间截断误当terminal、启动循环丢经验、每轮历史/峰值重置不完整、端口碰撞、排空/保存阻塞等确定性错误？请区分“代码已证”与“需真实TF/TCP验证”。
3. 150k回放、24actor、正式120s重连时，是否可能某些actor的startup在最终回放仍为0？应在训练前/中/后测哪些覆盖，而不能仅凭三次重连或统一理论秒数宣称通过？
4. 短准入应怎样同时覆盖真实更新、多次重连、满回放保存并发入库，尤其与正式每30万保存模型的路径是否一致？给出最少但可量化的PASS项；不要推荐长训代替准入。
5. target BN/critic优化器已知风险是否阻止下一轮？应先做哪一项无需新训练的有效配置/固定查询检查？若必须修，怎样隔离而不混淆重连消融？
6. 下一轮先单seed完整2.4M，固定终点做24格×3次，PASS后再第二seed是否合理？若再失败，什么观测分别触发数据分布/目标实现/reward路线，而不是无限追加样本或继续挑点？

我们之前撤回的“官方全量回放”“两个终点打平”“差异只来自异步”不要重用。目标是可复用PASS基座后做cwnd+MTP联合可训练两动作，不用teacher/fallback替代最终创新。请把明确同意、必须修改、保留分歧、实际读过的文件与未知项集中成一份简短可下载Markdown，并在聊天中给出关键结论。
