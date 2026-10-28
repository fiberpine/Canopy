# 第1步原配方复现：原后期守卫停止，停点冻结对照完成（2026-10-28）

**没有完成2.4M复现，没有新的健康H0。** 用户明确允许合理长训后，本轮在五台机器执行一次fresh seed3、原native、原配方训练；实际1,675,235条时由原配方后期守卫停止。随后完成实际停点与归档模型24条件各一次60秒冻结评测。完整2.4M终点、候选筛选及所选模型两重复确认均为NOT_RUN，没有自动续训。

## 实际执行与配方核对

- 1learner＋24actor，四台actor机各6条件；70维/1乘性窗口动作、agent_v2/Sonnet-IBP、原20ms执行链，5BDP、150k replay、balanced_actor、warmup12288及每actor512、UTD .1、iid探索.2、原moving-stat actor损失、原共享critic Adam及重叠target赋值组合。不是L1的TF-layers Orca，也不是双动作固定MTP模型。
- 原native `90ccb86a...`，原门控×1.1辅助与20ms pacing、原4MiB发送/6MiB接收/core212992，不flush缓存，不加启动锚定、重建连接、zero_sum、tau/n-step/reward改动或新critic放行门。
- 新旧随机step0的**106变量逐项完全相同**，哈希均`7ace55ea...`。去掉首次停止快照保存语句后，实际distributed训练代码与归档现场逐字一致。保存频率由500k改200k，独立路径、超时及评测余量等调度差异完整登记；异步TCP、到达顺序与minibatch不保证历史轨迹逐位相同。
- 90分钟绝对截止23:27:39。没有用完余额追加实验；所有正式网络和收尾都在本轮范围内。执行以[source manifest](../evidence/base-reproduction-20261028/source-manifest.json)和native SHA识别；`745bb47`是启动后的源码归档，不冒充运行时commit。

| 训练账本 | 实际 |
|---|---:|
| 真实generated=enqueued=inserted | 1,675,235 |
| 完整critic更新 | 166,294 |
| critic优化器操作 | 332,588 |
| actor / target更新 | 166,294 / 166,294 |
| cluster墙钟 | 1855.412s |
| 2.4M中尚未采集的预算 | 724,765条 |
| 已获得但最终未使用的原更新额度 | 0 |

触发的是原`PERSISTENT_LOW_WINDOW_LOW_UTIL_ACTOR0`：从150万条之后，G12-R20近期无噪声动作最小边界比例≥.99、平均cwnd≤10、近5秒利用率<.25，连续180.197s，达到原180秒规则。**不是百万窗口早停**。最后近期24条件17负饱和、7正饱和；本轮不能称恢复。原unit的COMPLETED表示干净排空，外层`TRAINING_BUDGET_NOT_COMPLETED`失败表示2.4M未跑满，两者同时保留，不改写状态。

首次观测停止时先保存全部106变量，之后沿用原配方排空；首次停止完整更新166,258、最终166,294，实际另排空36次critic/actor/target。首次停止已获得未使用额度1，其后仍有真实经验排入，所以不能把它与2.4M尚未获得的预算混称。本轮最终与首次停止有53个变量变化，两个快照均保留；终点106变量有限，150k replay birth连续，真实入库和回放采样账本一致。冻结评测使用的是**排空终点step1675235**，首次观测停止step1674884未另做网络评测，不能把终点冒充触发时策略。

另一个原元数据限制：保存的共享global_step为498,801，而分别记录的完整optimizer操作数为498,882，相差−81。原变量清单中“global_step counts optimizer operations”的继承文字原样保留并更正：本轮不能把它当精确更新账本，使用Python在真实sess.run返回后的分离完成计数；未据此额外改原配方或断言唯一原因。

## 停点的完整冻结评测

五机独占一流，每条件停点/归档同host、顺序事先固定随机，均用原native及启动。48次完整60秒、0网络运行失败、0optimizer/探索；12个actor/BN变量加载逐tensor相等且每run前后不变。它是**实际停止点的诊断**，不是2.4M终点或选模后确认；每格仅1次，全部为训练网格。

| trim10、24条件等权 | 利用率 | mean queue | 条件等权P95 | 条件等权P99 | drop |
|---|---:|---:|---:|---:|---:|
| 新训练排空终点step1675235 | 44.6637% | 190.5712ms | 229.9583ms | 230.8333ms | 1.97405% |
| 同日归档step2412694 | 97.9047% | 32.2477ms | 43.5417ms | 46.7917ms | 0.00000% |

停点预登记粗筛（util≥95%、mean queue相对32.32ms±30%）为**FAIL**；同日归档参考为**PASS**。逐场景、full/trim10、P95/P99/drop及控制周期全部公开，不能以总均值掩盖场景失败，也不将P95均值说成全流量联合P95。原资源记录仅Python/native直接进程树，未完整捕捉重挂接Mahimahi，不宣称完整CPU收益。

## 独立IPC短测：卡顿定位与仍未知的地方

训练之前先做G12-R200/G72-R50两臂各4次，共16个60秒冻结run，0optimizer。两臂用同一增加阶段日志的诊断重建native `12a4497f...`，env切换ASCII/framed；**训练用的仍是未改原90binary**。两臂同样记录cadence、样本等待、状态发布、动作执行和Python read/inference；不能冒称短测的重建binary与90逐位相同。

- G12-R200 trim10：legacy平均利用率77.5923%、verified91.8446%。legacy有199个>50ms原生周期，verified65个；**264/264均以等待新RTT样本为主要耗时**。状态发布到动作执行P99各run最高约2.76ms。两种传输都可出现长间隔，本轮不支持“只在verified发生”；空读多也不能独自证明IPC损坏或普通CPU扰动。这只能定位本轮重复，旧未记录阶段时间戳的单次成因不能回溯唯一证明。
- G72-R50：legacy98.5063%、verified95.9786%，host2 verified仍只有89.2774%、观测峰值/容量约.901；其余多数97–99.6%。没有重复到原58.6%的严重程度，但不足以宣布IPC逐场景等价，或排除启动/传输时间/host交互。
- verified的8次传输prefix/CRC/状态绑定核验通过，incoming rejected0；策略变量前后不变。没有把少调用、loss下降或符号一致当成新网络质量收益。

## 调度失败与限制

首次停点评测调度的总标记与单机phase同名，host0启动断言失败，四机TF启动阶段被协调停止。五机只读核对显示无native ownership、**0条评测TCP、0成功结果**；失败源代码/标记/traceback原样保留。修正为独立controller、worker、phase、STOP_ENDPOINT后才补齐48项，不重复成功TCP，不重启训练。原2.4M失败和该辅助失败没有被最终48完成覆盖。

原登记六条件checkpoint筛选含G48-R200，其42.016ms上限比历史相同子集65.4671ms还严格；在新checkpoint评测前已记录这一限制。它不是训练放行门，也不是复现的必要条件；本轮因原后期停止，该筛选**全部NOT_RUN**，不事后改阈值挽救结果。

本轮结论只到：**同一106变量起点和原算法/原native下，这次轨迹在原后期守卫前没有恢复，停点冻结表现如上，未得到新H0。** 它不证明若取消原守卫跑到2.4M必失败，不证明唯一根因是启动、BN、探索、异步调度或奖励，也不把一次历史成功升级为普遍可恢复。下一步须与Claude根据这些证据重新定方向，本轮不自动续训、改守卫或追加机制实验。

五机最终归属进程/共享内存、原源码/native、旧模型、51保护文件及sysctl核查均PASS。新完整CP、replay、逐包/state/control日志留云端；仅精确actor12tensor通过VPC供冻结推理、小证据回本机。没有完整TCP/history/RNG/队列无缝resume，不下载大文件替代验证。

[完整分析](../evidence/base-reproduction-20261028/final-analysis.json) · [96个endpoint窗口](../evidence/base-reproduction-20261028/all-network-windows.csv) · [全部登记/运行账本](../evidence/base-reproduction-20261028/run-ledger.csv) · [原守卫证据](../evidence/base-reproduction-20261028/collected/host0/results/guard-trigger-proof.json) · [首次停止/最终快照核对](../evidence/base-reproduction-20261028/terminal-tensor-audit.json) · [IPC阶段时钟](../evidence/base-reproduction-20261028/timing-refined.json)。

---

以下为第0步原复盘与历史讨论；其中“第1步未启动/先申请长训”等状态已由本节覆盖，历史数值与失败保留。

# 回到归档配方：证据核对与第0步实测（2026-10-28）

**第0步已完成并安全收尾，0训练；第1步2.4M未启动。** 先复现已知配方的方向合理，但目前建议先短重复两组有大差异的冻结配对，再决定长训。不能把“历史曾恢复”扩大为所有配方必经且必自愈，也不能由逐场景同号证明critic所有状态正确。

## 归档轨迹确认了什么

归档成功模型是`frameworks/baseline`的Canopy `agent_v2`/Sonnet-IBP工程配方，70状态、一个窗口动作、5BDP、seed3、选定step2412694；不是L1的TF-layers Orca实现，也不是完整官方论文复现。原三次冻结评测trim10条件等权为97.75%利用率、32.32ms排队。

| 来源 | 可以复算的事实 | 范围 |
|---|---|---|
| `balanced-trajectory-329992.json` | 16个actor动作均值≤−.95；7个≥+.95；另1个约+.417。8个窗口均值≥百万，其中7个INT_MAX | 当时每actor最近≤5秒在线片段，不是全训练轨迹或冻结评测 |
| `policy-diagnosis-500010.json` | 1704个查询输入中，5条件动作均值≤−.99，17条件有部分输入饱和；没有条件查询均值精确−1 | 冻结checkpoint对近期历史的离线查询；不能混成“6个在线动作全是−1” |
| `policy-diagnosis-2412694.json` | 同样1704个查询输入全部无饱和；各条件Q均值82.52—100.39，近期在线利用率条件均值约97.21% | 近期在线利用率不是正式97.75%冻结评测口径 |

因此，早期饱和和极大窗口**不能单独证明永久失败**，百万窗口S3若直接用于该历史轨迹也会截断这个成功样本。另一方面，一个最终成功、被选定的seed不能证明早期饱和是必要或有益探索，不能识别恢复唯一原因，更不能证明改变了多项配置的L1继续必会恢复。

L1的28,513条是原120万预算的**2.38%**；相对2.4M才约1.19%。原停止记录保留，不能把未完成训练称成已证明不可恢复。

## 原配方与L1不能混称单变量对照

启动（归档门控×1.1辅助/立即接管）、队列（5/2BDP）、agent实现、critic Adam共享方式、target/BN路径、replay（150k/300k）、头初始化、探索时序、IPC、主机缓冲和停止规则等确有多项差异。共同采用raw reward、balanced_actor、warmup12288及UTD .1并不足以消除这些混杂。

其中必须更正：**归档基线本身已经使用moving-stat actor损失**。`frameworks/baseline/distributed.py`在moving_stats后端安装`moving_stats_backend`，后者的actor损失查询`_moving_action`。L1是另一agent上的具体实现变化，不能说“原配方不用moving-stat actor损失也成功”，再把这一概念当作全新单变量加入。

启动态并非已证“完全无训练支持的分布外”：此前已复算首条状态确被采样。历史BN/critic/探索诊断保留其实际发现和失败，不再将它们升级为已验证的必要修复或在线放行证书。后续算法健康必须依靠预登记预算、checkpoint选择及完整性能验证；数据正确性、有限预算和基础设施停止仍需保留。

## L2-A逐场景符号复算

| 事后名义阈值 | 有响应估计且有probe的计分格 | Q1、Q2均同号 |
|---|---:|---:|
| J10，\|z\|≥3 | 14 | 14/14 |
| J10，\|z\|≥2.5 | 17 | 17/17 |
| J40，\|z\|≥2.5 | 2 | 2/2 |

3个actor×延迟类缺响应估计或probe，明确排除而非通过。这些阈值是后验描述，未作多重比较校正。按actor分层修复了此前RTT组的场景权重错配，但类内状态/时段权重仍可不同，补偿探索创新响应也不等同持续动作Q导数。结果支持“原聚合门不合适”，不足以证明critic全部正确、无其他误差或闭环已健康。本轮未追加L1/L2训练、机制干预或放行门。

## 第0步：今天环境中的冻结评测

原计划30分钟，用户明确批准延长15分钟，截止北京时间20:52:26。五机严格原known_hosts身份核验，归档模型index/tensor哈希等同Git选定模型，原4MiB发送/6MiB接收/core212992缓冲保持，不flush缓存。各host独占一条流；每个条件两臂同host、原顺序随机。先4次12秒准入，后24条件×原native/verified各一次60秒。

两臂都保留**完整原门控×1.1辅助及20ms pacing**。没有沿用立即接管的scalar native；新审核native恢复了原启动代码块，测试逐字核对。云端原native SHA `90ccb86a...`与历史成功实验回执相同；Git迁移包内`88c32e6a...`是后来构建，不能静默混称。新审核native使用历史`g++ -pthread`编译选项。

| trim10，24条件等权 | 利用率 | mean queue | 条件等权P95 | 条件等权P99 | drop |
|---|---:|---:|---:|---:|---:|
| 原native | 98.6938% | 33.0344ms | 43.8771ms | 67.3883ms | 0 |
| verified IPC | 96.3417% | 31.7452ms | 42.8333ms | 46.1250ms | 0 |

两臂均通过事先登记的粗筛（利用率≥95%、mean queue位于历史32.32ms的±30%）。这支持**原归档模型在今天环境仍能达到接近历史的总体表现**，不证明今天从头训练一定成功，也不替代逐场景质量验收。

| 弱配对 | 原native利用率 | verified利用率 | 差值 |
|---|---:|---:|---:|
| G72-R50 | 99.4850% | 58.5799% | −40.9051pp |
| G12-R200 | 95.0095% | 79.0092% | −16.0003pp |

其余22对平均差约+.0207pp。两弱条件都在同一物理host；G72-R50的两臂跨修复阶段，G12-R200均在R2。这些每格仅一次的结果不能区分启动随机性、运行时扰动与传输/时钟影响，不能宣称IPC已逐场景性能等价，也不能将其唯一归因于IPC。完整full/trim10、逐场景指标与差值全部保留。

CPU作为诊断记录：原评测器只追踪Python/native直接进程树，未完整捕捉重挂接的Mahimahi。因此不据此宣称实际完整CPU优势；各run的可用范围与原始资源口径保留在附加指标中。

## 失败、兼容修复与补测账本

V1完成25项正式60秒后，G72-R100 verified在首动作前因范围校验失败，协调器停止后续运行。另4项被中止、18项未运行。部分中止底层result写COMPLETED但wall不足60秒，**不纳入正式性能表**。

原始首帧的真实TF重建得到动作`−1.0000001192092896`；原`int(100*4**action)`映射为24%，而旧verified协议仅接受25—400。修复仅限独立审核副本：允许24—400，不clamp策略、不改原百分比映射，不修改全局scalar协议、原baseline或模型。Python边界测试及真实g++边界程序验证24接受、23/错误绑定拒绝。R2新连接没有实际发出24%帧，故不冒称本轮TCP重现了该具体帧；重建和独立原生边界验证分开记录。

用户明确批准延长15分钟后，R2只补23个失败/中止/未运行项，成功25项不重跑，最终恰48项完整正式结果。原失败、中止、未运行及两种协议源均保留。24个最终verified run的帧prefix/CRC/状态绑定检查通过，incoming rejected0。

另有辅助故障：本机路径优先级导致裸`import deploy`误载旧eager部署CLI，参数deploy/launch被当成未定义主机名；SCP失败发生在远端变更脚本之前，没有新增TCP/优化器。它重写了旧实验的两份本机部署素材。已保存误写副本，改为绝对路径加载，并从严格认证的云端原包按历史SHA `4b66b010...`恢复部署包、重建原清单。旧代码、模型、结果及云端旧实验未改；恢复回执和素材差异均保留，不能声称旧本机文件从未触及。

五机两阶段收尾PASS：本轮进程/归属IPC0，保护51文件、归档checkpoint、原native和系统参数未变。最后原生边界验证在截止前完成，0TCP；五台已可关机。所有模型tensor、逐包/state/control大日志仍留云端，没有下载新模型或replay。

## 下一方向与授权边界

我赞成先复现，再逐项变化；**当前不建议立刻进入2.4M长训**。先用明确的小预算，对G72-R50/G12-R200做同host多次冻结配对，记录相同启动链和控制时钟，判断这两次大差异是否重复。确认执行链后，再另行批准一轮原配方2.4M；预算须含约39分钟历史训练参考、准备、checkpoint选择与冻结评测余量，不能只分配“40分钟训练”后挤掉验收。

第1步若批准，应冻结完整配置与执行SHA，保留原配方已有moving损失、5BDP、150k replay、iid探索及原缓冲，明确verified传输兼容范围；事先登记checkpoint候选/验证集/选择规则，保留原基础设施、预算和后期持续低利用停止规则。它仍只是单seed工程复现，不能直接宣布论文复现或完成三seed/双动作目标。本轮未部署新训练、不加载旧权重续训、不启动长训或新机制门。

复算入口：

```sh
python3 tools/scalar_learning_check/baseline_history_review.py --root . --output /new/path/history.json
python3 tools/scalar_learning_check/scenario_alignment.py --evidence evidence/scalar-critic-only-L2A-20261028 --J 10 --z 3
```

[完整第0步分析](../evidence/base-first-principles-20261028/final-analysis.json) · [96个完整窗口及P99](../evidence/base-first-principles-20261028/all-network-windows-with-p99.csv) · [逐条件配对](../evidence/base-first-principles-20261028/paired-conditions.csv) · [历史复算](../evidence/base-first-principles-20261028/history-review.json) · [执行与数值保真](../evidence/base-first-principles-20261028/publication-provenance.json) · [辅助恢复回执](../evidence/base-first-principles-20261028/auxiliary-import-incident/recovery-receipt.json)。
