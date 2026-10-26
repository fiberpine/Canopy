# 健康V31兼容价值校准候选（2026-10-20）

> 2026-10-26发布说明：下文按日期保留原实验记录，并非当前服务器运行状态。
> 最新基线、奖励和窗口方向结果见[当前交接](RESEARCH_REVIEW_BRIEF.md)。
> 健康V31和原51模型文件不变；后续critic诊断未产生可替换的成功策略。

## 2026-10-21单次随机周期事实诊断（最新，原始V31不变）

用户继续授权后，本轮完成120次独立40秒真实TCP连接：六场景×两trace相位×两次独立重连×五种MTP。每连接仅在约8秒的真实决策边界随机替换一次20/40/80/120/160ms；窗口输出原样，随后恢复健康V31至少20秒。四台actor各同时一条流，learner做离线分析，没有actor训练、隐藏20ms观测、运行时teacher或native改动。

现场`/home/ubuntu/orca-v37-oneshot-20261020-followup`；本机`D:/Canopy/experiments/v37-oneshot-20261020-followup`。基础commit为d754c49；实际运行增量以audit/deployment.json及归档四份运行源码SHA为准，不能将事后文档commit当运行入口。trace计划五机SHA=b16acc480c741c0251c8e7cbd0e6c53303eca660cc073a635e4bd25d01ce98ad。相位1循环平移间隔序列，保留包数、周期与间隔多重集；这些trace是开发诊断，不能以后算盲测。

首次smoke因私有trace目录相对路径在TCP前失败，已修复并另建smoke40-r2，真实40秒通过。正式轮次原已完成41条，服务器关闭/重启中断3条；原失败与不完整日志保留。2026-10-21检查五机启动时间及无遗留进程后，复用41条，3条采用新`-recovery1` ID重试，补齐剩余未运行条件。原注册、原截止时间、原failure.json、恢复注册均保留；预算是有限恢复剩余试验，未重复完整轮次。部分第一轮区组跨重启，另报完全重启后第二轮敏感性，不能假装环境从未中断。

### 事实完整性与主要周期检验

全部120条完整结果、full/trim10共240网络窗口保留。逐条模型前后SHA相同；既有状态链连续、实际duration折扣正确、单次替换窗口不变。恢复父actor重算其余记录边界动作最大误差为**1.8242747e-05**。单次干预回报与相同原生观测时刻间的吞吐、mean/P95/P99排队及drop从已有日志对齐，不增加TCP读取。40秒不足以满足原25..50秒CPU端点，**不是CPU收益试验**。

主检验是Q对所随机周期的中心化偏好，能否预测随机化后20秒事实回报残差。按24个五连接区组等权；每场景/周期仅4个独立未来，不是同TCP状态的五条反事实，不是真实最优Q。区组置换和bootstrap只是本开发样本的有限诊断。

- 父V31：2.73792，区组bootstrap95%区间[0.68377, 4.70840]，区组置换p=0.01400。
- 父V31仅第二轮完整重启后区组：4.44547，区组bootstrap95%区间[2.13782, 6.94479]，区组置换p=0.01025。
- 4秒校准后，独立连接/相位留出：-0.45199，区组bootstrap95%区间[-4.37347, 2.74075]，区组置换p=0.85929。

必须按同一子集比较：父模型在phase1的12区组关联为2.398，区组bootstrap95%区间约[−0.441,4.857]；一步校准为5.133，[0.571,11.214]；4秒为−0.452，[−4.352,2.860]。4秒减一步的配对差为−5.585，区间[−14.285,0.810]仍跨0，不能宣称排序退化已在总体得到统计证实。父模型全24区组的正向证据以混合时间尺度最强（该场景置换p≈0.002），每个真实trace仅4区组，尚不足以确认其分别有效；也不能由总体正向宣称动态策略已经优于最佳固定周期。机器记录为paired-ranking-analysis.json。

1/4/8秒为补充描述性敏感性分析，在全轮结果完成前加入，未修改原20秒主门槛；短窗口bootstrap占比更大，更不能视为事实真值。source Q1/Q2、每输入五周期查询与逐场景指标均保存。奖励与网络目标的区组排序相关另记`secondary-objective-analysis.json`，它是描述性结果，没有另造加权目标或放宽开闸规则。

### 固定来源critic回归（不是新部署算法）

两臂同一健康V31完整函数、fresh共享Adam、相同scene-balanced batch，各2000完整critic迭代/4000critic优化器操作。唯一变量：一步或≥4秒事实回报目标；后续行为均为父策略。phase0训练、phase1独立连接/相位留出，公共≥20秒事实回报作诊断标签，尾项固定源critic。actor、BN、全部target未变化；不是完整TD3训练，也不是新的网络性能提升模型。重叠窗口不等于独立样本。

| 场景 | 一步留出RMSE | 4秒留出RMSE |
|---|---:|---:|
| 浅队列公交R20 | 26.036 | 53.239 |
| 公交R100 | 30.918 | 37.876 |
| 出租车R20 | 61.493 | 25.702 |
| 出租车R100 | 54.545 | 38.029 |
| 带宽突降R100 | 35.564 | 41.650 |
| 混合时间尺度R20 | 54.898 | 52.500 |

scene等权RMSE：43.909→41.499，相对改善5.49%。预注册要求整体改善至少20%，同时四个真实trace场景各不劣于一步的1.1倍；结果value screen=False，真实trace非劣=False，留出周期指导证据=False，联合证据gate=False。所有自动actor开放仍为False。

### 当前判断与可证伪的下一步

正确的累计ACK字节/actual_dt已实施；这不等于所有跨周期统计与奖励均等价。原延迟奖励仍取边界sRTT，不能反映区间P95/P99尖峰。MTP同时改变反馈盲区和原乘窗脉冲频率，不能被当作独立的节能旋钮。价值函数整体拟合变好不必然学对很小的周期价值差；周期奖励排序也不必然符合吞吐/尾排队门槛。当前尚未证明唯一根因。

本轮没有扩大联合actor训练，也没有论文级性能或实际CPU收益结论。先依据本轮留出排序、逐场景误差及真实网络目标一致性决定是否值得有限更新；若gate失败，不把更多actor、长训、换BN或制造动作方差当修复。后续若研究奖励测量或相对价值学习，应作为单独、匹配且预注册的变量；不能与本校准结果混称同一成功方案。

补充区间统计审计只读取120条已结束日志：公交R100共6142个有离开包的区间，其中259个区间（30.22秒）奖励延迟因子仍为1，而该区间离开包排队P95已超过50ms；38个区间超过100ms。出租车R100也有172个无延迟惩罚区间P95超过50ms。**这是边界sRTT与区间队列尾部并不等价的实证，不是证明native统计损坏或MTP导致了这些尖峰**：反馈有传播/ACK滞后，sRTT和离开包并非同一批包，原delay_margin也允许一定排队。20/50/100ms只作描述性切点，不是新的通过门槛。自然MTP分箱受状态混杂，单次随机probe区间另存，不从分箱均值做周期因果宣称。详见interval-visibility-summary.json及四机逐run小JSON。未改奖励、状态、协议或增加在线采样。

这使下一步优先级更清楚：先检验原奖励对逐场景吞吐/尾部约束的代理质量，以及已有周期相对价值在更新中能否保住；不要直接将4秒/20秒回报目标换入大训练。若代理目标确需修改，必须作为独立奖励/测量实验，明确schema与旧critic先验不再同语义，不能静默称兼容初始化。缺少观测信息时再讨论最小统计改动，当前尚无证据支持扩大网络、添加复杂模块或缩放窗口动作。

新探针通过已有冻结评测入口显式启用：在run.template('mtpjoint','eval')生成并localize的配置上，取一个condition后设置seconds=40、no_training=True、全新run_id、合法deadline及factual_mtp_probe_v37={period_ms:80,trigger_s:8,horizons_s:[1,4,8,20]}；随后运行`runtime/venv/bin/python frameworks/mtpjoint/repair_evaluate.py --config EXACT_CONFIG.json`。单次80ms smoke是确定性探针，不能冒称随机因果实验。完整随机化及恢复脚本随本轮小证据留存。该字段缺省时原V31路径不启用Probe；本轮无新增默认训练行为。

HyVar官方APNet展示证明动态MTP值得研究，但其selector/reasoner不同于本双连续头；TaCoS明确同步感知控制、区间累计回报，不能从通用合理性推导本TCP实现必然有效。参考：[HyVar官方展示](https://conferences.sigcomm.org/events/apnet2025/slides/Session3/3-4.pdf)、[TaCoS NeurIPS2024](https://papers.nips.cc/paper_files/paper/2024/file/746b0a1e6e3cff8d968ba6d2e6fff049-Paper-Conference.pdf)。

五机结束后无本轮活进程/用户IPC，51保护模型SHA与本机/原归档一致。小证据归档包回传3068993 bytes并逐文件校验，另回传四份区间可见性JSON共491011 bytes（逐SHA核对）；dense transition NPZ、两份diagnostic critic checkpoint和逐包日志留云端。原public main、冻结baseline和健康V31没有覆盖。

## 2026-10-20后续授权与完整实验（本节优先于下文历史状态）

用户随后明确要求继续研究并允许大规模训练，取代之前离线限制。五机已运行，本轮先完成有限配对诊断，没有以盲目长训代替定位。原始V31、baseline及全部51模型文件五机与本机SHA不变。原public main/tag未移动，研究仍在独立分支。

新现场：`/home/ubuntu/orca-v37-authorized-20261020-2248`；本机证据：`D:/Canopy/experiments/v37-authorized-20261020-2248`。实际训练算法源码为commit4fd822f；本轮额外修改验证工具，最终运行字节以deployment.json及五机小包evidence-files.json为准。不得将事后文档commit称训练入口commit。原部分run69640及停止证据继续保留。

最终真实TF1验证PASS：完整67非优化器张量和actor/Q/target查询最大误差0，actor/BN/slots冻结，critic确实变化，两head梯度非零，保存重载精确；留出诊断实际构图执行且op数不增长。1actor90秒真实smoke1042条、180完整critic/360操作/0actor；24actor35秒零优化容量通过。

两臂各300秒/actor、24actor，113个step0变量全部精确相同。保持原reward、Adam、架构、动作和实际周期语义：

| 臂 | 入库 | 完整critic / critic操作 | actor / target |
|---|---:|---:|---:|
| legacy_count_gate | 81450 | 14406 / 28812 | 9406 / 14406 |
| quality_gate | 81437 | 14404 / 28808 | 0 / 14404 |

两臂实际暴露约7201秒，最终无更新欠额。quality门槛因周期模型排序不稳定未开放，actor/BN精确为父V31；它是critic-only负结果，冻结性能波动不能算学到改进。control在5000后开闸。

四场景×8group×2次冻结60秒，共64次，0网络失败；full/trim10共128窗口。8group为父V31、两臂和独立训练fixed20/40/80/120/160。旧fixed谱系/覆盖不同，只作开发参考，不冒称完整匹配强对照。下表为trim10两次均值，单位利用率% / mean排队ms / P99ms：

| 场景 | 父V31 | 开闸control | fixed80 | fixed120 | fixed160 |
|---|---|---|---|---|---|
| 浅队列公交R20 | 74.84 / 7.00 / 40.00 | 77.37 / 7.29 / 40.50 | 76.08 / 6.99 / 40.00 | 82.12 / 8.68 / 45.00 | 78.86 / 8.12 / 45.00 |
| 公交R100 | 86.47 / 34.12 / 152.00 | 94.32 / 49.56 / 169.00 | 92.58 / 41.42 / 159.50 | 89.45 / 32.97 / 140.00 | 92.28 / 41.13 / 151.50 |
| 带宽突降R100 | 98.69 / 30.96 / 389.50 | 97.82 / 37.81 / 440.50 | 97.89 / 35.51 / 436.00 | 96.71 / 18.53 / 308.83 | 86.75 / 12.94 / 325.16 |
| 混合时间尺度R20 | 98.60 / 9.95 / 55.00 | 98.71 / 13.30 / 66.50 | 99.09 / 14.84 / 52.50 | 98.56 / 21.80 / 66.50 | 97.71 / 27.72 / 80.50 |

control浅队列trim10 MTP标准差仅约0.8ms，必要动作健康失败；高RTTmean和混合P95超过预注册非劣容限。因此拒绝当前control的综合收益，不能因整体输入动作有方差称所有场景健康。quality未更新actor，不能算部署改进。所有P95、drop、CPU、calls、冲量代理、RTT freshness和history跨度见final-analysis.json及逐run小包；一个训练谱系、两次冻结重复不是多seed。

同1024留出真实输入上，父/开闸control/冻结candidate的单步RMSE Q1约7.47/6.29/6.56，而周期Q160−Q20绝对差中位数约1.71/1.52/1.38；双critic周期差符号一致率仅约65—66%。排序是离策略模型敏感性，不是真实反事实。

真实冻结轨迹20秒以上回报审计：父V31浅队列/突降/混合的稀疏查询平均偏差约−36.56/−39.81/−53.61，冻结candidate约−35.02/−49.69/−61.40。尾折扣≤0.0065，且单独报告tail项。查询稀疏、窗口重叠、部署确定性与TD3 target smoothing不同，因此不能宣布真实最优Q已知或唯一根因已证。相同输入的共享trunk两动作梯度cosine约−0.014/−0.020/−0.076，未发现强整体梯度冲突。actor权重/BN参数交换极敏感，说明共同适配；不支持简单恢复旧BN当修复。

### 新的训练专用事实回报诊断

正式评测结束后，另采24actor各180秒无探索/无优化的健康V31轨迹。49938真实transition，连续性断点0；与恢复父策略重算的动作最大差3.10e−5，未改原模型。使用tools/v31_factual_targets.py，按真实duration/discount建立≥20秒回报，严格按连接切分：前60%训练，留出从切分后20秒开始，训练label终点不过界。24739训练/8664留出目标；这些重叠窗口不是独立样本或盲测trace。

同一初始化、固定源critic尾项、共享Adam、相同batch与2000完整critic迭代，仅比较一个transition与≥20秒事实回报目标；actor/BN/所有target不更新。本实验是离线固定teacher回归，不能冒充完整在线TD3或actor性能提升。最终actor等权留出RMSE40.07→15.90（下降60.33%），全行RMSE42.72→17.64。但20个合成条件改善，**4个真实出租车/公交R20/R100全部恶化**，例如出租车R20 23.80→38.23。因此统一改20秒目标No-Go。强证据是价值目标存在可修正偏差；反证是长目标的方差/分布偏移/状态信息限制不能忽略。

执行脚本与注册：云根factual_followup.py、factual_regression.py，configs/factual-target-registration.json；机器结果audit/factual-target-regression.json。源replay和两个diagnostic-2000模型留云端，绝不作为性能提升actor发布。工具4个纯数值测试与原18测试分开；不把数值测试冒充TCP。

### 继续投入的单一路线及失败标准

保留健康V31架构/状态/动作，下一步优先验证“真实时间、行为相容的价值校准”，而非加大网络或直接放大当前TD3：

1. 对训练专用连接，在可观测边界随机化一次MTP，随后恢复冻结健康策略至少一个校准窗口，记录真实动作、选择概率、每段duration及return。这样初始替代动作之后的continuation相容；不可将持续Gaussian/周期probe的多步轨迹当成确定性父策略回报。自然策略数据只能校准动作流形附近，不能证明其他周期排序。
2. 数据预算优先独立重连、trace相位/RTT/队列与真实trace重复，隔离合成/真实条件诊断；8664重叠留出窗口没有8664次独立未来。比较物理时间回报/短bootstrap的偏差方差，先要求四个真实弱条件不过度退化，再决定唯一目标方案，不能凭总体60%选择候选。
3. 只有真实留出价值及周期干预支持后，才以原actor更新规则做有限配对联合训练；失败则停止，不用放宽gate、加噪声方差或修改窗口乘法维持故事。若实证排序仍无网络收益，反驳价值校准为主要瓶颈，转向状态可观测性/奖励代理审计。
4. 只有逐场景网络质量与实际CPU收益通过，才扩大训练暴露及至少3（优先5）独立seed、统一训练fixed20/40/80/120/160与规则/纯学习/CUBIC/official Orca。当前没有进入论文大实验的充分依据。

MTP通过乘窗脉冲频率改变每秒控制冲量，同时改变观测盲区。不能把时间头当独立节能旋钮，也不能套用平滑连续动作Q优势O(dt)结论后私自按dt缩放原窗口动作。常值偏差思想实验：γ20=.995、γ160≈.96069，若未来Q偏差−50，仅折扣差可产生约+1.72长周期偏好，已接近观察到的时间Q差；偏差并非常数，故这里只给机制尺度，不作因果证明。

### CPU与结束状态

确认定制TCP_CWND update hook每次写窗printk，未改内核/服务/日志口径。四机被动observer在现有CPU窗口附近测得journald+rsyslog约0—0.25%单核CPU，host-global且不含全部printk内核开销，不能认定主要瓶颈。原all_owned范围包含Mahimahi、不含非归属daemon，后续日志/IPC/轮询优化必须独立对照，不能混入算法收益。

本轮网络与离线回归均已结束，五机归属活进程/用户IPC为0。回传五机小证据共4563234 bytes并逐SHA核对（另有直接下载小JSON）；所有tensor/replay/逐包日志继续留云端，没有下载大模型。后续训练授权仍来自用户最新指令，但无自动追加任务。

## 以下为前次暂停时的历史交付记录（不是当前运行状态）


## 结论与执行边界

最有支持的机制是周期、乘窗脉冲、反馈和价值学习耦合；尚未证明critic冷启动或开闸失配是唯一根因。候选保留149维/双动作、原奖励、20—160ms观测执行共周期、native协议及V31原Adam语义，新增完整同schema warm-start和独立actor更新门槛。没有新网络、提前唤醒、QC、fallback或teacher。

**最新用户要求以离线开发为限；本交付不授权后续云端训练。** 初始“执行并利用五机”的指令后，补充任务书到达前已经做了TF1检查、24actor35秒容量门槛及一组部分actor冻结校准；补充的“不执行正式训练”要求到达后，立即阻断pipeline、排空并停止。第二臂和正式冻结评测均未启动。不得把这次行政收尾称为完成配对实验。

当前最终门槛补丁尚未在TF1验证：本机Windows无TF1，状态为`BLOCKED_BY_ENVIRONMENT`。**Go：后续授权后的TF1验证和1actor smoke；No-Go：现在直接开始配对训练或论文大实验。**

## 来源与版本

- 不可变收敛快照：`85baf09b6f43a9f3223f7a8eaa1a4321b1a11b3b`，tag `consolidated-20261018`。
- 本分支从仅更新公开研究文档的`33d034908a66fd3b1b3931f353890487b4d471e4`派生。修改前主工作区clean。
- 源模型：`models/mtpjoint/model.*`，健康V31 step157704；所有原始模型文件保持不变。
- 配置来源：完整解析`templates/mtpjoint-train.json`和`environment/mtpjoint-original-training-config.json`。24actor的条件内有56个episode环境条目，不是56个并发actor。
- 分支：`innovation/v37-compatible-calibration`。早期运行代码按三文件SHA快照识别，不把后来的门槛补丁称为已运行版本。
- 本机独立证据：`D:/Canopy/experiments/v37-calibration-20261020-2220`。其中`runtime-source-r3`三文件已重建并与五机部署SHA逐项一致。
- 云端保留现场：`/home/ubuntu/orca-v37-calibration-20261020-2220`；只有旧的早期运行版本。最终离线补丁没有再次部署，不直接复用旧pipeline。

## 生效调用链

`run.py`/专用显式CLI → `cluster_control.execute` → `timed_distributed` → `distributed.setup`。setup依次安装moving-statistics actor、consistent target BN、trainable timing，然后构图及`build_learn`。moving backend覆盖actor loss为`-mean(Q1(s, moving_action))`；不是直接采用agent_v2中的旧actor训练实现。

训练actor把`evaluate.model`替换为learner参数代理，把transition hook接入TF FIFO；之后共用`decision_clock_v21.drive`和native。冻结评测独立恢复policy/BN，没有优化或训练探针。

两critic目标为`R + gamma**(dt/.02)*(1-terminal)*min(targetQ1,targetQ2)`。完整critic迭代包含两个optimizer操作。原双critic共用Adam时钟，本候选及control都保留它；V34独立Adam修正不混入本比较。target保持原tau=.001和BN同步机制。

`timing_learning_enabled=False`确实会把在线/target时间动作掩码成初始化周期。候选完整恢复源模型的True，Python learner只决定是否调用actor optimizer。冻结期间也不执行online BN更新；target仍按原tau追踪在线actor/critic。源target不会在加载后被`init_target()`覆盖，不能将target追踪误说为target永不变化。

## 证据与反证

实际解析的原始CSV：`evidence/causal/metrics/v36-causal-final2/{evaluation_runs,evaluation_summary}.csv`、`evidence/multialgorithm/metrics/v34-all12-final2-r3/{evaluation_runs,evaluation_summary}.csv`及`evidence/causal/audit/same-state-value-v36.json`。多算法为10group×12condition×2repeat=240run；summary中full/trim10是两个分析窗口，不是额外实验。

65个同输入查询：父V31最优20/160ms分别25/17次，中位Q160-Q20约-1.04；V35原奖励候选160ms偏好显著增加（41次，20ms仅2次，中位差约+1.79）。这是价值面变化的关联证据，不是真实网络反事实，更不能虚构Q1/Q2原始记录中没有的差异。

浅公交V31学习MTP约76.86%/7.28ms，强制80约76.48%/7.30ms，正常启动baseline约90.40%/13.05ms，支持窗口/状态/目标也弱；周期不是充分解释。高RTT同权重强制20约22.49%，支持重复乘窗和信息时机耦合。突降强制80约97.99%/18.69ms/P99310.50ms，对学习98.83%/31.49ms/P99404.33ms，但CPU更高。混合学习平均队列更低、CPU更高。不能把这些override当成独立训练fixed或精确同TCP状态反事实。

奖励的ACK字节/实际dt修复已经生效，但observed_peak受政策影响，低真实利用率仍可能得到较高归一奖励；边界sRTT无法恢复内部尾部。非线性效用的边界常数近似不等于区间积分。此次不改奖励。SMDP时间折扣正确不等于奖励排序或CPU目标正确。

Replay按actor均衡，25% batch用真实PeriodBank reservoir替换；周期bank是已有transition副本，不增加物理暴露。每新增20ms等效真实入库时间产生.08完整critic更新额度，warmup3600物理秒/12288条/每actor512。这个账本并不让loss自动成为按物理时间加权的目标。本候选额外预留`birth%17==0`真实样本，不进FIFO抽样或bank，仍计真实入库/物理额度；两臂同样处理。

## 实现及相对原V31的变化

1. `compatible_calibration_v37.initialize`：完整恢复actor、双critic、三个target、online/target BN和schema。校验149维、时间头、gamma、cost、区间计量/时钟和范围；逐张量形状/有限性/精确数值检查。仅epoch/global_step、Adam slots/powers保持fresh。不是96→149扩维、scratch或完整resume。
2. `distributed.learner`：opt-in兼容加载；把actor optimizer调用与时间forward分开；冻结期间定期和结束时直接逐变量检查actor/BN；记录开闸阶段和原因。预先检查三个loss有限性，非有限时在optimizer前报错。
3. `Replay`：opt-in留出真实样本；训练数据、bank仍只来自真实边界transition。留出最早1024条可能偏向启动和较早actor，不能叫独立盲测，也不能据其诊断估计总体泛化。
4. `Holdout.diagnose`：不使用训练batch；记录每周期Bellman residual、双Q差异、Q/target均值、actor梯度及同组输入的五周期模型排序。第一轮没有上次排序，flip为NaN，不能开闸。构图一次，诊断循环不创建TF op。
5. `actor_gate`：覆盖、更新数、明确配置允许、连续3次必要自洽门槛全部满足才允许actor更新；默认明确允许为False。阈值见`configs/v31-calibration-gates.json`。这些是必要一致性约束，不叫true_Q_accuracy，稳定排序也不证明正确。
6. `decision_clock_v21.validate`：允许第三种明确初始化来源；scratch、旧matching prior、新compatible三选一。训练actor转发来源字段，避免配置被入口遗漏。native调度/动作代码不改。

未设置`compatible_v31_checkpoint`时不增加TF变量、优化操作或holdout；原动作映射、奖励、replay抽样及gate分支保持原路径。最终补丁的未启用TF数值回归仍待兼容环境验证，不能只凭静态检查称已验证。

## 阶段、门槛和公平两臂

阶段`HEALTHY_ACTOR_FROZEN_CRITIC_CALIBRATION`中，原始actor forward和训练行为policy不同：forward为健康双动作；行为可加原Gaussian噪声及真实周期probe；target是兼容target actor加原TD3 smoothing。后两者属于off-policy数据/目标，不冒充冻结部署动作。两臂采用相同可审计probe：前150秒均匀支持周期，后10%周期probe；实际动作入replay。探针是训练覆盖，不算冻结策略健康。

Control A：同一完整V31初始化、fresh优化器和replay、5000次critic更新及每actor/周期128条后允许actor更新。Candidate B：完全同源，额外必要质量门槛及显式`--release-actor`。不提供此参数则整个有限单元只校准critic。首次运行只诊断冻结，不将其叫完整联合训练。轨迹会随更新分叉，不能说在线transition完全相同。

最后离线新增门槛是在部分运行停止后实现的，**不能回称这次早期运行已预注册/验证这些阈值**。它们是后续实验前需要锁定的配置，不能用已看到的结果宣称门槛有效。

240秒×24actor扣3600秒warmup，理论上最多8640完整critic更新；300秒理论最多14400，足够5000计数门槛但不保证覆盖/实际速度满足。未开闸就如实记录critic-only，不延长预算追逐结果。

冻结四场景：浅公交R20/.5BDP、高RTT公交、突降R100、混合时间尺度R20；60秒、2次、full/trim10均保留。每host一次只跑一流。保留每次departure/utilization、mean/P95/P99/drop、窗口与MTP、整数乘窗冲量代理、RTT freshness、历史跨度、calls及实际CPU。`v31_calibration_metrics.py`消费现有日志，不读TCP；控制区间与Mahimahi日志窗口来源分别标注。

前瞻工程筛查：相对同轮父V31，每场景利用率下降不超过2pp，mean排队不超过1.2×父+2ms，P95/P99不超过1.2×父+5ms，drop增加不超过.005；动态场景MTP SD≥5ms/window SD≥.02/单边界<.98。CPU改善必须伴质量不过度退化。以上是探索性非劣容限，不是用两个历史repeat即可标定的论文统计门槛；正式门槛须结合更多独立重复与预先设定的效应量。

直接停止/拒绝：冻结不保真、非有限值、持续Q漂移、开闸后动作边界退化、弱场景恶化、尾部/丢包失守，或不存在网络质量—实测CPU改善。留出自洽不能验证真实周期排序时保持不开闸，不追加长训。若完整兼容+校准仍复现V35退化，拒绝“校准是主要瓶颈”的解释，转向窗口能力与可观测奖励。

## 已执行与未执行

| 检查 | 状态 | 边界 |
|---|---|---|
| 最终Python数值/兼容/门槛/SMDP/物理额度/动作映射18测试 | PASS | 本机，无TCP/TF模拟宣称 |
| 最终源码编译与diff检查 | PASS | 静态，不证明运行正确 |
| 早期真实TF1完整67张量、critic更新、actor/BN不漂移、时间非恒定、保存重载 | PASS | 早期SHA版本，synthetic真实TF，不是最终gate |
| 早期24actor×35秒、零优化容量 | PASS | 有两次启动校验失败记录；未掩盖 |
| 部分actor冻结校准 | USER_STOPPED | 69640经验，6168.791172物理秒；每actor250.82—262.38秒；10275critic/20550critic操作/0actor/10275target，额度消化 |
| actor冻结直接变量核对 | PASS | 早期实际运行版本，不只是计数为0 |
| 第二臂、正式冻结/新CPU性能 | NOT_RUN | 停止后未启动 |
| 最终gate/新schema拒绝的真实TF图与开闸集成 | BLOCKED_BY_ENVIRONMENT | 本机无TF1；没有继续连接云端运行 |
| 最终代码1actor/24actor新协议和多seed盲测 | NOT_RUN | 后续需明确授权 |

部分校准的留出RMSE Q1在1000/10000迭代约6.953/6.338，双Q gap约3.010/2.784。目标网络在变化，且无同期control，不能推断真实价值变准或网络性能改善。原V31仍为PARTIAL。

云端checkpoint：`frameworks/mtpjoint/checkpoints/v37-quality_gate300/step-69640`，仅作部分诊断；完整tensor/replay/原始日志仍在云端。只回传798785字节小包，SHA256`c2bda47a357d3a4b9300062aa5575fb3c409d4c15e59a25b13cfdd8080d5d9c7`。五机终检本轮活进程0、下一臂未启动、原模型哈希一致；不会继续调度。恢复不支持完整TCP/history/max_bw/actor随机现场。

## 后续授权后可执行命令（现在未执行）

在独立候选Linux包目录执行；先把最终分支部署到各机同一绝对目录，并从Release安装原runtime/kernel。不得把最终源码测试结果套在云端旧早期副本。`configs/my-cluster.json`用现有私网节点/私钥/known_hosts，禁止提交凭据。

```sh
runtime/venv/bin/python run.py doctor
runtime/venv/bin/python tools/verify_repository.py
runtime/venv/bin/python tools/test_v31_calibration.py
runtime/venv/bin/python tools/verify_v31_calibration.py --output /tmp/v31-compatible-tf.json
runtime/venv/bin/python tools/v31_calibration.py plan
# 只有以上通过且获得新运行授权，再进行单actor真实冻结校准smoke。
runtime/venv/bin/python tools/v31_calibration.py smoke --cluster configs/my-cluster.json --run-id v31-cal-smoke-UNIQUE
# A/B必须使用同一预先冻结配置，顺序执行，预留冻结评测。
runtime/venv/bin/python tools/v31_calibration.py train --arm control --cluster configs/my-cluster.json --seconds 300 --run-id v31-cal-control-UNIQUE
runtime/venv/bin/python tools/v31_calibration.py train --arm candidate --cluster configs/my-cluster.json --seconds 300 --run-id v31-cal-frozen-UNIQUE
# 明确决定允许联合更新时，在新的有限单元中添加--release-actor；不是绕过质量门槛。
# 对每个选定checkpoint和scenario0..3各跑2次；每host仅一流。
runtime/venv/bin/python tools/v31_calibration.py evaluate --checkpoint /ABS/step-N --scenario 0 --run-id v31-cal-eval-UNIQUE
runtime/venv/bin/python tools/v31_calibration_metrics.py --run /ABS/frameworks/mtpjoint/logs/v31-cal-eval-UNIQUE --output /tmp/v31-cal-eval.json
```

当B未允许开闸时它的actor与父模型相同；因此不能靠B的冻结性能声称“校准提高了部署性能”。应先评估必要质量，再明确允许下一受限联合更新单元；或预先选择`--release-actor`并保留失败退出。以上命令尚未在最终Linux副本执行认证。

## 后续论文和CPU验证（仅规划）

局部门槛通过才考虑至少3个独立训练seed（主结论尽量5个）、统一覆盖/物理预算下独立fixed20/40/80/120/160、原健康V31、边界规则、匹配纯学习、CUBIC/Reno、baseline和官方Orca。旧fixed谱系不齐只作开发参考，Aurora预训练协议/覆盖不匹配时单独标注。盲测应独立trace/相位/RTT/队列，报告逐场景置信区间和全部失败。

动态收益须在同一连接内、决策边界可见信息足以指导周期变化时成立；不可预测的睡眠期间突降无法保证RL提前响应。不能用未来trace oracle作为部署策略。创新需要证明CUBIC持续反馈与RL乘窗脉冲耦合、价值稳定性和实际CPU—网络质量Pareto的具体贡献，不能仅称第二时间动作或SMDP新颖。

`resource_cost.py`以监测窗口归属进程user+system差额计费，不重复加children_time；包含重挂接Mahimahi，不能叫纯NN CPU。单独测Python/TF、native/IPC/轮询、日志、Mahimahi及整机/ACK数据面，按相同吞吐质量比较。未来使用现有profile_stages、进程CPU和perf；当前未做新profile或日志/轮询优化，不混进算法候选。


## 2026-10-24：真实云端 Stage 0 与中心化 D0（最新状态）

用户开启五机并授权执行。独立现场为 `/home/ubuntu/orca-v38-stage0-20261024`；本机小证据在 `D:/Canopy/experiments/v38-stage0-20261024`，机器可读结论是 `results.json`。源代码基础为 `d4ecb571246652f41ae7c78939cc176390bce60c`，实际运行的是该基础加本节的边界审计 diff；每机部署 SHA 见 `deployment.json`。本节覆盖此前“最终 TF 检查尚未运行”的历史状态，但不把旧实验回溯标成新版本。

### 做了什么，保持了什么

- learner 完成真实 TF1 离线拟合及数值审计，四台 actor 各一流并行做 48 条40秒冻结运行：4条件×20/80/160ms×2重复×stock/audit。另有25秒单流协议 smoke。无探索、无 replay 入库或在线优化，全部48条完成、网络失败0，原始 full/trim10 都保存。
- 条件是恒定24Mbps/R20/5BDP、恒定24Mbps/R20/.5BDP、浅公交R20/.5BDP、公交R100/1BDP。强制周期是同权重干预，不是独立训练 fixed baseline。
- `interval_rate_v28.h` 新增 opt-in `ORCA_INTERVAL_AUDIT_V38=1`：只从原有同一次 TCP_INFO snapshot 解码 SACK/MSS/重传等，另记既有区间平均 RTT、sRTT、窗口和读取序号。`evaluate.py` 显式透传开关。原 ACK 字节率、奖励、状态、乘窗、时钟及 schema 均未改变。新 binary 在独立 `native-audit-v38`，原 binary 保留。
- 24条 audit 的读取计数均等于 sequence+1，未增加 TCP_INFO 读取或中间观测。边界奖励从原始字段重建最大误差4.38e-12。所有48条 policy hash 一致。
- stock/audit 是不同 TCP 连接，不能因为函数没改就声称网络轨迹完全相同。恒定浅队列20ms组存在吞吐和丢包差异；高RTT20ms组重复离散很大。二重复不足以证明测量不干扰性；本轮 A/B 用于机制诊断，不作性能非劣宣称。

### 最强的新证据：ACK 恢复尖峰污染连接内峰值

恒定24Mbps浅队列、强制20ms的两条 audit，累计 ACK 区间速率最大为55.25和54.73Mbps。超过28.8Mbps的84+75=159个区间全部同时出现 SACK scoreboard 下降，且都在前五个区间内出现过丢失标记。诊断近似 `bytes_acked + sacked*MSS` 的峰值降到25.75和25.76Mbps。80ms两条峰值仅23.57/23.56Mbps。

这直接支持：累计 ACK 在恢复后释放的字节集中到较短区间，并经 `max_bw` 运行最大值持续影响状态及奖励归一。原 V28 的字节差/实际时间公式仍然是准确的“累计确认进展率”；它不等于每个短区间内新到达接收端的交付率，更不等于容量。不能再把吞吐区间一致性检查通过等同于峰值归一已经不受周期影响。

不能直接部署这个 SACK 近似修正：本轮没观察到负差分，不证明在 reneging、MSS变化等情况下它就是正确单调 delivered counter。正式改变速率/归一化前，须确定内核 delivered 计数语义、ABI、schema和模型兼容边界；不得给旧 V31 偷换输入。

### RTT、量化与 CPU

区间平均 RTT 确实已经可得且未进原效用。改用它的反事实效用密度差在恒定浅队列160ms约+0.0345、浅公交160ms约+0.0137；公交R100/20ms约-0.00013。该差异并非所有场景的大主因。短周期公交存在无新 RTT 样本的区间；不能把 avg=0 当成零延迟。ACK加权均值不是物理时间效用积分，更不是P95/P99，不能宣称替换它就解决尾延迟。

窗口 floor 与4包下限的实现保持原样，按 cwnd 分箱保留 requested/realized log-impulse。低窗口时下限有时会使偏差向正方向；“近中性负动作舍入是所有崩溃主因”没有得到本轮证实。

四台 actor 的内核 ring 均确认持续 `natcp_update_by_app` 日志；ring 只保留最后约4300条，少于约1万次成功写窗，不能用 ring 总数冒充逐次完整关联。内核/日志服务配置没改。40秒运行不足原预定 CPU 监测端点，因此 CPU 明确是 NOT_MEASURED_MATCHED_ENDPOINTS；没有新 CPU 获益结论。

### D0：已执行但未获准用于开闸

V35 中间 tensor/replay 在10月18日授权收敛整理中已退役。五机原路径只有退役标记，压缩历史保留源码/配置/证据但没有 tensor/replay，不能重建 checkpoint 轨迹。

本次使用现存120条独立40秒连接的 factual NPZ；phase0训练8135行，phase1留出7780行，6个明确 scene label。每行保留至少20秒真实未来及固定健康源 critic bootstrap 尾项；不再是4秒无尾项目标。20秒尾权重约0.0067，仍是代理参照，不是精确真实Q：目标平滑策略与确定性延续有差别，重叠窗口也不独立。

所有臂固定健康actor/BN、fresh随机双critic/target及fresh optimizer，使用原V31共享Adam；target actor保持健康函数，时间forward始终有效。数据仅有一次随机时间干预后健康延续，不是V35的24条件IID校准数据，优化器也不是V35修正版本，**不是V35复跑**。

训练密度基线只由 phase0、按真实 duration 加权估计，环境键来自scene而非actor；global b=0.61271。三臂各5000完整critic更新/10000优化器操作，actor更新0：

| 离线臂 | 留出逐场景等权RMSE（原奖励单位） | 窗口actor梯度与源critic余弦 |
|---|---:|---:|
| raw，冷critic | 85.76 | -0.580 |
| raw，加全局输出偏置先验 | 40.78 | -0.588 |
| 按环境中心化，冷critic | 36.46 | -0.417 |

这些数值仅是描述结果，不作为胜负结论。中心化没有全面校准窗口梯度，环境内仍有较大偏差；绝对RMSE下降不证明周期排序正确或TCP性能提高。

数值等价检查：单步TD最大差1.19e-5、梯度2.86e-6，通过预设2e-4/1e-3门槛。但100步原图Q差0.4967，超出预设0.02，**FAIL**。三臂在原执行脚本中仍完成，必须如实保留且判为无在线放行资格。

后续 A/A 诊断发现原奖励、同初值/数据/随机数重放的第一步Q也可差0.4967；共享优化器和重叠target操作原本在同一次session.run中无顺序约束。将原共享Adam两个操作及target操作显式串行后，原奖励A/A在1/10/100步完全一致。该串行只是诊断，改变了执行语义，未静默并入候选。串行下中心化差为首步7.39e-6、10步8.11e-5、100步0.04667，最后仍未通过0.02；不事后放宽门槛。可以确认原图重复性问题影响对照，不能把所有100步差异归为它，也没有证明中心化方案成功。

首次D0因标量assign_sub与[1]偏置形状不符在优化前失败，原stdout保留；只改为同形状张量后新stdout重跑。没有重跑TCP或更换seed。

### 代码与验证

`reward_centering_audit_v38.py` 只实现离线密度中心化，不安装进训练链。`tools/v31_centering_d0.py` 将实际执行脚本整理为显式路径CLI，默认在registered null失败后拒绝继续三臂；`--allow-failed-null-descriptive-screen` 只允许描述性拟合，永远不放行actor；`--stability-only` 复现A/A诊断。拒绝覆盖已有输出。实际运行的原始脚本及失败stdout另保存在实验目录，SHA清单区分两者。

- PASS：本机30项单元测试、Python解析/CLI检查。
- PASS：五机新native编译、25秒真实TCP smoke、48次冻结协议/权重检查。
- PASS：真实TF1恢复67张量，actor/critic/target查询最大误差0；冻结actor/BN/slots逐变量不变、critic确实更新；两个head梯度存在；临时checkpoint保存重载精确。synthetic时间输出SD=0.1382，不冒充TCP动作健康证据。
- FAIL：D0 registered100-step null；串行诊断的100-step中心化null也未过同一门槛。
- BLOCKED_BY_RETIRED_ARTIFACTS：V35中间checkpoint序列分析。
- NOT_RUN：新策略在线联合训练、匹配CPU收益、多seed盲测。没有把任何新模型替换健康V31。

在已独立准备好runtime、模型软链接及源码的全新实验目录执行：

```sh
/home/ubuntu/venv/bin/python -B tools/v31_centering_d0.py --experiment-root /ABS/NEW_ROOT --factual-root /home/ubuntu/orca-v37-oneshot-20261020-followup
/home/ubuntu/venv/bin/python -B tools/v31_centering_d0.py --experiment-root /ABS/NEW_ROOT --factual-root /home/ubuntu/orca-v37-oneshot-20261020-followup --stability-only
```

### Go / No-Go

当前 **NO-GO 大规模联合训练**。先明确ACK/delivered/峰值归一的计量候选及兼容schema；同时把优化器执行可重复性作为独立对照修正，再通过新的预注册数值null。不要把遥测、RTT奖励、中心化、探索、teacher正则等一次合并。当前证据提高了“计量与价值冷启动均有问题”的可信度，没有证明H1是唯一瓶颈，更没有证明新的双动作策略改善了性能—CPU Pareto。

五机终检本轮活进程/共享内存0，各机51个受保护模型文件SHA全部一致。服务器仍开机，未执行关机。完整新原始日志留各actor；旧factual NPZ与模型留learner；本机只收小JSON/配置/脚本/失败记录，没有传输大模型或原始逐包日志。


## 2026-10-24后续：独立Adam与窗口事实干预（最新）

独立云端 `/home/ubuntu/orca-v38-followup-20261024`，本机 `D:/Canopy/experiments/v38-followup-20261024`，仓库小结论 `evidence/v38-followup/results.json`。实际执行基础902520b加各角色SHA记录，最终归档commit不能冒称运行入口。先完成离线验证，再40秒joint-probe smoke及72条40秒冻结连接；没有在线训练actor。

### 已修正的工程问题

复用已有 `independent_critic_v34.py`，通过显式 `independent_critic_adam_v34=true` 在 `distributed.setup()` 构建前安装。默认关闭；额外处理同一进程随后切回legacy图的开关，禁用时保留原公式。两个critic参数范围不重叠，各自Adam时钟前进一次，完成后总优化操作计数加2。原reward/actor/target更新公式与架构保持原样，旧V31模型身份不变。`compatible_calibration_v37.initialize()`现在明确记录实际优化器方案，仍然只迁移模型函数、fresh slots/counters。

在同一120条旧factual数据、相同seed/批次/原target操作上，独立Adam的raw A/A在1、10、100步Q差均为0；中心化100步误差0.0161664，通过原0.02门槛，不改阈值。三臂各5000critic/10000操作，actor0：原奖励/全局偏置先验/按环境中心化的留出RMSE为85.50/40.63/36.43。为新事实查询保存critic-only张量而同seed重跑，三臂最终留出Q网格逐位一致。critic-only NPZ留learner，不是新策略checkpoint或resume包。

9组真实TF1测试覆盖1/2线程、MSE/Huber、PER与alternate入口的100步Adam递推、独立slots/powers、计数及保存后下一步；实际149维Agent完整恢复67张量并验证actor/target不触碰critic优化器。这证明修正的工程必要性，不证明网络性能提高；此前V34性能失败事实仍保留。

### 滑窗归一化只部分有效，没有部署

`boundary_rate_audit_v38.py`及`tools/v31_boundary_rate_audit.py`仅离线读取原有累计ACK边界，在至少max(160ms,当前sRTT)真实跨度上求速率并维护成熟窗口峰值，不插值或额外观测。浅24Mbps/20ms峰值约55Mbps被压到26.5—26.7Mbps，但24Mbps深队列/160ms的一次30.03Mbps峰值仍保留。该方法仍依赖观测稀疏程度和历史容量，启动前完整窗口还未定义部署行为，因此拒绝直接接入训练。

核对实际4.13源码及[上游TCP_INFO定义](https://github.com/torvalds/linux/blob/v4.13/include/uapi/linux/tcp.h)：它公开bytes_acked、delivery-rate样本，没有公开累计delivered字段；内部tp->delivered是包计数，并涉及SACK/DSACK处理。不能把它乘当前MSS就称为已验证的唯一交付字节计数。本轮没有改/安装/重启内核，没有把旧149维模型偷换计量schema。

### 72条窗口事实干预：没有通过价值指导门槛

四个开发条件：浅公交R20、高RTT公交、突降R100、混合R20；3个MTP×3个窗口delta×2独立连接重复，共72条，网络失败0。delta为-0.1/0/+0.1，只影响第8物理秒后一个动作；窗口仍用原乘窗映射，之后所有动作恢复健康V31。MTP仍同时决定观察和执行，无隐藏20ms观测。组合在scene×repeat块内预先随机排列，seed731224；未使用新训练seed、盲测或CPU比较。

`Probe`新增opt-in `window_action_delta_v38`，默认0保持旧时间探针语义；非法/非有限或大于0.1的扰动拒绝。执行的动作、裁剪后的有效delta、原窗口输出和随机机制均记录。72条均验证transition连续、物理discount、native整数窗口映射及MTP指令；48个非零扰动没有被native整数舍入完全吞掉。非探针动作与父策略前向误差不超过预设2e-5，policy/BN哈希不变。

初次后处理在一条run的batch前向误差2.63e-5处停止；保留r1失败。对5个超限输入使用原运行batch-size1前向复核后通过同一门槛，72条验证最大误差1.9919e-5。只重做actor3离线分析，复用其他三机结果，没有放宽阈值、修改数据或重跑TCP。

主要终点是条件于MTP的窗口优势：同输入三窗口Q网格与原奖励20秒事实回报（含小的固定源Q尾项），在24个period-triplet内中心化、按8个scene/repeat块等权。source min(Q1,Q2)关联-0.0345，块bootstrap95%区间[-0.4000,0.3193]，随机化p=.855。三种离线critic的p为.759/.582/.550，区间也全跨零。实际actor使用Q1，因此另外保留Q1/Q2次要分析；它们同样没有可靠正关联，不挑选其中有利的一支。

**这不是“critic方向已证错”，也不是“窗口控制无效”。** 两次TCP重复的噪声较大：零窗口delta的A/A残差差值绝对值中位数3.21，正负扰动对比为5.69；二者只是描述量，不是正式功效/等价性结论。20秒仍是主终点，1/4/8秒及对齐的利用率、平均/P95/P99排队、丢包全部保留，不改挑更短窗口。该实验检查窗口价值方向，不能推翻前一轮关于时间动作的独立关联证据。

### 当前决定与复现

中心化数值null已过，但真实价值指导门槛仍未过，actor保持冻结，**NO-GO大规模联合训练或论文收益宣称**。独立Adam是训练工程修正；滑窗不是完整计量修复。下一步先验证原边界可读的SACK-aware交付统计及其schema，再预注册有限配对实验；不同时加入新奖励、探索、网络层或运行时teacher。

```sh
# 原框架默认行为仍是legacy；只在新研究运行显式配置独立Adam。
/home/ubuntu/venv/bin/python -B tools/verify_v31_optimizer.py --output /ABS/NEW-optimizer-check.json
# 独立准备新root，不重复启动历史pipeline或覆盖已存在结果。
/home/ubuntu/venv/bin/python -B tools/v31_centering_d0.py --experiment-root /ABS/NEW_ROOT --factual-root /home/ubuntu/orca-v37-oneshot-20261020-followup --optimizer independent_v34
```

39项本机测试PASS；本机优化器TF测试正确标为BLOCKED_BY_ENVIRONMENT，云端真实TF PASS。五机终检本轮进程/IPC0，各机51保护模型SHA通过。新NPZ/逐包日志留各actor，critic-only张量留learner，仅小JSON、脚本、配置、失败证据回传。服务器仍开机，没有后台后续任务；public main未更新研究分支。
