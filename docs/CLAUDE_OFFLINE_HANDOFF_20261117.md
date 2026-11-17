# Claude离线实施交接：连续双动作，未通过全部门槛

用户最新指示：“让claude去做离线吧，你分配交接完后就停止，等Claude运行完我会复制给你”。本交接后Codex停止实施；由Claude独立复核和继续离线工作。**没有云端授权，本轮禁止SSH/TCP/云端训练。保持连续时间头，不改成离散选择器。**

## 工作位置与边界

- GitHub工作分支：`innovation/joint-continuous-conditioned-v4`，父提交`ca2c4bf9031994e82e515bad8701efc7feaf8d43`。本次提交是WIP交接，不能当成已验收修复。先读本文件、`docs/JOINT_CONTINUOUS_REPAIR_PLAN_20261116.md`和`docs/JOINT_COST_PRESERVE_REPAIR_20261116.md`，再检查实际代码。
- 本机工作树：`D:/Canopy/innovations/joint-cwnd-mtp-continuous-v4`；本机结果：`D:/Canopy/experiments/joint-continuous-offline-20261117`。原成本分支在`D:/Canopy/innovations/joint-cwnd-mtp-cost-v2`，未改。
- 原健康基座仍是main/c67717de585692bf6a79d1c0545ace92e66347d2，模型`models/healthy-baseline-seed3/training/step-6013196`。不得改原代码、模型、native、trace、内核、ssthresh、窗口范围，不删除负面证据。不训独立固定MTP组。
- 本机TF1.14/Python3.6解释器：`D:/Canopy/experiments/joint-healthy-offline-20261113/python36/python.exe`。系统python没有TF。若你在网页沙盒而非本机，先说明实际可用工具；不能把浏览代码当成已执行本机TF测试。
- 现有真实输入在`evidence/joint-cost-repair-20261116/real-frozen-probes.npz`。175冻结输入、512近期回放状态及960分层真实SMDP还在`D:/Canopy/experiments/joint-cost-repair-20261116-075300`，不必开云下载。960不是MC连续轨迹，不能当真实反事实Q。

## 背景，避免重复走旧路

旧成本.01第三短模型网络性能过门、调用只降0.59%，没有晋升；MTP约20.2ms。不是导出错，也不是完全没有长周期经验，不能简单归因集群太小/600秒注定不够/唯一reward bug。成本0的旧3.47M长训配方不同，不能推导当前必然失败。

实际SMDP仍为`D=.995^(actual_dt/.02)`、`R=u*(1-D)/.005-.01`。重连物理边界保留在replay用于序列验证，但`bootstrap_last_observation`下不是Bellman终点。效用是端点测量近似，不是已证明无偏的全区间积分。原Orca奖励没有Canopy性质约束。

常值效用、只改当前一次等待且后继回20ms时，.01成本的40/80/160ms优势只有.00995/.029701/.068614；不能混作永久固定周期价值。旧相对TD门不认证这么细的排序；TD RMSE大于优势也不证明排序错误。

## 已实施的候选代码（请独立审阅）

1. 新模块`frameworks/joint_healthy/continuous.py`，opt-in架构`continuous_conditioned_v4`，新schema，旧profile默认保留。
2. 连续物理周期`T=20+140*sigmoid(z)`，转换回旧协议连续log动作。fresh20.5ms/empty clip(minRTT,40,120)初始函数保留；时间特征unit-L2。sigmoid仍可饱和，未消除边界风险。
3. 先选本次T，再算`window=tanh(base(s)+log(T/20)*residual(s,log(T/20)))`。条件支路最后层零初始化；T=20结构上回到base。base和条件残差都可学，原BN统计冻结。
4. 行为图复用在线主干，一次session调用中先对z做局部连续噪声或全范围分层连续采样，再按实际执行T计算窗口，最后才加窗口噪声。五个coverage桶只是审计；不是五档策略。target同样先选T再条件窗口，保留原窗口target smoothing .1/clip.2，时间不平滑。
5. learner-only teacher只约束base20窗口支路，时间参数不接teacher梯度。窗口LR1e-5、时间候选LR3e-4、teacher权重10；两Adam梯度通过屏障读取同一更新前策略。成本.01、实际dt奖励/折扣和native未改。
6. critic新增零初始化连续物理周期残差，输入含归一化state、窗口、线性T/T²、交互项。原窗口Q主干的第二动作输入固定-1，时间依赖走物理周期支路，避免两种时间坐标同时竞争。初始输出仍是迁移Q。**不是Phi+A价值势修复，没有加入解析正确Q或容量真值。** 此结构为开发候选，尚未充分验证。
7. 训练过程中持续写绝对双TD误差及同状态条件窗口/保持窗口两种Q周期扫描，明确标`value_ordering_verified=False`。Q扫描不是反事实真值；原相对TD/短bootstrap门只作数值检查。
8. 新profile/template `continuous-v4`/`templates/joint-healthy/joint-continuous-v4.json`。旧G4准入不能放行新架构，需新源/native/config匹配的离线与真实机制/初始健康证据。没有创建虚假PASS receipt。CLI同权重周期干预会重算条件窗口。

## 已有结果，哪些能说，哪些不能说

公开小证据目录：`evidence/joint-continuous-offline-20261117`；完整stderr/stdout仅在本机H。所有优化均临时合成任务；没有TCP或新模型交付。

- `continuous-graph-final.json`：真实TF1图检查PASS。128输入，初始窗口误差0；初始周期误差1.1444e-5ms；探索后实际T重算窗口误差1.788e-7；故意设置非零条件残差后，旧的错配方式可产生0.2178窗口差。20ms结构恒等；校准actor/target/BN逐位冻结；校准actor更新被拒绝；两分支可更新；teacher时间梯度0；完整链式梯度误差1.49e-8；全CP恢复/独立推理导出误差0；拒绝旧schema。这个PASS不包含TCP健康或价值排序。
- 原63项加新6项共69项unittest曾PASS，**在后续critic/backend改动前，最终版本尚未重跑**。旧基座真实TF等价/全仓校验/更广测试也尚未重跑；不得报最终全部通过。
- `capability-10k.json`：**FAIL**。受控单步、同20ms后继、正/零/负优势任务，每任务10000完整双critic迭代、batch512。最小已知间隔.00995，预设MAE上限.0024875，排名需≥95%。新双Q：正MAE .0026833/.0026926；零 .0002696/.0002263；负 .0019662/.0026681。符号排名100%，幅度仍有项未过，不能放宽门槛称PASS。
- `legacy-critic-10k.json`：同预算旧critic参考**FAIL**；正约.009649/.009642，负约.009657/.009657。新支路在此任务改善，但没达到门，且单一模型seed，不是正式泛化证明。
- 连续actor受控任务用**解析已知Q，未接上训练得到的critic**。两个保存真实输入仅fresh标记设1；固定初始窗口参考，目标连续周期40/120ms，窗口目标也随状态改变。LR1e-4在5000步仅40.0004/101.4000；LR3e-4为39.9996/119.99995，窗口误差0。用于学习率候选，不是TCP收益/learned-critic闭环成功。早期`capability-first.json`的actor目标错误地随当前base漂移且可能超动作范围，已修正；该早期actor结果无效，不能用于结论。
- 其他5000步/3000步开发版本`capability-final/period-residual/physical-only`均保留FAIL。架构、batch、预算曾调整，不能混在一起做精确单变量因果比较。最终10k旧/新参考才有匹配预算。这里是开发调试，不冒称独立预注册验证。
- 新continuous learner FIFO fixture **FAIL/PARTIAL_CALIBRATION_ONLY**，见`pipeline-failure.json`与本机`continuous-pipeline.stderr`。500合成经验、201critic、actor0、phase0、debt0；不是网络失败。fixture每actor100步，但旋转holdout模257，有些actor可能没有完整8步holdout块，应先保留gate reasons定位，不要放宽真实门。临时目录已被TemporaryDirectory清除，需修可重复保留失败现场的工具后重跑。本机没有新正式训练。
- 用户要求交接时，已停止新增工作并核对没有本轮Python测试进程。10k JSON均已写出完整任务结果；不要把用户交接/进程收尾误称实验失败。**全部价值任务本来就FAIL，不能拿进程退出码替代科学结论。**

## 请Claude接手的具体任务

1. 先说明你实际读取、修改、执行了哪些文件/工具，能否访问本机。独立审阅我的结构/梯度/测试设计，不要接受摘要作真值，也不要为了过门换成离散时间头。
2. 复核critic受控任务：单步与固定后继语义、已知小优势量级、训练/查询覆盖、相对优势指标、双Q、误差来源。区分表达能力、公共价值校准、优化噪声/拟合形状。当前略超阈值不是数学不可学；也不能仅增加步数挑最好CP。固定预算保留终点，修测试缺陷与修算法分开。
3. 决定最小修复。可以比较更合适连续特征/优化尺度；可选状态势Phi+A只有代数、终止语义、保存恢复、真实已有回放的残差尺度/方差和能力对照支持时再采用。不要直接把成本加大、加推向长周期的伪奖励或隐性兜底。
4. 修FIFO fixture覆盖/失败现场保留，验证两阶段自然放行、两头更新、均衡采样、物理账本/排空、持续价值监控和live checkpoint都真实执行；保持合成阈值与正式阈值分离。
5. 最终统一源码后重跑69项及必要现有更广测试、真实TF图/迁移/BN/梯度/恢复/新旧schema、已知正负零优势及状态相关连续动作任务。需要补真实回放误差分层时用已有小数据。保留失败，禁止后改门槛包装成功。
6. 写清下一轮云端最短门：真实CUBIC阶段/cwnd/ssthresh/动作范围可行性、连续周期实际执行与奖励观测；再有预算的短可学习性准入和预登记冻结点，满足后才考虑正式长训。T≤2RTT只是持续slow start等条件近似，不能直接硬裁剪。最终调用降≥20%及原逐格吞吐/mean/P95门保持，不另训fixed组。
7. 交付分支/提交、文件清单、实际命令/结果/失败、可信限度、待云问题；按仓库工具重建inventory并校验保护文件。若你只有网页分析工具，请交付精确补丁和复算结果/未执行项，**不要说已在Windows本机改好或TF通过**。用户将把你的完整回复复制给Codex再复核；无需与Codex来回等待，也不要连接云端。

复现入口（TF1.14解释器执行）：`tools/verify_joint_continuous.py`、`tools/check_joint_continuous_capability.py`、`tools/verify_joint_healthy_pipeline.py`，均支持`--config`和`--output`；前两者还需`--states evidence/joint-cost-repair-20261116/real-frozen-probes.npz`。`--legacy-critic-reference`只生成旧critic能力参考，不训练独立fixedMTP TCP模型。
