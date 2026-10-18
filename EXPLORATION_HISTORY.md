# MTP探索记录：V01—V36

本记录保留试错依据与负面结果，供迁移训练和冻结评测时理解模型谱系。它不是模型排行榜，也不是论文复现声明。各轮场景、队列、启动、观测语义和重复数不同，不能用跨轮总均值挑“全局最好”。本次科学选型只依据既有证据；另进行了有限TCP打包验证，验证记录与性能研究证据分开，不能算新算法收益。

## 最终保留的三个角色

- **冻结baseline**：原70维单窗口模型，commit `ff5395a085d8940cc8c48c9c2b26262370176b9c`；原云根目录 `/home/ubuntu/orca-repair-20261010-114820`，CP `checkpoints/formal24-grid24-seed3-balanced-u010/step-2412694`。它是seed3、24缩减环境工程基线，完整旧执行链与正常CUBIC启动参照分别报告，不能混用。
- **固定20ms观测、动态执行周期路线：V14**。原云根目录 `/home/ubuntu/orca-adaptive-mtp-20261014-115233`，CP `checkpoints/v14-joint1800/step-277317`；实际训练commit `c43c72f9e8fbe1c9402560ed6ca33f85572dc2a8`。选择依据是修复时钟/IPC后的可用窗口与完整训练/冻结实现，不是所有条件双动作通过。训练未完成原1800秒计划，四个浅队列动态条件MTP近固定，仍为PARTIAL。
- **观测与执行共动态周期路线：V31**。原模型根目录 `/home/ubuntu/orca-feedback-context-20261016-1203`，CP `checkpoints/v31-feedback600/step-157704`，149维状态。后续恢复/辅助归档目录不能替代原模型来源。V34/V35/V36复评支持保留其双动作健康，但浅队列、高RTT与尾延迟仍有缺口，不能称全场景领先或论文PASS。

V08只作历史性能/阶段机制参照。其旧时钟遗漏和IPC风险到V13才修复；不允许静默换修复接口后沿用V08旧得分。固定20/40/80/120/160ms保留为性能强对照或同权重干预入口，具体身份由manifest标明；它们不算新增双动作主模型，也不是每个周期都在每轮独立训练过。

## 逐版动机、实现、结果与去留

下列来源根目录统一省略前缀 `/home/ubuntu/`；CP为该根目录下的相对路径。没有新模型的版本明确注明继承关系。来源指向历史现场，不要求新机器具备原本机D盘报告目录；是否已复制tensor以交付清单为准。

### V01

动机：联合窗口/周期。实现72维双动作、20ms观测、物理信用；修复启动过量放窗。167k样本后利用率54.70%，高RTT低窗；短训练不能判永久坍塌，评测42/48。

来源：`orca-joint-mtp-dev-20261010；CP checkpoints/joint-physicalcredit008-seed3-900/step-167029。`

### V02

动机：target BN批上下文失配。改moving统计初始化与tau跟随；随机同初始900秒/actor。24条件三重复利用率94.90%，调用少82.2%，排队高26.9%；PARTIAL，单seed。

来源：`orca-joint-repair-20261010-230437；CP checkpoints/targetbn-joint-seed3-900/step-207595。`

### V03

动机：公平常数周期消融。同初始/物理信用joint对独训120ms，各900秒。joint81.93%/11.74ms，对照90.53%/10.59ms，joint调用更多；周期与减窗耦合，未证自适应收益。

来源：`orca-joint-ablation-20261011-110849；CP checkpoints/paired-fixed120-900/step-167663；checkpoints/paired-joint900/step-197721。`

### V04

动机：显式调用成本。原效用积分单次扣.02，对零成本同初始900秒。成本组90.24%/14.95ms，对scalar少85.39%调用；工程PASS，弱于fixed120折中，时间上界84.95%，研究PARTIAL。

来源：`orca-joint-cost-20261011-211954；CP checkpoints/cost002-900/step-156711；checkpoints/cost-zero900/step-175276。`

### V05

动机：隔离周期学习。冻结fixed120窗口teacher，新时间头/critic，600秒。93.21%/11.65ms/7.57调用，P95比20ms高37.8%；仅一次全网格，窗口未学习，不满足最终双动作。

来源：`orca-staged-mtp-20261012-1950；CP checkpoints/joint-teacher-time600/step-112998。`

### V06

动机：先保动态窗口能力。继承baseline窗口/Q，固定20ms微调900秒。动态24条件97.51%/15.20ms，mean/P95改善、P99升；仅一重复，时间梯度0，不能称双动作成功。

来源：`orca-dynamic-mtp-20261012-215502；CP checkpoints/v6-dynamic-fast20-900/step-958807。`

### V07

动机：去除周期恒定输入外推，双动作联合微调。V6先验折入bias、5000critic校准，900秒。89.89%/14.09ms，未过性能；初始多周期被FIFO淘汰，动态头仍有变化，覆盖缺口非唯一因果。

来源：`orca-period-mtp-20261013-101317；CP checkpoints/v7-joint900/step-284976。`

### V08

动机：持续周期覆盖。FIFO加actor×周期reservoir，真实bank抽样25%，900秒。两重复96.26%/15.23ms，动态无单tick异常，调用少70.5%；P95高，后发现旧时钟/IPC缺陷，保留历史参照。

来源：`orca-coverage-mtp-20261013-114720；CP checkpoints/v8-period-bank900/step-281887。`

### V09

动机：改善尾部/新鲜度。近期ring与显式延迟惩罚分别训练，另独训fixed80。近期92.70%，惩罚78.34%，均失败；fixed80强，V8保留。432评测，动态局部近固定，单seed。

来源：`orca-tail-mtp-20261013-144741；CP checkpoints/v9-recent900/step-295961。`

### V10

动机：定位MTP阶段价值。V8冻结四策略、16条件两重复，无训练。部分高RTT恢复段少调用/排队，整体不胜独训80；纠正phase标签6秒错位，局部相关不能当调度因果。

来源：`orca-mtp-phase-20261013-182050；无新训练CP，沿用V08及V09独立fixed80。`

### V11

动机：突降首秒局部覆盖与新时间尺度。冻结V8、边界40/80ms，84run。首冲下降但吞吐损失/后段反弹；高RTT局部折中，整体未省CPU；无新模型，不是提前打断interval。

来源：`orca-mtp-regime-20261013-210043；无新训练CP，沿用V08及V09独立fixed80。`

### V12

动机：真实LTE/快慢混合覆盖筛查。V8六方法八条件两重复，100run，无训练。覆盖缺口、短周期反复减窗；ATT近长周期。发现native漏计发布等待，整体最大约.205%，须先修时钟。

来源：`orca-trace-screen-20261013-215618；无新训练CP，沿用V08及V09独立fixed80。`

### V13

动机：修时钟/IPC并配对训练。连续采样端点、稳定快照；新覆盖joint/fixed80同先验900秒。87.72%对94.75%，六动态run低SD；训练完整但性能/健康失败，接口修复不能回溯验证旧模型。

来源：`orca-clock-mtp-20261014-101009；CP checkpoints/v13-fixed80-paired900/step-248242；checkpoints/v13-joint900/step-242876。`

### V14

动机：恢复窗口并补浅队列/噪声/NYC覆盖。V13fixed80先验、180秒重连；timeout使1800秒计划实测约1434—1452秒。24条件93.02—99.98%，四浅队列R20近固定；无同覆盖完整固定对照。

来源：`orca-adaptive-mtp-20261014-115233；CP checkpoints/v14-joint1800/step-277317。`

### V15

动机：补深度覆盖。16轮换角色0.5/1/5BDP、56配对，900秒。8条件仅4/8健康，高RTT时间tanh饱和；不启固定训练/正式比较。Q低估非统一原因，短周期新鲜度不足待证。

来源：`orca-depth-mtp-20261014-140508；CP checkpoints/v15-joint900/step-185779。`

### V16

动机：避免时间tanh数值失活。仅时间头改softsign，其他保留，900秒。8筛查96.29%但仅4/8健康，高RTT SD约.7—1.3ms；未精确饱和不等于有效自适应，正式对照未启动。

来源：`orca-softsign-mtp-20261014-144258；CP checkpoints/v16-joint900/step-178114。`

### V17

动机：补本地时间尺度。增绝对RTT/吞吐EWMA残差为76维，joint/fixed80同先验900秒。56条件51个≥90%，216正式评测；72自适应run中37低SD、含26动态，无全局边界坍塌仍PARTIAL。

来源：`orca-context-mtp-20261014-151515；CP checkpoints/v17-joint900/step-187124；checkpoints/v17-fixed80-900/step-249040。`

### V18

动机：解除校准周期固定循环关联。私有IID周期RNG替循环，900秒；25组合真实覆盖。8筛查仅4/8健康，高RTT近恒定长周期，正式对照未启动；关联修复非充分根因。

来源：`orca-randomcal-mtp-20261014-191500；CP checkpoints/v18-iid900/step-178493。`

### V19

动机：检验等待中可见压力提前唤醒。冻结V17，双帧RTT/丢包规则，8条件四方法两重复。68run无失败，预定动态门槛0通过，无训练；规则机制探针不是学习调度。

来源：`orca-preempt-mtp-20261014-204500；冻结V17，无新训练CP。`

### V20

动机：减少观测开销。借V17fixed80旧权重改80/160ms观测/动作，68run、无训练；修无效RTT计数丢失。历史/统计语义变了，旧权重探针不作新路线，用户撤销工作副本，模型保留。

来源：`orca-experiment-archives（失败探针惰性归档）；沿用V17fixed80 step-249040，无新训练CP。`

### V21

改为86维真观测/执行共周期；随机先验后fresh重训。56条件95.718%/19.986ms，核心8工程门槛过，动态动作健康；单seed谱系、已见前缀验证，未证优于匹配常数周期。

来源：`orca-decision-mtp-20261014-224600；CP checkpoints/v21-refresh900-r3/step-267711；checkpoints/v21-joint1800/step-220725。`

### V22

补真正固定80ms同观测/执行强对照；同86维先验、900秒/actor。固定96.470%/22.863ms，对V21调用多12.69%；56已见条件有限比较，不能混用旧20ms观测fixed80。

来源：`orca-decision-fixed80-20261015-104623；CP checkpoints/v22-fixed80-900/step-260545。`

### V23

72次同权重/固定/等调用诊断：周期作用依场景，突降等调用自适应87.77%对70.76%；窗口恢复仍弱，换周期不足。准备固定先验转联合，但尚未训练；非普遍因果优势。

来源：`orca-clock-budget-diagnostic-20261015-1220；无新训练CP，沿用V21/V22。`

### V24

同预算continued对fixed80先验转联合，各600秒；48冻结0失败。continued高吞吐伴更高尾延迟，fixedprior高RTT严重退步；两组各17/18动作门槛，保留V21，不算全条件通过。

来源：`orca-curriculum-pair-20261015-131707；CP checkpoints/v24-continued-joint600-r2/step-149818；checkpoints/v24-fixedprior-joint600-r2/step-160538。`

### V25

同clock窗口/周期2×2冻结诊断，16正式+4协议，0训练。弱模型换80ms未恢复源窗口能力；自峰值分母使低吞吐奖励更高，离线统一分母排序反转。双网络仅诊断，非部署收益。

来源：`orca-factorial-v25-20261015-153708；无新训练CP，沿用V22/V24。`

### V26

原奖励对冻结训练参考分母，各600秒/actor、36冻结。突降利用率改善但排队升，常规恒定仍低，修复17/18动作门槛；奖励尺度有影响，非唯一根因或全面修复，不替换正常模型。

来源：`orca-reward-reference-20261015-181710；CP checkpoints/v26-reference600/step-159151；checkpoints/v26-original600/step-155428。`

### V27

86→96追加真实间隔RTT样本率，masked配对各900秒。突降mean下降70.42%但吞吐低2.43pp；浅队列弱，动作门槛新7/18、masked10/18，近常数长周期仍在，否定全面通过。

来源：`orca-time-statistics-20261015-192250；CP checkpoints/v27-rates900/step-214140；checkpoints/v27-masked900/step-202582。`

### V28

改累计bytes_acked/真实dt计量并加schema禁旧CP混搭；两组900秒。突降新97.43%/15.42ms，浅队列76.92%弱；动态15/15动作过、总17/18，计量修正不等于总体成功。

来源：`orca-interval-accounting-20261015-205000；CP checkpoints/v28-acked900/step-283381；checkpoints/v28-samplemean900/step-213655。`

### V29

隔离时间梯度进入共享trunk，有限单变量900秒配对。候选仅4/12动作门槛且尾延迟严重退步，拒绝并保留V28。失败不证明所有历史问题由梯度冲突导致。

来源：`orca-period-repair-20261016-101304；配对900秒失败模型，精确CP见运行manifest。`

### V30

修定时边界被fresh RTT门控：初始化后按计划发布真实零新RTT快照。7个C++用例及6×35秒接口探测通过，1862零新RTT样本、0后续阻塞；未训练，不作性能/CPU收益结论。

来源：`orca-timer-boundary-20261016-1104；协议修复，无新训练CP。`

### V31

96→149加执行动作/RTT新鲜度界/TCP反馈，定时边界；五臂600秒及冻结诊断。feedback动作健康，网络仍依场景、浅队列高RTT弱；保留step157704供诊断，整体研究门槛未过。

来源：`orca-feedback-context-20261016-1203；CP checkpoints/v31-feedback600/step-157704。恢复归档另在orca-feedback-context-r2-20261016-1527。`

### V32

按sqrt(MTP/80ms)缩放训练窗口探索噪声，尝试保持单位物理时间脉冲扩散；仅本机数值/随机流兼容测试，部署前暂停，未云部署/训练，无新模型或收益。

来源：`离线候选physical-window-noise-v32；未部署，无云端训练CP，继承V31。`

### V33

固定3,000,000 bytes/s奖励尺度候选；V31 actor/BN迁移与TF校验通过，优化0/TCP0，临时合成CP弃用。只完成离线可行性，不是训练模型或网络收益。

来源：`orca-reward-calibration-v33-20261016-r2；仅TF验证，临时合成CP弃用，无训练模型。`

### V34

修双critic共享Adam时钟，配对legacy/independent及固定对照；240+48冻结0失败。动作24/24旧V31、22/24legacy、17/24新，候选总门槛失败；数学修复有效不等于性能成功。

来源：`orca-v31-repair-scenarios-20261016-2136：checkpoints/v34-legacy600/step-156888；orca-v31-recovery-v34-20261017-1300：checkpoints/v34-independent600-r2/step-146948。`

### V35

同健康V31 actor、fresh随机critic与独立Adam，原/固定尺度奖励各600秒；90冻结0失败。两臂动作0/30、近160ms，V31父30/30；奖励排序修正未阻止退化，冷启动机制仍待证。

来源：`orca-reward-pair-v35-20261017-1950；CP checkpoints/v35-original600/step-122221与checkpoints/v35-candidate600/step-122249。`

### V36

仅冻结因果缩窄：4场景×8变体×2=64正式，另8协议，0训练/优化。时钟/字节/执行审计过，启动非主解释，短周期可失稳，收益依场景；强制周期不是独立训练模型，唯一原因未证。

来源：`orca-causal-diagnosis-v36-20261017-2130-r2；冻结V31、baseline及官方Orca，无新训练CP。`

## 已修缺陷与不能回溯的结论

**target BN**：V02确认Sonnet默认local统计使target依赖回放batch，而部署使用moving统计；修复显式moving、初始复制、tau跟随。真实TF和固定查询验证支持这一缺陷，但单seed改善不能证明全部历史退化只有它一个原因。

**时钟、IPC与吞吐计量**：V12发现旧native发布后另取起点，处理/等待未进入下一delta_t；V13改连续采样端点，并修新stateID可能配旧时长的共享内存快照。SMDP仍按观察边界归账，sample到实际写窗非零延迟继续保留。V20发现读取即清零的TCP计数在无有效RTT时可能丢失，该修复没有自动移植旧版本。V28以累计bytes_acked差/真实区间dt修正动态观测的吞吐计量，增加schema阻止旧CP混搭；V30让定时边界不再被fresh RTT门控。旧sample-mean、ACKed区间速率和不同历史跨度不是同一输入语义，维数相同也不能互载。

**奖励与价值**：观察峰值max_bw不是trace容量。低吞吐可能仍取得较高归一效用，V25/V26/V33/V35分别审查排序/参考尺度。统一奖励分母修正某些排序，不等于保证吞吐或解决周期学习。V34修复双critic共享Adam时钟；数学修复没有自动变成性能收益，旧健康V31仍保留原始实现和模型，不静默改成新优化器后声称同一次训练。

**V35失败具有配对意义**：两臂都从健康V31 actor/BN出发，critic/target、独立Adam、replay与计数fresh初始化，仅奖励分母不同，各600秒/actor。父模型动作门槛30/30，两臂均0/30、接近160ms；90冻结run无网络失败。原奖励与固定尺度奖励都退化，排除“改分母即可修复”的充分解释。最终候选critic并非统一低估，有限尾段回报也不等于无限期望Q。价值冷启动/target传播是待验证机制，尚未证明唯一根因。

## V36：冻结诊断缩窄了什么

依据V36最终`v36-analysis.json`及版本清单，完成四个开发场景×八变体×两重复64正式run，另8协议；训练与优化均0，学习策略8/8动作健康。时钟、区间字节和native执行审计通过，没有发现执行链路失败；改变scalar启动不是主要差距解释。官方Orca模型来源为`/home/ubuntu/orca-canopy-pretrained-eval-20261008-1148/models/orca/model.ckpt-1283529`，与本地训练baseline不是同一模型。

- NYC bus/R20/0.5BDP：V31约76.86%利用率；强制80/160ms约76.48%/76.05%，不能仅换MTP救回浅队列窗口能力。
- NYC bus/R100：强制20ms降到22.49%，自适应87.04%、160ms为93.39%；短周期可使闭环失稳，低排队伴严重低吞吐不能叫收益。
- compound_drop24/R100/5BDP：自适应98.83%/31.49ms；80ms为97.99%/18.69ms，160ms为99.29%/39.71ms。自适应相对160ms有延迟折中，却非优于80ms。
- mixed_timescale48/R20/5BDP：自适应98.85%/9.68ms，对80ms的99.13%/17.13ms排队更低，但调用18.56/s对12.24/s。收益依场景与指标，不能宣称统一省开销。

同权重强制周期改变整个未来状态、窗口输出及累积减窗冲量，不是保持窗口轨迹不变的纯频率试验，更不是独立训练固定模型。V36没有证明唯一根因，也没有生成新策略；保留V31，不把诊断当修复成功。

## 迁移与证据边界

模型/target/BN/优化器、replay、bank、计数及部分RNG是否保存，逐版本看manifest。多数新版本重建replay/优化器或critic，只继承actor/窗口先验；它们是新环境微调或重训，不是无缝resume。TCP连接、队列、history/max_bw、actor探索及TF随机执行现场缺少完整连续恢复加载器；载权重重建连接不能称连续恢复。

训练信用来自成功入库经验携带的真实物理暴露；bank副本、重复采样、预分配零槽位和墙钟空转不产生新信用。1learner+24actor的验证不代表48actor或最大集群。一个训练seed谱系、多次冻结重复和已见trace前缀不等于多seed、盲测试或论文泛化。

departure吞吐不是应用goodput，排队不是绝对RTT；full60与预定trim10同时保留，P95/P99是run分位数而非跨场景混合包分位数。CPU按冻结单流、25—50秒归属进程user+system计，含Mahimahi时明确说明；它不是纯推理或训练集群CPU，调用减少不等比例省CPU。窗口/时间真实无探索响应、网络质量与成本须一起判断，动作方差或非精确饱和本身不是创新成功。

最终收敛为冻结baseline＋V14＋健康V31，保留必要固定周期强对照及负面来源。历史高分不抹去接口缺陷，修复后的可运行实现也不把旧模型自动升级为全局最佳或论文PASS。
