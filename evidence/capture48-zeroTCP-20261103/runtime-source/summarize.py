import pathlib,json,csv,statistics,collections,math,sys,hashlib,os
import numpy as np
E=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path(__file__).parent;C=E.parent/'common-period-capture48-20261103';A=E.parent/'common-period-cluster-20261102';mean=statistics.mean
if not (A/'evaluation-metrics.json').exists():A=E.parent/'common-period-20261102/cluster'
def read(n):return json.loads((E/n).read_text())
def save(n,x):(E/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
def table(n,x):
 with (E/n).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
probe=read('probe-results.json');support=read('training-support.json');gr=read('gradient-runs.json');points=read('gradient-points.json');cleanup=read('final-cleanup.json');old=json.loads((C/'analysis-results.json').read_text())['runs'];oldruns=list(csv.DictReader((C/'run-summary.csv').open(encoding='utf-8')))
checks={}
for label,xy in [('run48',[(mean(f['Q_label'] for f in r['finite_returns']),mean(f['predicted_Qmean'] for f in r['finite_returns'])) for r in old]),('overlapping384',[(f['Q_label'],f['predicted_Qmean']) for r in old for f in r['finite_returns']])]:
 x,y=np.asarray(xy).T;checks[label]=dict(Q_on_label_slope=float(np.cov(x,y,ddof=0)[0,1]/np.var(x)),correlation=float(np.corrcoef(x,y)[0,1]),points=len(x),independent_unit='run;384 suffixes overlap')
scan=[]
for condition in sorted({r['condition'] for r in oldruns}):
 ts=sorted({int(r['period_ms']) for r in oldruns if r['condition']==condition})
 if len(ts)!=5:continue
 vals={t:mean(float(r['utility_time_weighted']) for r in oldruns if r['condition']==condition and int(r['period_ms'])==t) for t in ts};best=max(vals,key=vals.get);scan.append(dict(condition=condition,descriptive_best_period_ms=best,utility_by_period=vals,utility40_minus80=vals[40]-vals[80]))
checks['four_scans']=scan
original=json.loads((A/'evaluation-metrics.json').read_text());cells=collections.defaultdict(list)
for r in original:
 j=r['job']
 if j['group'] in ['A20','A40','A80','A120','A160']:cells[j['condition'],j['group']].append(r)
gaps=[]
for (condition,period),rs in cells.items():
 if len(rs)!=2:continue
 for window in ['full','trim10']:
  values=[next(n['queue_mean_ms'] for n in r['network'] if n['window']==window) for r in rs];gaps.append(dict(condition=condition,period=period,window=window,queue_values_ms=values,gap_ms=abs(values[0]-values[1])))
checks['original_A_max_repeat_queue_gap']=max(gaps,key=lambda r:r['gap_ms']);save('review-checks.json',checks)
rows=[]
for r in probe['records']:
 row={k:v for k,v in r.items() if k not in ['per_test_run','top_features','MAE_mbps','RMSE_mbps','R2','baseline_MAE_mbps','baseline_RMSE_mbps']};row.update({key+'_'+target:r[key][i] for key in ['MAE_mbps','RMSE_mbps','R2','baseline_MAE_mbps','baseline_RMSE_mbps'] for i,target in enumerate(['run_capacity','recent1s_capacity'])});rows.append(row)
table('capacity-probes.csv',rows);table('gradient-by-run.csv',gr);table('gradient-state-points.csv',points);table('retained-replay-coverage.csv',support['retained_replay']);table('full-training-coverage.csv',support['full_training'])
fs=[r for r in support['full_training'] if r['period_ms']==20];ret=[r for r in support['retained_replay'] if r['period_ms']==20];summary=dict(status='COMPLETE_ZERO_TCP_CAUSAL_INTERPRETATION_PARTIAL',new_TCP=0,Canopy_optimizer_operations=0,supervised_probe_fit_only=True,D_Q='NOT_RUN',A_prime='NOT_RUN',A='FAIL_A_UNCHANGED',capacity_states=63156,capacity_models=2,capacity_feature_sets=2,capacity_validation_folds=8,probe_records=32,full_training_rows=sum(r['rows'] for r in support['full_training']),A20_full_rows=sum(r['rows'] for r in fs),A20_full_high_SRTT_rows=sum(r['high_SRTT_rows'] for r in fs),A20_full_high_ACK_rows=sum(r['high_ACK_rows'] for r in fs),A20_full_high_SRTT_seconds=sum(r['high_SRTT_seconds'] for r in fs),A20_full_seconds=sum(r['seconds'] for r in fs),A20_full_actors_with_high_SRTT=len({r['actor'] for r in fs if r['high_SRTT_rows']}),A20_retained_rows=sum(r['rows'] for r in ret),A20_retained_high_SRTT_rows=sum(r['high_SRTT_rows'] for r in ret),gradient_states=len(points)//3,gradient_high_SRTT_states=sum(r['queried_period_ms']==20 and r['high_SRTT'] for r in points),machine_hours=cleanup['task_reserved_machine_hours'],wall_seconds=cleanup['wall_elapsed_seconds'],source_execution_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [E/'export.py',E/'analyze.py']});save('results.json',summary)
# Both the statewise and run-mean signs are reported. No post-hoc significance gate.
gradient_comparison=[]
for queue in ['high_SRTT','low_SRTT','high_ACK','low_ACK']:
 for t in (20,40,80):
  selected=[r for r in gr if r['queue_stratum']==queue and r['queried_period_ms']==t and r['points']];ps=[p for p in points if p['queried_period_ms']==t and p['high_'+queue.split('_',1)[1]]==queue.startswith('high_')]
  gradient_comparison.append(dict(queue=queue,period_ms=t,runs=len(selected),Q1_positive_run_means=sum(r['Q1_mean']>0 for r in selected),Qmean_positive_run_means=sum(r['Qmean_mean']>0 for r in selected),points=len(ps),Q1_positive_point_fraction=mean(p['Q1_gradient_at_period_policy']>0 for p in ps),Qmean_positive_point_fraction=mean(p['Qmean_gradient_at_period_policy']>0 for p in ps),Q1_gradient_fixed_A20_window_mean=mean(p['Q1_gradient_at_fixed_A20_window'] for p in ps),Q1_gradient_period_policy_mean=mean(p['Q1_gradient_at_period_policy'] for p in ps)))
save('gradient-comparison.json',gradient_comparison)
report=E/'report';report.mkdir(exist_ok=True);p1=[r for r in probe['records'] if r['feature_set']=='full149' and r['model']=='extra_trees' and r['fold'].startswith('repeat')];hi=[r for r in gradient_comparison if r['queue']=='high_SRTT'];ratio=summary['A20_full_high_SRTT_rows']/summary['A20_full_rows'];gap=checks['original_A_max_repeat_queue_gap'];lines=['# Claude审阅复核与零TCP小核查','',
'结论：同意先暂停D-Q、做小核查和保持A′单变量。不同意把现有相关性直接升级为“critic造成常驻队列”“容量不可观测”“长周期控制力度不足已证实”。零TCP核查已完成，Canopy全变量/优化器/BN/target不变，原FAIL_A不改判；独立监督探针拟合不是D-Q，不更新Canopy。','',
'## 对审阅的逐项复核','',
f"1. Q校准数值可复现：48个run均值拟合Q对有限标签的斜率{checks['run48']['Q_on_label_slope']:.5f}，相关{checks['run48']['correlation']:.5f}。这说明价值区分能力差，但不单独识别容量遗漏、TD误差、训练覆盖、critic容量或其他根因。384重叠后缀拟合的数字不同，不能把它们当384个独立连接。",'',
'2. 四个完整周期扫描并非都以80ms最优。用原始逐run时间加权原效用、每格两run等权，NYC-bus与anchor的最高均值在40ms，mixed-timescale和development constant24在80ms。只是开发数据的描述性排序，未登记并列容差，不能因差不显著宣布等效或把它当连接内调度收益。', '',
'| 条件 | 40ms效用 | 80ms效用 | 描述性最高均值 |','|---|---:|---:|---:|']
for s in scan:lines.append('|%s|%.6f|%.6f|%dms|'%(s['condition'],s['utility_by_period'][40],s['utility_by_period'][80],s['descriptive_best_period_ms']))
lines += ['',f"3. 原A重复间确有大幅排队差异：本次重查最大{gap['gap_ms']:.2f}ms（{gap['condition']}，{gap['period']}，{gap['window']}）。这支持自然重连波动/多吸引态是候选解释，不证明16个超差格主要由多稳态造成，也不撤销原降级。",'',
'4. 高排队时critic仍给出正动作梯度是机制线索。真实环境中削窗/加窗对未来回报的效应没有被本轮只读查询测出，actor单次乘性动作与actor参数梯度也不同；不能据此宣称critic已被证明导致队列或双稳态。长周期贴下限且Q梯度继续削窗同样可能来自critic错误，不能当成控制力度极限已证。', '',
'## 容量探针：部分可读，跨条件失败','',
'全部63,156个完整149维状态原地导出，仅经云内网络汇总。标签分别为整run的实际service容量均值及当前s0之前1秒的service容量（启动期缩短到已有时段）。标签只供监督核查，不进入原策略/critic/奖励。Mahimahi init epoch与base相对时间显式换算；整run容量和旧独立指标最大差7.1e-15Mbps。两次重复正反互测及6次留一条件；禁止随机拆相邻行。每run按物理时间等权，Ridge α=1和32棵深度≤8 ExtraTrees参数均采前冻结。另做移除72—75上下文的固定消融。', '',
'| 完整149维ExtraTrees | run均容量MAE Mbps | run均容量R² | 最近1秒容量MAE Mbps | 最近1秒容量R² |','|---|---:|---:|---:|---:|']
for r in p1:lines.append('|%s|%.3f|%.3f|%.3f|%.3f|'%(r['fold'],r['MAE_mbps'][0],r['R2'][0],r['MAE_mbps'][1],r['R2'][1]))
lines += ['', '同连接分布的独立重复中，容量信息可被部分读出，不能称149维状态完全看不到容量；额外探针能读出也不证明当前critic已利用它。留一条件时泛化很差，全部完整149维模型在run均容量上的MAE都差于仅预测训练均值的基线，可能涉及条件/RTT指纹、容量外推、观察混叠或样本分布。这6个选定条件不足以识别普遍的可观测性。留出常量容量条件时run均容量R²记null；近常量1秒标签的R²会因微小分母极端负，主要看MAE和基线。不能挑一个失败模型证明无信息，也不能挑一个成功划分证明通用可估计。', '',
'状态编码确无显式绝对吞吐/绝对cwnd/容量输入：速率、RTT样本数量和in-flight等以观测峰值/cwnd归一化；绝对RTT和时钟可用，throughput-log上下文只有变化/离散度，未保留其EWMA绝对水平。但比例化并不自动消除动态、离散化和路径相关的容量信息；应区分“无显式字段”与“统计上不可读”。', '',
'## 高排队A20状态的跨周期局部查询','',
f"12个A20 run各取128个等物理时间点（共{summary['gradient_states']}个），当前SRTT/minRTT≥2的{summary['gradient_high_SRTT_states']}点为主分层，ACK区间RTT另列。每个完全相同149维状态查询20/40/80ms：既在各周期mu处求导，也固定重建的A20窗口只改变critic周期输入。其他周期均为离策略，没有新TCP动作干预。", '',
'| 高SRTT分层 | Q1均值为正的run数 | Qmean均值为正的run数 | 固定A20窗口时Q1平均导数 |','|---|---:|---:|---:|']
for r in hi:lines.append('|%dms|%d/%d|%d/%d|%.4f|'%(r['period_ms'],r['Q1_positive_run_means'],r['runs'],r['Qmean_positive_run_means'],r['runs'],r['Q1_gradient_fixed_A20_window_mean']))
lines += ['', '同一状态、固定A20窗口后跨周期梯度确实会改变，故周期条件化并非完全失效；各周期mu处和固定窗口处的符号可以不同。Q1与Qmean也可不同，actor实际用Q1。各run高/低队列子集都存在，不能把“run整体平均RTT高”与“高排队状态子集”混用来复现7/8等计数。128个时点的子集均值不是独立环境梯度试验，尚无削窗真实收益标签。逐状态和逐run结果完整保留。', '',
'## 训练覆盖：不是没见过，但不等于覆盖充分','',
f"全216训练episode的711,285条决策按pre-action原生序列对齐检查。A20共有{summary['A20_full_rows']:,}条，其中高SRTT{summary['A20_full_high_SRTT_rows']:,}条（{ratio*100:.2f}%），对应{summary['A20_full_high_SRTT_seconds']:.2f}/{summary['A20_full_seconds']:.2f}物理秒；22/24 actor出现。ACK定义高队列{summary['A20_full_high_ACK_rows']:,}条。最终回放A20为{summary['A20_retained_rows']:,}条，其中高SRTT{summary['A20_retained_high_SRTT_rows']:,}条；随机与近中性行为来源均有。",'',
'全程计数使用native真实SRTT；最终回放由状态上下文恢复的是策略观测SRTT，含4个显式噪声条件的观测扰动，两者口径另列。排除4个noisy条件后，A20仍有38,623/274,630条真实高SRTT状态，20个无噪声actor都出现。', '',
'这排除了“训练完全没有见过此粗分层”这一解释；不证明各失败条件、容量、历史、窗口与actor策略的联合覆盖充分。6个补采条件中部分是开发或不同trace/queue别名，不能把总体12%冒充每个失败格已覆盖。仅因高队列存在也不能直接断言加长探索分段无用。完整分actor/条件/周期和最终回放行为分解见CSV/JSON。', '',
'## 对A′奖励的建议与原约定','',
'全局6MB/s使相同相对利用率在不同容量上产生不同奖励尺度，这一设计事实成立；却不自动证明它是本轮高队列/弱条件的原因。按链路容量归一化是值得优先比较的单变量候选，尤其当前“完全没见过高队列”的解释不成立；仍应标假设，不称已修复结构性根因。','',
'原md1.2明确只允许观测峰值/因果平滑峰值/冻结单一全局尺度，并写“真实容量只用于离线审计”。因此训练期oracle奖励虽然不要求部署查表，仍是对原研究协议的实质修订。若获批，应定义reward v2、独立checksum/回放与模型版本、重新校准K/critic，原A和FAIL_A完整保留。','',
'还须区分两种分母：常量“训练条件的固定参考容量”只在条件内对原奖励作正比例缩放，主要改变跨条件损失/梯度权重及价值尺度；它并不直接改变该条件内真实策略排序。瞬时C(t)则给不同时段不同权重，一般有mean(throughput/C(t))≠sum(throughput)/sum(C(t))，不能说天然等于动态网络的容量积分评测。优先具体讨论固定条件容量的reward v2单变量匹配A′，不要同时改探索、状态、网络头、正则或周期。','',
'下一项只作为待讨论/待定预算的A′登记草案：冻结A原启动权重/seed/24条件/重置/样本与物理更新账本；对照同管线重跑原reward v1，试验只改归一化分母；匹配冻结评测及新的分层健康标准必须采前登记。健康标准可把“边界且利用率差/高队列”等与单纯边界分开，但不因critic还要求削窗就豁免控制失败，不追溯改判旧A。本轮未选定最终分母值、未授权任何新训练预算，也未启动A′或D-Q。','',
'## 预算、只读与错误保留','',
f"从首次云连接至五机终检{cleanup['wall_elapsed_seconds']/60:.2f}分钟；learner连续保留和四actor短导出/核查加总{cleanup['task_reserved_machine_hours']:.4f}机时，小于1机时。五机本轮进程0、未分配新IPC、51保护模型/native/源码/sysctl未变；模型全变量前后digest与checkpoint SHA不变，replay SHA不变。actor导出数据约5.45MB仅经VPC，完整63156状态留云端。本机只回传578,694字节压缩小指标和选定梯度点，未下载checkpoint/replay或完整日志。实例未自动关闭，本轮已无云任务。",'',
'第一次离线导出把Mahimahi相对base误作epoch，严格断言在写任何特征包前失败；据原日志init epoch显式转换后仅重读旧文件，原失败响应/旧源码保留，未重置预算或追加TCP。两个控制脚本字符串定界语法错误均在云端执行前发生并修复。代码执行SHA与公开脱敏字节分别记录。','']
(report/'review-and-zeroTCP.md').write_text('\n'.join(lines),encoding='utf-8')
# Two compact scientific figures.
sys.path.insert(0,str(A/'plot-deps'));os.environ['MPLCONFIGDIR']=str(E/'mplconfig');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True);periods=[r['period_ms'] for r in hi];axes[0].plot(periods,[r['Q1_gradient_fixed_A20_window_mean'] for r in hi],'o-',label='Fixed A20 window');axes[0].plot(periods,[r['Q1_gradient_period_policy_mean'] for r in hi],'o--',label='At each period policy');axes[0].axhline(0,color='gray',linewidth=1);axes[0].set(xlabel='Queried period (ms)',ylabel='Q1 derivative wrt window action',title='A20 high-SRTT states: off-policy queries');axes[0].legend(fontsize=8)
for i,r in enumerate(p1):axes[1].bar(np.arange(2)+i*.3,r['MAE_mbps'],width=.3,label=r['fold'])
axes[1].set(xticks=np.arange(2)+.15,xticklabels=['Run mean capacity','Past1s capacity'],ylabel='Held-out MAE (Mbps)',title='Small readout on separate TCP repeats');axes[1].legend(fontsize=8);fig.savefig(str(report/'zeroTCP-checks.png'),dpi=150);fig.savefig(str(report/'zeroTCP-checks.svg'));plt.close(fig)
print(json.dumps(summary,indent=2))
