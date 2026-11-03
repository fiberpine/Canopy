import pathlib,json,csv,statistics,collections,sys,os
E=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path(__file__).parent;mean=statistics.mean
reg=json.loads((E/'registration.json').read_text());metrics=json.loads((E/'captured-run-metrics.json').read_text());analysis=sum([json.loads(p.read_text())['runs'] for p in sorted(E.glob('node*-analysis.json'))],[]) if list(E.glob('node*-analysis.json')) else json.loads((E/'analysis-results.json').read_text())['runs'];byrun={r['run']:r for r in analysis};native={r['run']:r for r in json.loads((E/'native-audit.json').read_text())['runs']};assert len(metrics)==len(analysis)==48
def write(name,x):(E/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
def table(name,rows):
 with (E/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
runs=[];points=[];returns=[];checks=[];groups=collections.defaultdict(list)
for m in metrics:
 a=byrun[m['run']];j=m['job'];net=next(w for w in m['network'] if w['window']=='trim10');fs=a['finite_returns'];assert len(fs)==8
 row=dict(run=m['run'],condition=j['condition'],period_ms=j['period_ms'],repeat=j['repeat'],node=j['node'],instrumentation_pass=m['instrumentation_pass'],utilization=net['utilization'],queue_mean_ms=net['queue_mean_ms'],queue_p95_ms=net['queue_p95_ms'],decisions=a['all_run']['decisions'],utility_time_weighted=a['all_run']['utility_time_weighted'],raw_R_sum=a['all_run']['raw_R_sum'],centered_R_sum=a['all_run']['centered_R_sum'],boundary_fraction=a['all_run']['boundary_fraction'],exact_saturation_fraction=a['all_run']['exact_float32_saturation'],actor_forward_max_abs_error=a['actor_forward_max_abs_error'],Q_prediction_mean=mean(x['predicted_Qmean'] for x in fs),Q_label_mean=mean(x['Q_label'] for x in fs),Q_label_minus_prediction_mean=mean(x['Q_label_minus_prediction'] for x in fs),Q_tail_abs_max=max(abs(x['Q_tail']) for x in fs),N_label_minus_prediction_mean=mean(x['N_label_minus_prediction'] for x in fs),min_fc1_cross_period_cosine=min(x['by_layer']['fc1'] for x in a['shared_state_gradient_pairs']),min_fc2_cross_period_cosine=min(x['by_layer']['fc2'] for x in a['shared_state_gradient_pairs']),BN1_batch_action_abs_mean=a['BN_hypothetical']['batch1_moving2_action_abs_mean'],BN_both_batch_action_abs_mean=a['BN_hypothetical']['batch_both_action_abs_mean'])
 components=native[m['run']]['reward_components'];assert abs(components['raw_R_sum']-row['raw_R_sum'])<=reg['tolerances']['reward_abs'];assert abs(components['centered_R_sum']-row['centered_R_sum'])<=reg['tolerances']['reward_abs'];row.update(components)
 runs.append(row);groups[j['condition'],j['period_ms']].append(row)
 points += [dict(run=m['run'],condition=j['condition'],period_ms=j['period_ms'],repeat=j['repeat'],**x) for x in a['point_samples']]
 returns += [dict(run=m['run'],condition=j['condition'],period_ms=j['period_ms'],repeat=j['repeat'],**x) for x in fs]
 checks += [dict(run=m['run'],condition=j['condition'],period_ms=j['period_ms'],repeat=j['repeat'],window=w['window'],**x) for w in m['instrumentation_checks'] for x in w['checks']]
cells=[]
for (condition,t),rs in sorted(groups.items()):
 assert len(rs)==2 and {r['repeat'] for r in rs}=={0,1};rs.sort(key=lambda r:r['repeat']);row=dict(condition=condition,period_ms=t,interpretation='MATCHED_WITHIN_REGISTERED_MARGINS' if all(r['instrumentation_pass'] for r in rs) else 'DOWNGRADED_ORIGINAL_A_TRANSFER',runs=2,passing_runs=sum(r['instrumentation_pass'] for r in rs))
 for key in ['utilization','queue_mean_ms','queue_p95_ms','utility_time_weighted','boundary_fraction','exact_saturation_fraction','Q_prediction_mean','Q_label_mean','Q_label_minus_prediction_mean','N_label_minus_prediction_mean','min_fc1_cross_period_cosine','min_fc2_cross_period_cosine']:
  row[key+'_r0']=rs[0][key];row[key+'_r1']=rs[1][key];row[key+'_mean']=mean(r[key] for r in rs)
 cells.append(row)
table('run-summary.csv',sorted(runs,key=lambda r:r['run']));table('cell-summary.csv',cells);table('instrumentation-checks.csv',checks);table('state-point-diagnostics.csv',points);table('finite-return-labels.csv',returns);write('analysis-results.json',dict(runs=analysis,optimizer_operations=0,D_Q='NOT_RUN'))
table('offpolicy-period-preferences.csv',[dict(run=a['run'],condition=a['job']['condition'],executed_period_ms=a['job']['period_ms'],repeat=a['job']['repeat'],**{'argmax_'+k:v for k,v in a['offpolicy_Qmean_argmax_counts'].items()}) for a in analysis])
stats=dict(status='CAPTURE_COMPLETE_INTERPRETATION_PARTIAL',TCP_attempts=48,TCP_completed=48,TCP_failures=0,optimizer_operations=0,D_Q='NOT_APPROVED_NOT_RUN',A='FAIL_A_UNCHANGED',captured_transitions=sum(m['capture_checks']['rows'] for m in metrics),complete_state_dimension=149,raw_fields=15,exact_utility_max_error=max(m['capture_checks']['utility_max_abs_error'] for m in metrics),actor_forward_max_abs_error=max(r['actor_forward_max_abs_error'] for r in runs),instrumentation_passing_runs=sum(r['instrumentation_pass'] for r in runs),instrumentation_passing_cells=sum(c['passing_runs']==2 for c in cells),downgraded_cells=sum(c['passing_runs']<2 for c in cells),instrumentation_failed_checks=sum(not c['passed'] for c in checks),instrumentation_total_checks=len(checks),any_exact_float32_saturation=any(r['exact_saturation_fraction']>0 for r in runs),max_boundary_fraction=max(r['boundary_fraction'] for r in runs),runs_with_negative_fc1_cross_period_cosine=sum(r['min_fc1_cross_period_cosine']<0 for r in runs),runs_with_negative_fc2_cross_period_cosine=sum(r['min_fc2_cross_period_cosine']<0 for r in runs),min_fc1_cross_period_cosine=min(r['min_fc1_cross_period_cosine'] for r in runs),min_fc2_cross_period_cosine=min(r['min_fc2_cross_period_cosine'] for r in runs),finite_return_labels=len(returns),independent_runs_per_cell=2,Q_label_minus_prediction_run_mean_min=min(r['Q_label_minus_prediction_mean'] for r in runs),Q_label_minus_prediction_run_mean_max=max(r['Q_label_minus_prediction_mean'] for r in runs),max_abs_Q_tail=max(abs(r['Q_tail']) for r in returns),max_discounted_tail_twin_half_difference=max(r['tail_twin_half_difference'] for r in returns),same_initial_state_distinct_hashes=len({a['initial_state_sha256'] for a in analysis}),all_model_variables_unchanged=all(json.loads(p.read_text())['model_variables_unchanged'] for p in E.glob('node*-analysis.json')),BN1_batch_action_abs_mean_range=[min(r['BN1_batch_action_abs_mean'] for r in runs),max(r['BN1_batch_action_abs_mean'] for r in runs)])
write('results.json',stats)
report=E/'report';report.mkdir(exist_ok=True)
lines=['# 48 次冻结补采：采证完成，原 A 解释保持降级边界','',f"48/48 次 60 秒连接完成，0 网络失败、0 优化器操作。采得 {stats['captured_transitions']:,} 个完整 149 维状态/动作/后继区间及 15 个原始字段。原 FAIL_A 不改判，D-Q、B0/C、新训练和 λ 扫描均未执行。",'',
'用户批准原38次加10次；实际24格×2次，新增anchor/R20和mixed-timescale/R20的A80，以及constant24/R100的A80/A120/A160。四个条件因此具有20/40/80/120/160ms完整扫描，另外两个仅20/40ms。条件按原host和重复配对，60秒运行长度不变。','',
'## 插桩容差：先登记，逐格降级','',f"利用率差≤3个百分点，mean queue差≤max(2ms,原值20%)，P95差≤max(3ms,原值20%)；full、0—10秒和trim10全部逐重复核对。48个run中{stats['instrumentation_passing_runs']}个全过；24格中{stats['instrumentation_passing_cells']}格两次全过，{stats['downgraded_cells']}格降级。{stats['instrumentation_total_checks']}项核对有{stats['instrumentation_failed_checks']}项超差。详见逐项CSV与cell-summary.csv。",'',
'超差不能分辨是插桩影响还是重新建连的自然变化。对应格只用于描述本轮新轨迹，不能直接解释旧A失败轨迹；不通过合并平均、放宽容差或追加重复挽救。完整状态只在RAM保留、原生停止后写NPZ，没有加入TCP中的额外神经网络查询。','',
'## 奖励、边界与局部梯度','',f"逐区间复算原reward v1效用与部署记录的最大绝对误差为{stats['exact_utility_max_error']:.3g}；原始reward和K中心化reward分别保留，没有替换为包均值代理。actor实际逐状态前向核对最大差{stats['actor_forward_max_abs_error']:.3g}，未放宽1e-6容差。",'',
f"全部新连接未观察到float32精确±1饱和，最高abs(a)>.99占比为{100*stats['max_boundary_fraction']:.2f}%。因此整数25%动作或边界集中不能直接解释为tanh精确死亡。每run预登记32个均匀点及前4次边界跨越相邻点，逐点记录z、局部导数、Q1/Qmean动作梯度和正则分量。没有精确饱和也不意味着动作正确，Q导数仍是critic的判断。",'',
f"部署状态上的16点固定cohort跨周期查询，fc1有{stats['runs_with_negative_fc1_cross_period_cosine']}/48个run、fc2有{stats['runs_with_negative_fc2_cross_period_cosine']}/48个run出现负梯度余弦，最低分别{stats['min_fc1_cross_period_cosine']:.4f}、{stats['min_fc2_cross_period_cosine']:.4f}。这与D0训练状态样本上的正方向结果不同，说明不能外推训练状态上的一致性到部署状态。各周期RL/正则范数、逐层余弦均保留；不能仅据period_w因编码符号产生的负方向判冲突。",'',
'这些是冻结critic驱动的离策略跨周期查询，含启动/steady混合16点的批平均；不是真实环境梯度或历史训练损失的总体估计。共享trunk已有反向案例，当前证据不足以直接决定“仅每周期分一个末层头”能解决。BN反事实batch路径差异只说明敏感性；实际部署前向已核对，不判BN接线错误。','',
'## Q 的周期偏好与有限回报','',f"每run固定物理0、5、…35秒起点，得到{stats['finite_return_labels']}个H≥20秒同连接后缀标签；按run聚合，每格仍只有2条独立连接。Q标签用真实中心化回报加原冻结mean(Q1,Q2)尾项，N同理；没有训练或校准。run均值的“标签−预测”范围{stats['Q_label_minus_prediction_run_mean_min']:.2f}至{stats['Q_label_minus_prediction_run_mean_max']:.2f}，既有高估也有低估，不能用一个统一偏差概括所有条件/周期。",'',
f"最大折扣Q尾项绝对值{stats['max_abs_Q_tail']:.4f}，最大折扣twin半差{stats['max_discounted_tail_twin_half_difference']:.4f}；幅度与twin差都不是未知尾项误差的上界。重叠后缀不增加独立样本数，不据此生成虚假的窄置信区间。",'',
'四个完整扫描条件的实际效用、当前Q预测和同策略标签绘在图中。另保留每run同一16状态上五周期的Qmean argmax计数；其偏好随条件/状态变化，不是固定总偏向某个周期，也不等于选择器收益。各周期自己的轨迹不是同状态反事实真值，启动状态SHA也逐run保留；不以固定策略扫描宣布动态selector可行或不可行。','',
'## 工程核验与下一项选择','',
'模型所有变量（含BN/target/优化器槽）前后摘要一致；native动作映射、连续状态ID、同连接s1→s0、原生计时/ACK/loss字段另表核对。所有NPZ、原始逐包/计时日志仍在云端，本机仅有小JSON/CSV、代码、图和哈希，不下载checkpoint/replay。五机模型/51保护文件/源码/系统参数和归属进程/IPC最终核查见cleanup.json。', '',
'优先候选仍是另行审批D-Q，以固定数据和明确预算检验价值估计能否改善；当前结果不足以直接换头、加正则、改采样或长训。D0短周期采样权重结论不变。CPU仍沿用D0模型微基准；约90ms的CPU交点只是线性实现投影，本轮未做selector闭环CPU验证。', '',
'分析辅助错误均保留：首次批量矩阵路径出现1.788e-6前向差，改成部署单状态路径并保持1e-6容差；随后NumPy索引JSON序列化失败，仅修输出类型。只重算已采数据，未重跑任何TCP，未改采样/科学阈值。analysis-v1/v2及两次错误记录、修订SHA都保留。', '',
'文件：registration.json（采前登记）、captured-run-metrics.json（原始小指标）、analysis-results.json（只读诊断）、run-summary.csv、cell-summary.csv、instrumentation-checks.csv、state-point-diagnostics.csv、finite-return-labels.csv。两次重复逐项保留，不只给平均。']
if (E/'cleanup.json').exists():
 c=json.loads((E/'cleanup.json').read_text());lines += ['',f"预算：首次准备连接至五机终检{c['wall_elapsed_seconds']/60:.2f}分钟，保守连续五机占用{c['reserved_machine_hours']:.3f}机时，分别低于45分钟/4机时。17:54:27起保留最后10分钟收尾；实例未自动关机，任务机时不等于云商账单。"]
lines += ['', '## 两次重复的周期扫描明细', '', '每项以“重复0 / 重复1”给出；Q差为同策略有限标签减当前预测。降级仅限制转用到旧A，数据不删除。', '', '| 条件 | 周期ms | trim利用率% | mean queue ms | 平均效用 | Q差 | 旧A可转用 |', '|---|---:|---:|---:|---:|---:|---|']
for c in cells:
 pair=lambda key,scale=1:' / '.join('%.2f'%(scale*c[key+'_r%d'%r]) for r in (0,1))
 lines.append('| %s | %d | %s | %s | %s | %s | %s |'%(c['condition'],c['period_ms'],pair('utilization',100),pair('queue_mean_ms'),pair('utility_time_weighted'),pair('Q_label_minus_prediction_mean'),'容差内' if c['passing_runs']==2 else '降级'))
lines += ['', '48个启动状态SHA全部不同，没有确认任何跨run同初始149维状态；不把启动时刻当成精确匹配反事实。', '', '![四条件完整周期扫描](period-scans.png)']
(report/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Standard scientific plots, with both repeats shown separately.
sys.path.insert(0,str(E.parent/'common-period-cluster-20261102/plot-deps'));os.environ['MPLCONFIGDIR']=str(E/'mplconfig');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
scans=[c for c in sorted({r['condition'] for r in runs}) if len({r['period_ms'] for r in runs if r['condition']==c})==5];fig,axes=plt.subplots(4,4,figsize=(16,12),constrained_layout=True)
for column,c in enumerate(scans):
 for rep in (0,1):
  rs=sorted([r for r in runs if r['condition']==c and r['repeat']==rep],key=lambda r:r['period_ms']);ts=[r['period_ms'] for r in rs]
  for row,(key,label) in enumerate([('utilization','Trim10 utilization'),('utility_time_weighted','Raw utility (time weighted)'),('Q_prediction_mean','Own-state Q / suffix label'),('min_fc1_cross_period_cosine','Min shared fc1 cosine')]):
   ax=axes[row,column];ax.plot(ts,[r[key] for r in rs],marker='o',color=['tab:blue','tab:orange'][rep],label='repeat%d'%rep);ax.set_xticks(ts);ax.set_ylabel(label);ax.grid(alpha=.25)
   if row==0:ax.set_ylim(0,1.05)
   if row==3:ax.set_ylim(-1.05,1.05)
   if row==2:ax.plot(ts,[r['Q_label_mean'] for r in rs],linestyle='--',marker='x',color=['tab:blue','tab:orange'][rep],label='label repeat%d'%rep)
  axes[0,column].set_title(c.replace('V34-development-','dev-').replace('V17-',''),fontsize=9);axes[3,column].set_xlabel('Fixed period (ms)')
axes[0,0].legend(fontsize=7);axes[2,0].legend(fontsize=7);fig.suptitle('Frozen capture48: 2 independent runs/cell; own-state curves are not counterfactuals')
fig.savefig(str(report/'period-scans.png'),dpi=150);fig.savefig(str(report/'period-scans.svg'));plt.close(fig)
print(json.dumps(stats,indent=2))
