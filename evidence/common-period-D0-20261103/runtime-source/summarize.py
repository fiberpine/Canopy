import collections,csv,json,pathlib,statistics,math
E=pathlib.Path(__file__).parent;P=E/'report';P.mkdir(exist_ok=True)
reg=json.loads((E/'registration.json').read_text());nodes=[json.loads((E/('node%d-results.json'%i)).read_text()) for i in range(5)];learner=json.loads((E/'learner-results.json').read_text());cleanup=json.loads((E/'cleanup.json').read_text());coverage=[c for n in nodes for c in n['coverage']];train=[r for n in nodes for r in n['training']];reward=[r for n in nodes for r in n['training_reward_audit']];grad=learner['S1']['actor_gradients'];pairs=[r for r in grad if 'pair_ms'in r];norms=[r for r in grad if 'period_ms'in r]
assert len(coverage)==264 and sum(r['count'] for r in train)==711285
table=[]
for actor in range(24):
 for period in [20,40,80,120,160]:
  rs=[r for r in train if r['actor']==actor and r['period_ms']==period];table.append(dict(actor=actor,period_ms=period,inserted=sum(r['count'] for r in rs),physical_seconds=sum(r['physical_seconds'] for r in rs),empty_RTT=sum(r['empty_RTT'] for r in rs)))
def write_csv(name,rows):
 if not rows:return
 fields=sorted(set().union(*(r.keys() for r in rows)))
 with (E/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
write_csv('actor-period-ledger.csv',table);write_csv('actor-period-time-bins.csv',train)
coverage_csv=[dict(run=r['run'],group=r['group'],condition=r['condition'],repeat=r['repeat'],state_rows=r['state_rows'],state149_recorded=r['state149_recorded'],reward_status=r['reward_status'],missing_reward_fields=';'.join(r['missing_reward_fields']),S4_status=r['S4_status']) for r in coverage];write_csv('frozen-field-coverage.csv',coverage_csv)
layer_stats={k:dict(min=min(r['by_layer'][k] for r in pairs if r['by_layer'][k] is not None),median=statistics.median(r['by_layer'][k] for r in pairs if r['by_layer'][k] is not None),max=max(r['by_layer'][k] for r in pairs if r['by_layer'][k] is not None)) for k in ['shared_fc1','shared_fc2','output','period_input']}
period_ledger=[]
for period in [20,40,80,120,160]:
 rs=[r for r in table if r['period_ms']==period];period_ledger.append(dict(period_ms=period,inserted=sum(r['inserted'] for r in rs),physical_seconds=sum(r['physical_seconds'] for r in rs),retained_sampler_expected_weight=learner['S1']['expected_final_actor_balanced_period_weights'][[20,40,80,120,160].index(period)]))
cpu=learner['S5'];bn=learner['S3']['BN_counterfactual_paths'];local=learner['S3']['training_state_local_queries'];stateparity=cleanup['nodes'][0]['data']['startup_state_parity']
results=dict(status='PARTIAL_D0_MISSING_DEPLOYMENT_FIELDS',A_status='FAIL_A_UNCHANGED',B0='NOT_RUN',C='NOT_RUN',scope=dict(new_congestion_control_TCP=0,optimizer_operations=0,lambda_scan=False,original_model_files_unchanged=True,model_variables_unchanged=learner['model_variables_unchanged'],actor_transport_unchanged=learner['actor_transport_variables_unchanged']),S1=dict(status='PARTIAL',cumulative_inserted=711285,cumulative_physical_seconds=sum(r['physical_seconds'] for r in table),period_ledger=period_ledger,gradient_pair_count=len(pairs),RL_pair_cosine=dict(min=min(r['rl_cosine'] for r in pairs if r['rl_cosine'] is not None),median=statistics.median(r['rl_cosine'] for r in pairs if r['rl_cosine'] is not None)),by_layer=layer_stats,historical_sampling_weights='NOT_RECORDED',contemporaneous_policy_lag='NOT_RECORDED',version_distance_to_final=learner['S1']['retained_distance_to_final_version'],caveat='Cumulative ingress, final replay expected weights, and finite frozen gradient queries are different quantities; final-version distance is not online lag'),S2=dict(status='NOT_IDENTIFIABLE_FOR_CORE120',frozen_runs_audited=len(coverage),A_core_runs=sum(x['group'].startswith('A') for x in coverage),A_core_reconstructible=sum(x['group'].startswith('A') and x['original_reward_reconstructible'] for x in coverage),missing_core_fields=['avg_rtt_us','srtt_us_scaled8','min_rtt_us'],training_control=dict(episodes=len(reward),max_reward_abs_error=max(r['max_abs_error'] for r in reward)),caveat='Training reward controls do not reconstruct frozen deployment reward. Logged time-weighted utilities, if present, are not interval SMDP returns; external legacy utilities use different rewards.'),S3=dict(status='PARTIAL_DEPLOYMENT_STATE_NOT_RECORDED',frozen_state149_runs=sum(r['state149_recorded'] for r in coverage),failure_cells_covered_by_native=len(reg['failed_cells']),deep_conditions=reg['deep_conditions'],training_startup_queries=len(learner['S3']['training_startup_queries']),startup_state_parity_matches=sum(r['status']=='PASS_EXACT_RECONSTRUCTION' for r in stateparity),startup_state_parity_abs_max=max(r['max_state149_abs_error'] for r in stateparity if r['max_state149_abs_error'] is not None),actual_actor_forward_max_abs=learner['S3']['actor_only_vs_learner_max_abs_error'],moving_diagnostic_forward_max_abs=max(r['moving_forward_max_error'] for r in bn),BN_batch1_moving2_mean_action_difference=statistics.mean(r['batch1_moving2_action_abs_difference']['mean'] for r in bn),BN_batch_both_mean_action_difference=statistics.mean(r['batch_both_action_abs_difference']['mean'] for r in bn),training_query_exact_saturation_fraction=statistics.mean(r['float32_exact_saturation'] for r in local),training_query_local_zero_RL_fraction=statistics.mean(r['zero_RL_local_fraction'] for r in local),caveat='Training-state and BN alternative forwards do not establish missing failure-state local gradients or a BN implementation bug. Integer25-percent commands do not identify precise tanh saturation.'),S4=dict(status='NOT_IDENTIFIABLE',registered_horizon_seconds=20,reason='No full149 frozen deployment states and no original ACK-RTT reward intervals; no same-policy finite-return labels fabricated',initial_state_matched_frozen_reference='NOT_RECORDED',new_value_optimization='NOT_RUN'),S5=cpu,cleanup=cleanup['status'],budget_at_final_check=dict(wall_seconds=cleanup['wall_elapsed_seconds'],task_machine_hours=cleanup['task_reserved_machine_hours']),next_action='Separate approval for a finite frozen data-collection run list; no automatic D-Q, intervention, or training')
(E/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
lines=['# D0 冻结诊断结果（2026-11-03）','', '**结论：PARTIAL_D0_MISSING_DEPLOYMENT_FIELDS。原 FAIL_A 不改判，B0/C 均未运行。**','',
'本轮只读旧模型和旧数据，0 新拥塞控制 TCP、0 优化器操作、0 λ 扫描、0 生产算法修改。五机身份、51 保护模型、A 最终 checkpoint 与源码身份核验通过；模型/actor 克隆变量摘要前后一致。SSH/小文件传输不算拥塞控制实验。','',
'## 能确认的结果','',
'| 项目 | 结果 |','|---|---|',
'| S1 入库与学习分配 | 全部 711,285 训练区间重建；全程按周期采样/梯度贡献及同时刻策略滞后未记录，不能由末尾回放反推 |',
'| S2 实际奖励 | 核心120次冻结评测全部缺平均ACK RTT、sRTT、minRTT，无法精确重算；216训练episode作为对照可重建，误差见下 |',
'| S3 失败状态与BN | 264个冻结run均无完整149状态；native动作轨迹可解析，失败状态局部梯度不可识别；训练启动/保留状态上的冻结查询单列 |',
'| S4 现有critic核验 | 缺冻结部署状态和原区间奖励，不能构造H≥20s同策略标签，NOT_IDENTIFIABLE |',
'| S5 CPU | 同机同状态、单线程、预热后的微基准完成；仅支持成本投影 |','',
'### S1：三种分配分别记录','',
'| 周期 ms | 全程入库条数 | 实际物理秒 | 最终actor均衡采样器的周期期望权重 |','|---|---:|---:|---:|']
for r in period_ledger:lines.append(f"| {r['period_ms']} | {r['inserted']} | {r['physical_seconds']:.6f} | {100*r['retained_sampler_expected_weight']:.4f}% |")
lines+=['',f"最终20/160ms期望权重比 {period_ledger[0]['retained_sampler_expected_weight']/period_ledger[-1]['retained_sampler_expected_weight']:.4f}。这是最终回放内 actor 均衡、actor 内均匀抽样的期望，不是全程实际采样次数或梯度贡献。原始采样器没有逐周期累计抽样账本，不能补成精确历史。",'',
f"8个固定64状态batch、同状态跨周期查询，共80个周期对；RL全参数梯度cosine最小 {results['S1']['RL_pair_cosine']['min']:.6f}、中位 {results['S1']['RL_pair_cosine']['median']:.6f}。共享fc1/fc2最小分别 {layer_stats['shared_fc1']['min']:.6f}/{layer_stats['shared_fc2']['min']:.6f}。本有限训练状态集没有显示共享trunk的负向梯度冲突；不据此排除其他失败状态的冲突或表达不足。period_w的正负周期编码天然会使该参数梯度方向不同，不能单独当分头证据。batch不是独立TCP重复。",'',
f"保留行距离最终快照版本的中位差 {learner['S1']['retained_distance_to_final_version']['p50']:.1f}，其中包含采集结束后排空的更新；没有同时刻learner/actor版本记录，因此在线策略滞后是NOT_RECORDED。",'',
'### S2–S4：缺失字段的具体边界','',
'`frozen-field-coverage.csv`列出264个run原始列名覆盖。A的120个run均未打开扩展RTT审计；states.csv只有表头，没有逐步状态。旧packet平均排队/包丢失率不能代替ACK区间RTT、原丢失字节或完整149状态。已有summary里记录的实际时间加权utility只作原字段描述，不冒充逐区间折扣奖励或跨旧奖励参照。','',
f"训练日志正对照216个episode的最大原奖励误差为 {results['S2']['training_control']['max_reward_abs_error']:.4g}。48个按预登记首/末训练episode抽取的启动状态中，24个仍在最终replay，149维逐位一致、最大误差0；另外24个已不在保留回放，未假定匹配。这些是训练启动状态上的最终模型查询，不是原冻结失败连接状态。",'',
f"实际actor-only传输前向与最终learner前向最大差 {results['S3']['actual_actor_forward_max_abs']:.3g}；相同moving-BN诊断前向最大差 {results['S3']['moving_diagnostic_forward_max_abs']:.3g}。只读改用BN1 batch/BN2 moving、两层batch统计，动作绝对差的batch均值分别约 {results['S3']['BN_batch1_moving2_mean_action_difference']:.6f}/{results['S3']['BN_batch_both_mean_action_difference']:.6f}。这量化统计路径敏感性，不是部署路径接错或性能因果的证据。",'',
f"512个保留训练状态×5周期查询的float32精确饱和比例均值 {100*results['S3']['training_query_exact_saturation_fraction']:.6f}%，局部RL项为零比例均值 {100*results['S3']['training_query_local_zero_RL_fraction']:.6f}%。缺少实际失败状态，不能将这些值推广到原8个失败格；abs(a)>.99、精确±1与梯度为零必须继续分开。",'',
'完整native整数动作/边界窗口/交付/丢失/RTT样本数时序仅在本轮预登记的格中抽样（首8点+32均匀位置）；回涨比描述动作后的时间演变，不裁定唯一原因。无完整状态与奖励，因此没有缩短H或改拿训练随机策略回报来冒充最终同策略价值标签。','',
'### S5：模型段CPU微基准','',f"6个交错顺序block，各512次，固定128状态池，预热128次，TF intra/inter、OMP/OpenBLAS均为1。单actor进程CPU均值 **{cpu['actor_cpu_ms']:.6f} ms/调用**，五候选actor+双Q+N的原两次Session取值结构为 **{cpu['selector_cpu_ms']:.6f} ms/调用**，倍率 **{cpu['ratio']:.4f}**。保留只读保护器开销，另测wire序列化桥接；不是完整run CPU。",'',
'| 既有固定周期调用率参照 | 实测调用/s | actor模型CPU%单核 | 选择器在相同调用率CPU%单核 |','|---|---:|---:|---:|']
for r in cpu['cost_projection']:lines.append(f"| {r['period_reference_ms']}ms | {r['observed_fixed_actor_calls_per_s']:.4f} | {r['actor_cpu_one_core_percent']:.4f} | {r['selector_at_same_calls_cpu_one_core_percent']:.4f} |")
lines+=['','这些投影固定调用率，未计策略改变后的network/native/全流程负载。倍率大于2不构成一般不可能性证明，也不宣布闭环CPU节省；λ没有扫描，选择输出没有用于TCP。','',
'## 下一项待审批提案','',
'优先补齐冻结部署证据，再决定是否做D-Q或新训练。建议**另行审批38次完整60秒冻结采证**：原8个边界失败格＋5个深缓冲条件的A20/A40共17个去重格，再补NYC-bus-R20/q5的A20/A40正常参照，共19格×2独立连接。条件均已见，只作诊断，不称盲测。固定原A最终模型、周期、窗口语义和时钟；不改奖励、采样器、网络，不做价值优化或λ扫描。','',
'采证只增加每条真实边界的原始15字段、完整149状态、实际浮点动作/整数命令、同连接后继、初始状态和模型摘要；优先RAM缓冲结束后写出。登记每格顺序/独立重复、精确开始/截止、20秒合法后缀及采证开销校验。建议该新阶段最多45分钟墙钟、最后10分钟收尾、每机一流；**这些是本次结果形成的待审批建议，本轮没有执行，也没有沿用旧45次干预草案**。若不批准新TCP，应保持S2–S4不可识别，不能凭当前D0定分头/正则/改采样/加训。','',
'本提案是补齐诊断数据，不能重写原FAIL_A或直接完成B0。D-Q、动作干预、新A′训练和正式B0/C仍须分别审批。','',
'## 预算、保护与交付','',
f"五机终检 {cleanup['status']}；原模型、51保护文件、系统参数未变，本轮归属进程0。最终核查时墙钟 {cleanup['wall_elapsed_seconds']:.3f}秒、任务占用 {cleanup['task_reserved_machine_hours']:.6f}机时；最终小证据交付若再连接learner，另计到delivery.json，仍受60分钟/4机时限制。四个actor解析结束即释放本轮任务，终检按短新任务另计。任务占用不等于厂商账单，用户五台实例仍开启，未自动关机。",'',
'只回传小JSON/CSV、48个预登记训练启动状态切片、图和脚本；checkpoint/replay及全量日志仍在云端。所有正负/缺失项保留，0新拥塞控制TCP、0优化，无生产模型/算法改动。']
(P/'report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(dict(status=results['status'],S1_layer_cosine=layer_stats,cpu_ratio=cpu['ratio'],period_ledger=period_ledger)))
