import datetime,json,pathlib
E=pathlib.Path(__file__).parent;P=E/'report';P.mkdir(exist_ok=True)
g=json.loads((E/'A-gates.json').read_text());receipt=json.loads((E/'A-training-receipt.json').read_text());l=receipt['ledger']
audit=json.loads(json.loads((E/'A-audit-learner.json').read_text())['output']);cal=audit['calibration_summary'];actual=audit['learner'];reg=json.loads((E/'A-registration.json').read_text())
def stamp(t):return datetime.datetime.fromtimestamp(t,datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def f(x,n=4):return '不可用' if x is None else ('%.'+str(n)+'f')%x
lines=['# 共周期 A 阶段执行报告','',f"状态：**{g['status']}**。正式冻结评测 {g['completed']}/264 次。",'']
if g['status']=='FAIL_A':
 lines+=['A 未通过原定可行性门槛，本候选止于 A。B0 与 C 未启动。没有改奖励、追加训练、换 seed、删除 20ms 或加入兜底；需要用户决定新的研究分支。失败只否定这一候选，不证明条件窗口路线或双动作强化学习不可能。','']
elif g['status']=='PASS_A':lines+=['A 可行性筛查通过；仅允许进入 B0 的确定性价值评估与独立校准，不代表整个双动作目标通过。B0/C 的最终状态另行补充。','']
elif g['status']=='PARTIAL_A_WINDOW_AMBIGUITY':lines+=['full 与 trim10 的门槛结论不同。A0 登记固定了数值界，但没有明确 A 主判窗口；因此保留 PARTIAL，不事后挑选通过窗口，也不自动放行 B0。','']
else:lines+=['评测仍在进行；局部结果的条件子集不同，不能作为总体组间比较或阶段判定。','']
lines+=['## 范围与身份','',
f"本轮 A 启动：{stamp(reg['start_epoch'])}；登记截止：{stamp(reg['deadline_epoch'])}。用户明确授权合理使用五机、不设总时长上限，并要求通过后继续；本次具体 A 仍采用四小时封顶、固定训练和评测清单。",
'',f"实际训练 commit：`{reg['code_commit']}`；12 项实际运行字节 SHA 见 `../A-registration.json`。独立源码在 `@LOCAL_WORKSPACE@/innovations/common-period-cluster-20261102`；本机后来仅改统计、未来保存逻辑和文档，归档 commit 不冒充训练版本。",'',
'云端独立目录 `/home/experiment/orca-common-period-cluster-20261102`。原 baseline、V31 源模型与每机 51 个保护文件保留。A 使用 V31 step157704 窗口/critic/BN 初始化，fresh Adam、replay 和计数；不是随机从零训练。实际为 1 learner＋24 actor，四台机器各六个不同条件，没有扩大到 48 actor。', '',
'24 个训练条件和 12 个开发条件来自实际模板，包含真实/动态/噪声及合成条件；身份、内容 SHA 见登记清单，不能套用早期六带宽×四 RTT 的旧网格。所有对照均是原有旧奖励训练权重的外部可行性参照，不是本轮新奖励的单变量因果消融。','',
'## 准入与训练账本','',
'首次 24×30 秒容量运行网络完成，但遗漏平均 RTT 审计字段，因此没有判通过。补字段后新登记的 24×30 秒运行全部通过：16,795 条、720.222811 秒、FIFO 峰值 13、动作/时钟/奖励/连接错误 0。静态更新吞吐与短协议试验分开测量，短测尚未产生 warmup 更新信用，不能声称验证了持续训练无积压。','',
f"A 的 264 个校准/训练 TCP episode 全部完成。每 actor 实际总暴露 {min(actual['exposure_by_actor']):.6f}–{max(actual['exposure_by_actor']):.6f} 秒，合计 {sum(actual['exposure_by_actor']):.6f} 秒；校准暴露 {sum(actual['calibration_exposure_by_actor']):.6f} 秒。180 秒训练连接重置，结束只用同连接真实后继，不跨 reset bootstrap。",'',
'| 项目 | 实际值 |','|---|---:|',
f"| 隔离校准区间 | {l['calibration_intervals']:,} |",f"| 有效训练入库 | {l['inserted']:,} |",f"| 真实训练物理暴露 | {l['real_training_seconds']:.6f} s |",f"| warmup 结束累计训练暴露 | {l['warmup_seconds']:.6f} s |",f"| 完整 critic / N / target | {l['critic_iterations']:,} / {l['N_updates']:,} / {l['target_updates']:,} |",f"| critic 优化器操作 | {l['critic_optimizer_ops']:,} |",f"| actor 更新 | {l['actor_updates']:,} |",f"| 最终信用欠额 | {l['issued']-l['critic_iterations']} |",f"| actor 回放采样贡献最大差 | {max(l['sampled_by_actor'])-min(l['sampled_by_actor'])} |",f"| 最终每 actor×周期留存最小经验数 | {min(v for r in l['retained_actor_period'] for v in r)} |",f"| FIFO / 更新欠额峰值 | {receipt['statistics']['max_backlog']} / {receipt['statistics']['max_debt']} |",'',
f"按真实入库时间发信用，无墙钟补发；校准信用为 0。前 5,000 critic 更新冻结 actor 含 BN，其后仍须每 actor×周期至少 128 留存样本，故 actor 实际稍晚开启；每两次 critic 一次 actor。全程真实 U={l['critic_iterations']/l['inserted']:.6f}，不能说固定每决策 U。训练结束后排空所有信用再保存。学习端含保存/传输的真实更新速度低于静态测量，产生了上述短暂欠额；最终没有持续欠额。",'',
f"逐 episode 独立奖励重建最大误差 {receipt['reward_error_max']:.3g}；实际 dt 折扣、原 int(100·4^a) 映射、计划周期和受控 native 采样数审计通过。最终 `step-711285.metadata.json` 与结果账本逐项一致。仅凭 CLOSED 状态文件中的旧缓存数字会差 17 更新，最终以结果/metadata 为准；后续源码已修正该状态刷新。",'',
'## 奖励与一次校准','',
'奖励 v1 全局尺度固定 6,000,000 B/s；canonical SHA `4dfd78dec9590fedf790d1be879e1e0d7ea4a2c1bd5affc8d930e82e6af75cef`。零 ACK RTT 样本仍回退边界 sRTT；149 维状态不改。无每决策成本、teacher、QC、fallback、隐藏 20ms 观测或部署容量查表。','',
f"120 个隔离块估计 c={cal['params']['c']:.9f}、K={cal['params']['K']:.9f}。偶数 actor 拟合、奇数 actor 独立复检；一次全局偏置 +{cal['global_bias_offset']:.9f}。拟合残差均值 {cal['fit']['residual_mean']:.3g}，独立复检均值 {cal['independent_check']['residual_mean']:.6f}、标准差 {cal['independent_check']['residual_std']:.6f}。复检不是 A 停止条件，未为过关重复补偏置。",'',
f"冻结尾项预测绝对均值/最大值：拟合 {cal['fit']['tail_abs_mean']:.6f}/{cal['fit']['tail_abs_max']:.6f}，复检 {cal['independent_check']['tail_abs_mean']:.6f}/{cal['independent_check']['tail_abs_max']:.6f}。这是预测尾项幅度，不是真实尾项误差界。大量后缀点相互重叠，不能当成数万独立连接、不能据此宣称 critic 真值已校准。n0 逐周期为 `{json.dumps(cal['params']['n0'])}`。",'',
'[训练梯度与饱和诊断](figures/training-gradients.png)：每 100 次完整更新，RL 与预激活正则梯度分开记录。图中的变化均衡 minibatch 不是冻结部署健康门槛，也不证明所有条件无饱和。','',
'## 冻结评测与门槛','',
'预先冻结的随机顺序：11 组×12 条件×2 次独立 60 秒连接，共 264 次；每台机器同时仅一流。A 强制 20/40/80/120/160ms，另有独立训练 fixed80/120/160、正常 CUBIC 启动 H0、官方 Orca 与 CUBIC。优化/探索关闭，变量摘要前后一致；Python 逐步日志关闭，推理/实际暴露仅作汇总计数。CPU 来自预定约 25–50 秒采样，含可归属的重挂 Mahimahi，不把调用下降等同于 CPU 下降。','',
'所有指标均按条件等权、条件内重复等权。P95/P99 是各连接包延迟分位数的期望，不是合并包分位数。逐条件、重复差和前 10 秒数据见 `../A-condition-metrics.csv`；完整门槛见 `../A-gates.json`。A0 没有明确 A 主判窗口，因此 full 与 trim10 独立套用原数值界；同结论才判 PASS/FAIL，冲突为 PARTIAL，不事后选有利口径。启动窗口只报告，不添加新阈值。','',
'| 组别 | trim10 利用率 % | 平均排队 ms | P95 ms | P99 ms | 调用/s | 总归属 CPU % 单核 |','|---|---:|---:|---:|---:|---:|---:|']
for r in g['aggregates']:
 if r['window']=='trim10':lines.append('| '+r['group']+' | '+f(r['utilization']*100)+' | '+f(r['queue_mean_ms'])+' | '+f(r['queue_p95_ms'])+' | '+f(r['queue_p99_ms'])+' | '+f(r['policy_calls_per_exposure_s'])+' | '+f(r['all_owned_cpu_percent_one_core'])+' |')
lines+=['',f"动作健康通过：{g.get('action_health_pass')}；窗口门槛：`{json.dumps(g.get('window_pass'))}`。完整失败条目 {len(g['failed_checks'])} 项。",'']
if g['failed_checks']:
 lines+=['失败条目前 12 项（全部条目保存在 JSON，未删去负面条件）：','']
 for c in g['failed_checks'][:12]:lines.append(f"- {c['window']} / {c['name']} / {c.get('group',c.get('period_ms',''))} / {c.get('condition','总体')}：实际 {f(c['actual'])}，界 {f(c['limit'])}。")
lines+=['','## 工程修正与证据保留','',
'训练前修正了平均 RTT 审计开关和首次反馈等待。评测准备中依赖路径/脚本匹配失败均发生在 TCP 启动前，未额外训练。评测在线汇总曾把丢包数除以到达字节；收尾工具按原包日志统一重算为丢包数/到达包数，保留在线原值，并要求其他网络指标逐项完全相同，不重跑 TCP。收尾是否已完成以 `eval-final-audit-*.json` 回执为准。','',
'曾预置本轮 checkpoint 重名保护 watcher；本次没有触发文件移动，最终保存已验证为完整更新状态。默认 Saver 只保留最近五组模型，实际剩余 checkpoint 文件清单见 cleanup 中的 inventory；不能把 metadata 存在说成 tensor 文件仍全部存在。后续本地源码已改独立 final 名称和有限实验全保留设置，没有热改本轮训练。异常关闭路径仍有未排空最后异步消息的已知限制；本轮正常完成路径无此缺口。','',
'## 留存与后续','',
'最终模型、优化器、replay 与原始日志留云端 `A/learner/checkpoints/step-711285` 及各机 A/evaluation 目录。评测所需的四个模型文件仅从 learner 经已核验内网复制给四台 actor，不下载 replay。本机只复制小指标、审计、图、实际 Python 运行源与增量。权重/优化器/replay 保存不等于 TCP/history/max_bw/探索 RNG/队列现场可无缝恢复。','',
'当前只有一个训练 seed、每条件两次评测；不称多 seed、盲测泛化或论文级创新通过。A 只有学习的周期条件窗口族，周期训练使用预登记探索、冻结时强制指定；尚未执行 B0 价值选择器，不能把这一步称为最终两个动作的收益已证实。C 当前没有通过曝光审核的盲测 trace，已曝光 heldout36-long 与 ATT 不能重新包装为未见测试。','']
cleanup=E/'cleanup.json'
if (E/'observation-diagnosis.json').exists():
 obs=json.loads((E/'observation-diagnosis.json').read_text());training=[r for r in obs if r['kind']=='training'];count=sum(r['decisions'] for r in training);empty=sum(r.get('empty_rtt',0) for r in training);delivered=sum(r.get('empty_rtt_with_delivery',0) for r in training)
 v=json.loads((E/'offline-value-diagnosis.json').read_text())
 lines+=['## 有限离线复盘','',f"训练期 {int(empty):,}/{int(count):,}（{100*empty/count:.4f}%）受控区间无 RTT 样本，其中 {int(delivered)} 个仍有交付；回退规则保留。零观测并非本轮全部负面结果的已证明唯一原因。",'',f"最终模型只读查询预先固定的 3,840 个保留状态（每训练 actor×行为周期 32 条），全部变量前后相同、0 优化/0 新 TCP。同状态五周期窗口输出极差均值 {v['conditional_action_range_mean']:.6f}。无成本 Q 均值 argmax 在 {100*v['preferences'][0]['proportion_by_period']['160']:.3f}% 状态偏好 160ms；这是未经过 B0 确定性评估/独立校准的离策略查询，不能作为真实价值、切换收益或部署效果证明。",'', '8 个健康失败格都来自三个 R20 条件的负向窗口饱和；训练末期混合样本的低整体饱和率没有保证逐部署条件通过。后续是否追加 seed、使用每周期独立窗口输出层，或单独验证迟醒，必须由用户根据原方案第6.1节决定；本轮没有自动执行这些变更。','']
if (E/'client-cleanup.json').exists():
 records=json.loads((E/'client-cleanup.json').read_text());cpu=sum(sum(r['cpu_times'].get(k,0) for k in ['user','system']) for x in records for r in x['data']);n=sum(len(x['data']) for x in records)
 lines+=[f"收尾首检发现 {n} 个标量参照 client/sudo 进程在连接结束后仍存活（整个生命周期 user+system 合计 {cpu:.6f} CPU 秒）。已逐项核对 PID 创建时间、完整命令、专属运行路径和登记端口后清理，原失败收尾回执保留，随后五机复检。它们是清理遗漏，不计为已完成网络 run 的失败；本轮 CPU 只按原有持久进程快照计量，不据小于此类开销的差异宣称严格优势。未来 runner 的端口归属匹配已在控制源码修正，未重跑本轮。",'']
if cleanup.exists():
 c=json.loads(cleanup.read_text());lines+=[f"五机最终归属/IPC/保护检查：**{c['status']}**；细节见 `../cleanup.json`。",'']
(P/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
print('Report updated: '+g['status'])
