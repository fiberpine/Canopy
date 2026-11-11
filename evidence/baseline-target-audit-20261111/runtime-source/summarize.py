"""Summarize the closed first batch without reconnecting to any server."""
import csv
import io
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
read=lambda name:json.loads((HERE/name).read_bytes())
old,new=read('stock-target-audit.json'),read('repaired-target-audit.json')
evidence=read('collected-evidence.json');failure=read('admission-failure-review.json')
producers=[row['producer'] for host in failure for row in host['reports']]
assert len(producers)==24
generated=sum(p['generated'] for p in producers);enqueued=sum(p['enqueued'] for p in producers)
training=list(csv.DictReader(io.StringIO(evidence['training_csv'])));last=training[-1]
assert generated==enqueued==int(last['inserted'])==236392
expected=int((generated-12288)*.1)
assert all(int(last[key])==expected for key in ('critic_iterations','actor_updates','target_updates'))
assert int(last['critic_optimizer_ops'])==2*expected
closure=read('closure.json');reg=read('registration.json')
assert all(row['status']=='PASS' for row in closure)
result=dict(status='FAILED_ADMISSION_NO_FINAL_CHECKPOINT',healthy_baseline=False,
    selected_variable='target_bn_consistent only; original single connection',
    query=dict(stock_action=old['action'],repaired_action=new['action'],stock_td=old['td_target'],repaired_td=new['td_target'],
               query_variables_unchanged=old['query_variables_sha256_before']==old['query_variables_sha256_after'] and new['query_variables_sha256_before']==new['query_variables_sha256_after'],
               synthetic_statistics_contract=new['synthetic_statistics_contract']),
    initial_parity=read('initial-tensor-parity.json'),
    observed_ledger=dict(generated=generated,enqueued=enqueued,last_logged_inserted=int(last['inserted']),
        complete_critic_actor_target=expected,critic_optimizer_operations=2*expected,
        final_manifest_missing=True,nominal_sample_budget=225000,overshoot=generated-225000,
        overshoot_reason='Historical5s training-log interval plus5s controller polling and stop tail. Fixed offline with per-actor9375 quotas, not used in this run.'),
    persistence=dict(probe=evidence['probe'],probe_rows=150000,probe_origin_join_actors=24,postfull_live_checkpoint=200009,
                     final_replay_exists=evidence['final_replay_exists'],final_result_exists=evidence['final_result_exists']),
    teardown=dict(initial_port_check_failures=18,initial_passes=6,current_remaining=0,
                  cause='One-shot listener checks still saw18 LISTEN sockets after recorded owned processes exited. Later all listeners disappeared; exact kernel/process release timing not isolated.',
                  coordinator_failure='Failure path forwarded only actor0 producer before raising, then killed learner after3s; no final model/replay/manifest persisted.'),
    runtime_source_commit='b7161b7',runtime_source_manifest='source-manifest-r3.json',
    postrun_offline_repair_commit='735f03f',postrun_repairs_online_verified=False,
    cloud_started_epoch=reg['start_epoch'],cloud_closed_epoch=max(row['epoch'] for row in closure),
    conservative_machine_hours=(max(row['epoch'] for row in closure)-reg['start_epoch'])*5/3600,
    old_files_and_systems_preserved=True,full_training_run=False,frozen_performance_evaluation=False)
(HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
report='''# 目标BN检查与首次真实准入

本轮确认并隔离了目标BN的批上下文依赖，但真实准入在停机清理检查时失败，不能放行完整重训，没有健康基座或网络性能结论。五机收尾通过；实际约23.8分钟、保守约1.99机时。后续追加任务需另登记，不覆盖本轮失败。

## 已取得的证据

- 旧完整终点checkpoint和150k真实回放，96个固定真实输入、8组均衡512批。相同target权重、只换同批其他行，原动作最大跨度0.136217、中位0.044983；修复后动作、logit、TD目标跨度均0。只读查询所有变量哈希不变，旧CP/replay文件不变。
- 独立的内存合成统计契约：4个BN变量的初始复制和tau=.001软更新误差均0；没有执行优化器。真实准入step-0全部106张量与11-10原随机起点逐项相同。
- 唯一训练因素为target_bn_consistent=True，保留单连接、原状态/窗口动作/原效用/MSE/均衡采样/U=.1。
- 已观察到236392条生成、入队、末次日志入库一致，critic/actor/target完整更新各22410、critic优化器44820，符合floor((N-12288)*.1)。最终结果缺失，这不是完整持久化账本。
- 满150k回放复制0.00659s、写盘0.09132s，写盘期间入库150001→150103。模型在step200009满回放后、停止前真实保存。150k探针来源在24个actor逐行连接通过；不是最终回放。
- 四个target统计量在step200009均已变化且有限。critic作用域包含critic2的6个变量，target2有6个重复assign目的地；其训练影响未干预，本轮不同时修改优化器。

## 失败与修复边界

18/24个actor的一次性清理检查仍看到相应监听端口；同次报告的存活归属进程与IPC均为空。稍后五机均没有这些端口或进程。不能仅凭稍后消失就撤销原始失败，也未证明旧bind冲突完全同因。

协调器遇首个actor失败后，没有继续转发其余producer记录给learner，约3秒后清理learner，导致最终CP/replay/result未落盘。step0/100006/200009和150k探针保留云端。

22.5万是旧轮询的停止触发值，尾部实际多11392条；它没有被落实为严格上限。事后修复改成准入专用每actor9375条配额，不丢弃真实转移，总数严格225000。正式训练仍保留原全局约2.4M预算。

另两项事后修复：最多3秒有记录地等待资源释放，保留持久残留失败；异常时先停止生成、收齐producer并给learner有限排空保存时间，整体仍FAILED。修复commit735f03f，离线测试通过，尚未在线验证。实际运行commit来源b7161b7，模块实际LF字节以source-manifest-r3.json为准。

## 旧日志对比与传输

四个弱条件在180万和终点附近训练窗口利用率约96.5%–98.0%，峰值/容量约1.007–1.031。与新连接冻结的差别包含探索、历史和在线权重变化，只作描述线索。

未下载完整模型、回放或逐包。为连接探针来源，只经VPC传338321字节的分host索引元数据；完整探针与原始日志仍在云端。本机接收小JSON/统计/选定96输入的动作值，没有导出完整70维状态。
'''
(HERE/'summary.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(status=result['status'],observed_inserted=generated,complete_updates=expected,closure='PASS')))
