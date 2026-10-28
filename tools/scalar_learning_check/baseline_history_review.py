"""Recompute archived baseline recovery evidence. Stdlib, read-only, no experiments."""
import argparse,json,pathlib,statistics,hashlib


def analyze(root):
    root=pathlib.Path(root);audit=root/'evidence/baseline/audit'
    paths=[audit/'balanced-trajectory-329992.json']+[audit/('formal24-grid24-seed3-balanced-u010-policy-diagnosis-%d.json'%n) for n in (500010,2412694)]
    trajectory,mid,last=[json.loads(p.read_text()) for p in paths]
    rows=trajectory['rows'];assert len(rows)==24
    def diagnosis(record):
        rows=record['rows'];assert len(rows)==24
        return dict(checkpoint_sample_step=record['checkpoint_sample_step'],probe_states=sum(r['states'] for r in rows),
                    probe_mean_exact_minus_one=sum(r['predicted_action_mean']==-1 for r in rows),
                    probe_mean_near_minus_one=sum(r['predicted_action_mean']<=-.99 for r in rows),
                    classes_with_any_probe_saturation=sum(r['min_saturation']>0 or r['max_saturation']>0 for r in rows),
                    critic_Q_class_mean_range=[min(r['critic_Q_mean'] for r in rows),max(r['critic_Q_mean'] for r in rows)],
                    recent_5s_training_utilization_equal_condition=statistics.mean(r['observed_network']['utilization'] for r in rows),
                    scope=record['scope'],rows=rows)
    completion=json.loads((audit/'balanced-formal-completion.json').read_text())
    return dict(status='READ_ONLY_ARCHIVED_HISTORY_REVIEW',trajectory329992=dict(
                    negative_mean_le_minus95=sum(r['recent']['action_mean']<=-.95 for r in rows),
                    positive_mean_ge_plus95=sum(r['recent']['action_mean']>=.95 for r in rows),
                    positive_mean=sum(r['recent']['action_mean']>0 for r in rows),
                    mean_cwnd_ge_million=sum(r['recent']['cwnd_mean']>=1e6 for r in rows),
                    mean_cwnd_intmax=sum(r['recent']['cwnd_mean']==2147483647 for r in rows),
                    negative_cwnd_mean_range=[min(r['recent']['cwnd_mean'] for r in rows if r['recent']['action_mean']<=-.95),max(r['recent']['cwnd_mean'] for r in rows if r['recent']['action_mean']<=-.95)],
                    positive_unsaturated=[r for r in rows if 0<r['recent']['action_mean']<.95],scope=trajectory['scope']),
                diagnosis500010=diagnosis(mid),diagnosis2412694=diagnosis(last),
                historical_complete=dict(inserted=completion['final']['inserted'],wall_s=completion['wall_s'],counters=completion['final']['counters']),
                L1_budget_fraction=28513/1200000.,L1_relative_to_baseline_budget=28513/2400000.,
                moving_actor_loss_already_in_baseline=True,
                code_evidence=['frameworks/baseline/distributed.py: backend moving_stats installs moving_stats_backend',
                               'frameworks/baseline/moving_stats_backend.py: build_actor_train_op uses _moving_action'],
                sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                limitations=['One selected successful run demonstrates possible recovery, not universal/necessary recovery or that L1 would recover.',
                             'Intermediate live5s windows and fixed-state checkpoint probes are not frozen performance evaluations.',
                             'Archived base is the Canopy agent_v2 engineering recipe, not full official Orca paper reproduction.',
                             'Saturation can generate experience; its necessity, benefit and cause of recovery are not established by this trajectory.',
                             'Shared host buffer influence does not establish the unique bottleneck or bound all network harm.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output);assert not out.exists();out.write_text(json.dumps(analyze(a.root),indent=2)+'\n')
