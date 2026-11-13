"""Recompute the interrupted startup validation using saved evidence only."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median


def window(row, name='trim10'):
    return next(v for v in row['windows'] if v['window']==name)


def cell_checks(rows, gates):
    rtt=rows[0]['job']['condition']['base_rtt_ms']
    values={k:median(window(r)[k] for r in rows) for k in
            ('utilization','queue_mean_ms','queue_p95_ms','drop_fraction')}
    checks=dict(utilization=values['utilization']>=gates['cell_utilization'],
        mean_queue=values['queue_mean_ms']<=gates['queue_rtt_factor']*rtt+gates['queue_margin_ms'],
        p95=values['queue_p95_ms']<=gates['p95_rtt_factor']*rtt+gates['p95_margin_ms'],
        trim_drop=values['drop_fraction']<=gates['trim_drop'],
        tail=median(r['tail_utilization'] for r in rows)>=gates['tail_utilization'],
        no_catastrophe=all(window(r)['utilization']>=gates['catastrophe_utilization'] for r in rows),
        post_handoff_drop=median(r['post_handoff_drop_fraction'] for r in rows)<=gates['post_handoff_drop'])
    return dict(medians=values,checks=checks,pass_observed_subset=all(checks.values()))


def analyze(folder):
    load=lambda n:json.loads((folder/n).read_bytes())
    screen=load('screen-latest.json')
    original=load('screen-registration.json')['jobs']
    rows=[r for h in screen for r in h['rows']]
    assert all(h['finished'] for h in screen) and len(rows)==40
    expected={j['run']:j for j in original}
    assert len(expected)==40 and all(r['job']==expected[r['job']['run']] for r in rows)
    assert all(r['status']=='COMPLETED' for r in rows)
    gates=load('screen-assessment.json')['original_health_gates']
    cells=[];grouped=defaultdict(list)
    for r in rows:grouped[(r['job']['group'],r['job']['startup'],r['job']['condition']['condition'])].append(r)
    for (group,mode,name),values in sorted(grouped.items()):
        cells.append(dict(group=group,startup=mode,condition=name,repeats=len(values),
            utilizations=[window(r)['utilization'] for r in sorted(values,key=lambda r:r['job']['repeat'])],
            startup_utilization_median=median(window(r,'startup0_10')['utilization'] for r in values),
            startup_drop_median=median(window(r,'startup0_10')['drop_fraction'] for r in values),
            **cell_checks(values,gates)))
    assert not all(v['pass_observed_subset'] for v in cells if v['group']=='onlinebn-final6m' and v['startup']=='first-feedback')
    natural_plan=load('natural-bound-plan.json')
    assert len(natural_plan)==12 and len({j['run'] for j in natural_plan})==12
    natural_expected={j['run']:j for j in natural_plan}
    observed=[r for phase in ('natural-protocol','natural-screen')
              for h in load(phase+'-latest.json') for r in h['rows']]
    assert len({r['job']['run'] for r in observed})==len(observed)
    assert all(r['job']==natural_expected[r['job']['run']] for r in observed)
    complete={r['job']['run']:r for r in observed if r['status']=='COMPLETED'}
    assert len(complete)==4
    artifacts={r['run']:r for h in load('all-startup-feedback-audit.json')['rows'] for r in h['records']}
    statuses=[]
    for j in natural_plan:
        a=artifacts[j['run']]
        status='COMPLETED' if j['run'] in complete else 'INTERRUPTED_PARTIAL_LOG' if a['started'] else 'NOT_STARTED'
        statuses.append(dict(run=j['run'],condition=j['condition']['condition'],repeat=j['repeat'],host=j['host'],
            status=status,last_packet_log_elapsed_s=a.get('last_packet_log_elapsed_s')))
    counts=Counter(r['status'] for r in statuses)
    assert counts==dict(COMPLETED=4,INTERRUPTED_PARTIAL_LOG=2,NOT_STARTED=6)
    audit=load('natural-completed-source-and-handoff-audit.json')
    assert audit['status']=='PASS_COMPLETED_SUBSET' and audit['complete_runs']==4
    proofs={r['job']['run']:r for h in audit['proof'] for r in h['records']}
    assert set(proofs)==set(complete)
    natural_cells=[]
    for name in sorted({j['condition']['condition'] for j in natural_plan}):
        rs=sorted([r for r in complete.values() if r['job']['condition']['condition']==name],key=lambda r:r['job']['repeat'])
        legacy=next(v for v in cells if v['group']=='onlinebn-final6m' and v['startup']=='legacy' and v['condition']==name)
        startup=median(window(r,'startup0_10')['utilization'] for r in rs)
        natural_cells.append(dict(condition=name,completed=len(rs),registered=sum(j['condition']['condition']==name for j in natural_plan),
            descriptive_only=True,startup_median=startup,startup_minus_earlier_legacy=startup-legacy['startup_utilization_median'],
            startup_descriptive_gate=startup-legacy['startup_utilization_median']>=-.05,**cell_checks(rs,gates)))
    details=[]
    for name,r in sorted(complete.items()):
        p=proofs[name];a=artifacts[name]
        assert 0<=p['handoff_s']<=10 and p['engineering_pass']
        details.append(dict(run=name,condition=r['job']['condition']['condition'],repeat=r['job']['repeat'],
            metrics=window(r),handoff_s=p['handoff_s'],startup=p['startup'],first10_controls=a['first10_controls']))
    closure=load('closure-post-interruption.json')
    assert len(closure)==2 and all(x['status']=='PASS_CURRENT_CLOSURE' for x in closure)
    partial=load('partial-log-audit.json')
    assert partial['registration']['window_start_s']==10 and partial['registration']['window_end_s']==49
    partial_values=[]
    for row in partial['rows']:
        value=window(row)
        assert value['complete_window'] and value['start_s']==10 and value['end_s']==49
        assert next(s for s in statuses if s['run']==row['run'])['status']=='INTERRUPTED_PARTIAL_LOG'
        partial_values.append(dict(run=row['run'],descriptive_only=True,metrics=value,
            full60_status='INCOMPLETE',observed_segment_below50=value['utilization']<.5))
    result=dict(status='INCOMPLETE_INTERRUPTED',performance_confirmation_go=False,training_go=False,
        registered_natural_runs=12,counts=dict(counts),natural_run_statuses=statuses,
        screen40_status='FAIL',screen40_execution_errors=0,screen40_cells=cells,
        natural_cells_descriptive_only=natural_cells,completed_natural_runs=details,
        partial_log_descriptive_window=partial_values,
        original_seed3_legacy_FAIL_unchanged=True,independent_full96_started=False,seed7_started=False,
        current_two_host_closure='PASS',new_experiments_on_recovery=0,
        limits=['Four completed natural runs cannot estimate a failure rate or establish all-grid health.',
                'Two partial runs are not 60-second measurements and are never imputed as zero or pass.',
                'Current post-reboot closure cannot prove cleanup before the interruption.',
                'Earlier legacy/first-feedback and adaptive natural observations are not concurrent causal controls.'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot-dir',type=Path)
    args=parser.parse_args()
    result=analyze(args.folder)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.plot_dir:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        args.plot_dir.mkdir(exist_ok=True,parents=True)
        fig,axes=plt.subplots(1,3,figsize=(12,4.3),sharey=True)
        modes=('legacy','first-feedback','natural-cubic')
        colors=('#586a7a','#d2761b','#246b9b')
        for ax,name in zip(axes,('G12-R200','G48-R200','G72-R20')):
            for i,(mode,color) in enumerate(zip(modes,colors)):
                if mode=='natural-cubic':
                    rs=[r['metrics']['utilization'] for r in result['completed_natural_runs'] if r['condition']==name]
                    cell=next(c for c in result['natural_cells_descriptive_only'] if c['condition']==name)
                    label='%d/%d done'%(cell['completed'],cell['registered'])
                else:
                    cell=next(c for c in result['screen40_cells'] if c['group']=='onlinebn-final6m' and c['condition']==name and c['startup']==mode)
                    rs=cell['utilizations'];label='n=%d'%len(rs)
                positions=[i+(k-(len(rs)-1)/2)*.055 for k in range(len(rs))]
                ax.scatter(positions,[v*100 for v in rs],c=color,s=46,alpha=.85,zorder=3,
                           marker='o' if mode!='natural-cubic' else 'D')
                ax.plot([i-.16,i+.16],[median(rs)*100]*2,color=color,linewidth=2.5)
                ax.text(i,-12,label,ha='center',fontsize=9)
            ax.axhline(90,color='#59913c',linestyle='--',linewidth=1)
            ax.axhline(50,color='#a94c4c',linestyle=':',linewidth=1)
            ax.set_xticks(range(3));ax.set_xticklabels(['Legacy','Early','Natural'])
            ax.set_title(name);ax.set_ylim(-17,108);ax.grid(axis='y',alpha=.17)
        axes[0].set_ylabel('Trim10 utilization (%)')
        fig.suptitle('Same frozen seed3 policy: startup sensitivity',fontsize=14)
        fig.text(.5,.015,'Natural: incomplete adaptive arm (4/12); earlier arms are descriptive references. No all-grid health claim.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.94))
        for suffix in ('png','svg'):fig.savefig(args.plot_dir/('startup-sensitivity.'+suffix),dpi=180)
        plt.close(fig)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],screen40_status=result['screen40_status'],
                         performance_confirmation_go=False,training_go=False)))


if __name__=='__main__':main()
