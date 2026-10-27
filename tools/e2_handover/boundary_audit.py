"""Post-run boundary diagnostics. Does not claim exact within-interval TCP mode."""
import csv
import json
import os
import pathlib
import statistics

ROOT=pathlib.Path(os.environ['CANOPY_E2_WORKSPACE']).resolve()


def main():
    status=json.loads((ROOT/'formal-status.json').read_text())
    assert status['stage']=='COMPLETE'
    rows=[]
    for folder in sorted((ROOT/'framework/logs').glob('e2-formal-*')):
        cfg=json.loads((folder/'config.json').read_text())
        raw=list(csv.DictReader((folder/'raw.csv').open()))
        selected=[r for r in raw if float(r['feedback_elapsed_s'])>=10]
        actions=list(csv.DictReader((folder/'e2-actions.csv').open()))
        released=next(r for r in actions if r['released']=='True')
        out=dict(run_id=cfg['run_id'],condition=cfg['condition'],arm=cfg['e2_arm'],
                 repeat=cfg['e2_repeat'],host=os.uname().nodename,
                 first_cwnd=float(raw[0]['cwnd']),first_ssthresh=float(raw[0]['ssthresh']),
                 mean_cwnd=statistics.mean(float(r['cwnd']) for r in selected),
                 below_ssthresh_boundary_fraction=statistics.mean(float(r['cwnd'])<float(r['ssthresh']) for r in selected),
                 mean_packets_out_to_cwnd=statistics.mean(float(r['packets_out'])/float(r['cwnd']) for r in selected),
                 first_released_reference=float(released['max_bw']),
                 reference_to_configured_capacity=float(released['max_bw'])*8/(cfg['capacity_mbps']*1e6),
                 first_released_policy_prediction=float(released['policy_prediction']))
        if (folder/'handover.json').exists():
            event=json.loads((folder/'handover.json').read_text())
            out['handover_reference_before']=event['max_bw_before']
            out['reference_after_to_before_ratio']=out['first_released_reference']/event['max_bw_before'] if event['max_bw_before'] else None
        rows.append(out)
    result=dict(mode='READ_ONLY_POST_RUN',rows=rows,
                scope='Boundary samples after10s on feedback clock. Below ssthresh is not proof of pathology; occupancy ratio is not a complete cwnd-limited test. No acknowledged-packet counts inferred from RTT sample cnt.')
    (ROOT/'boundary-state-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
