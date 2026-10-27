"""Recompute published zero-update gates without TF/TCP or original servers."""
import argparse
import csv
import json
import pathlib


def analyze(directory):
    rows=json.loads((directory/'formal-results.json').read_text())
    expected=json.loads((directory/'final-summary.json').read_text())
    assert len(rows)==32 and len({r['run_id'] for r in rows})==32
    with (directory/'evaluation-runs.csv').open(newline='') as stream:
        windows=list(csv.DictReader(stream))
    assert len(windows)==64
    index={(r['run_id'],r['window']):r for r in windows}
    assert len(index)==64
    for row in rows:
        assert all(v==0 for v in row['optimizer_counters'].values())
        assert row['tensors']['status']=='PASS_ALL_TENSORS_UNCHANGED'
        assert row['tensors']['variables']==96 and not row['tensors']['changed']
        assert row['ipc']['status']=='PASS_BOTH_DIRECTIONS'
        assert row['ipc']['received_crc_or_sequence_failures']==0
        assert row['million_window_pass']==(max(row['observed_max_cwnd'],row['requested_max_cwnd'])<1e6)
        assert row['four_bdp_queue_time_pass']==(row['boundary_weighted_fraction_above_bound']<.01)
        for window in row['packet_windows']:
            tab=index[(row['run_id'],window['window'])]
            for key in ['utilization','departure_mbps','queue_mean_ms','queue_p95_ms','queue_p99_ms','drop_fraction']:
                assert float(tab[key])==window[key],(row['run_id'],key)
    groups={}
    for mode in ['upstream','small_uniform']:
        group=[r for r in rows if r['mode']==mode]
        assert len(group)==16
        groups[mode]=dict(runs=len(group),million_failures=sum(not r['million_window_pass'] for r in group),
            duration_failures=sum(not r['four_bdp_queue_time_pass'] for r in group),
            transitions=sum(r['transitions'] for r in group),
            matched_ipc_frames=sum(r['ipc']['matched_frames'] for r in group))
    assert groups==expected['groups']
    return dict(status='PASS_PUBLISHED_EVIDENCE_RECOMPUTED',groups=groups,
        exact_full_trim_rows=64,optimizer_updates=0,new_TCP_runs=0,
        research_conclusion='NO_GO_LONG_TRAINING; small initialization is insufficient')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2]/'evidence/scalar-ipc-init-20261027')
    args=parser.parse_args()
    print(json.dumps(analyze(args.evidence),indent=2))


if __name__=='__main__':main()
