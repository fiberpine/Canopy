"""Read-only legacy echo audit. states.csv logs the PREVIOUS action at the NEXT state."""
import bisect
import argparse
import csv
import json
import math
import os
import pathlib
import time
from collections import defaultdict


def audit(folder):
    control=folder/'control.csv';states=folder/'states.csv'
    if not control.exists() or not states.exists():return None
    native=[];byid=defaultdict(list);times=defaultdict(list)
    for r in csv.DictReader(control.open()):
        item=(float(r['time_us']),int(r['state_id']),int(r['percent']),int(r['cwnd_before']))
        byid[item[1]].append(len(native));times[item[1]].append(item[0]);native.append(item)
    def find(rid,stamp):
        seq=times[rid]
        if not seq:return None
        j=bisect.bisect_left(seq,stamp)
        ks=[k for k in [j-1,j,j+1] if 0<=k<len(seq)]
        k=min(ks,key=lambda k:abs(seq[k]-stamp))
        return byid[rid][k] if abs(seq[k]-stamp)<=1e6 else None
    counts=dict(control_rows=len(native),out_of_range_percent=sum(not 25<=x[2]<=400 for x in native),
                python_state_rows=0,incoming_cwnd_compared=0,incoming_cwnd_mismatch=0,
                outgoing_percent_compared=0,outgoing_percent_mismatch=0,unmatched_state=0)
    examples=[]
    for r in csv.DictReader(states.open()):
        counts['python_state_rows']+=1
        rid=int(r['state_id']);j=find(rid,float(r['time_us']))
        if j is None:counts['unmatched_state']+=1;continue
        n=native[j];counts['incoming_cwnd_compared']+=1
        if float(r['cwnd'])!=n[3]:
            counts['incoming_cwnd_mismatch']+=1
            if len(examples)<8:examples.append(dict(kind='cwnd',state_id=rid,python=r['cwnd'],native=n[3],time_delta_us=n[0]-float(r['time_us'])))
        # The logged percent is action A while this row contains resulting state B.
        if j and native[j-1][1]==(rid+999)%1000:
            counts['outgoing_percent_compared']+=1
            if int(r['percent'])!=native[j-1][2]:
                counts['outgoing_percent_mismatch']+=1
                if len(examples)<8:examples.append(dict(kind='percent',state_id=rid,python_previous=r['percent'],native_previous=native[j-1][2]))
    return dict(path=str(folder),counts=counts,examples=examples,
                alignment='Nearest same stateID within1s; previous native action compared to logged previous percent. Unmatched rows are not silently treated as matches.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scalar-root',type=pathlib.Path,required=True,help='Directory containing orca_control/orca_moving')
    ap.add_argument('--e2-logs',type=pathlib.Path,required=True)
    ap.add_argument('--output-directory',type=pathlib.Path,required=True)
    args=ap.parse_args()
    started=time.time();rows=[]
    root=args.scalar_root
    for arm in ['orca_control','orca_moving']:
        for d in sorted((root/arm/'logs').iterdir()):
            if d.is_dir() and ('twotrack-formal-' in d.name):
                r=audit(d)
                if r:rows.append(r)
    for d in sorted(args.e2_logs.glob('e2-formal-*')):
        r=audit(d)
        if r:rows.append(r)
    out=dict(host=os.uname().nodename,epoch=time.time(),wall_s=time.time()-started,
             mode='READ_ONLY_NO_TCP',logs=rows,
             limitations=['Agreement does not establish all15 fields intact.',
                          'Old logs have no raw wire/native snapshot checksum, so torn-message source cannot be uniquely proved.',
                          'Not an exact actor/cwnd-limited or PRR diagnosis.'])
    dest=args.output_directory;dest.mkdir(exist_ok=True)
    (dest/'legacy-echo-audit.json').write_text(json.dumps(out,indent=2))
    summary={k:sum(r['counts'][k] for r in rows) for k in ['control_rows','incoming_cwnd_compared','incoming_cwnd_mismatch','outgoing_percent_compared','outgoing_percent_mismatch','out_of_range_percent','unmatched_state']}
    print(json.dumps(dict(host=out['host'],logs=len(rows),wall_s=out['wall_s'],summary=summary)))


if __name__=='__main__':main()
