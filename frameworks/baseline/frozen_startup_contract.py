"""Post-run checks from the validated natural-handoff protocol; no actuation."""
import csv
import json
from pathlib import Path


def verify(folder, seconds=60, maximum_handoff_seconds=10):
    folder=Path(folder)
    result=json.loads((folder/'result.json').read_text())
    assert result['status']=='COMPLETED'
    before=result['policy_sha256_before'];after=result['policy_sha256_after']
    assert isinstance(before,str) and len(before)==64 and before==after
    assert all(result['counters'][k]==0 for k in
        ('critic_iterations','critic_optimizer_ops','actor_updates','target_updates','exploration_calls'))
    lines=[x for x in (folder/'server.log').read_text().splitlines() if 'ORCA_STARTUP_HANDOFF ' in x]
    assert len(lines)==1,'Exactly one audited RL handoff required'
    fields=dict(x.split('=',1) for x in lines[0].split('ORCA_STARTUP_HANDOFF ',1)[1].split())
    assert fields['mode']=='natural-cubic' and fields['scheme']=='cubic'
    assert int(fields['assist_writes'])==0 and int(fields['ssthresh'])<int(fields['cwnd'])
    with (folder/'control.csv').open() as stream:controls=list(csv.DictReader(stream))
    assert controls
    for row in controls:
        expected=max(4,min(2147483647,int(row['percent'])*int(row['cwnd_before'])//100))
        assert int(row['cwnd_requested'])==expected and row['state_id']==row['action_id'] and int(row['return_code'])==0
    init=base=None;last=0
    with (folder/'down.log').open() as stream:
        for line in stream:
            if line.startswith('# init timestamp:'):init=float(line.split(':',1)[1])
            elif line.startswith('# base timestamp:'):base=float(line.split(':',1)[1])
            elif line.startswith('#'):continue
            elif base is not None:
                parts=line.split()
                if len(parts)>=3:last=max(last,(float(parts[0])-base)/1000)
    assert init is not None and base is not None and last>=seconds-.01,'Incomplete packet-clock window'
    handoff=(float(controls[0]['time_us'])/1000-init-base)/1000
    assert 0<=handoff<=maximum_handoff_seconds,'Late handoff cannot pass as policy-controlled evaluation'
    proof=dict(status='PASS',handoff_s=handoff,startup=fields,controls=len(controls),policy_sha256=before,
               complete_packet_seconds=seconds,no_optimization_or_exploration=True)
    (folder/'startup-contract.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof
