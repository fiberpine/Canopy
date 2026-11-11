"""One frozen flow per actor host, with owned cleanup and a fixed task list."""
import csv
import numpy as np
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code/runtime/_deps'))
import psutil
import node_control as node
from common_period_metrics import network
from mahimahi_metrics import mm


def atomic(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def run(job, registration):
    prefix=job['checkpoint']
    hashes=lambda:{suffix:hashlib.sha256(pathlib.Path(prefix+suffix).read_bytes()).hexdigest()
                   for suffix in ('.index','.data-00000-of-00001')}
    assert hashes()==job['model_sha256']
    name=job['run']
    config=dict(job['condition'],method='canopy',run_id=name,group=job['group'],seconds=60,
                port=job['port'],seed=3,checkpoint=prefix,backend='moving_stats',code_commit='source-manifest.json',
                control_period_ms=20,history_frames=10,use_normalizer=False,no_training=True,
                absolute_deadline_epoch=registration['run_cutoff']-15,timeout_s=105,
                stop_file=str(ROOT/'logs'/(name+'-stop.json')),server_binary=registration['native'])
    path=ROOT/'configs'/(name+'.json')
    with path.open('x') as stream:json.dump(config,stream)
    work=ROOT/'logs'/(name+'-work');work.mkdir()
    started=time.monotonic()
    with (ROOT/'logs'/(name+'.stdout')).open('wb') as stream:
        process=subprocess.Popen([sys.executable,'-B',str(ROOT/'capture_frozen.py'),'--config',str(path)],
                    cwd=str(work),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        node.dump(node.own_path(name),[dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),role='evaluation',index=0)])
        try:
            code=process.wait(timeout=min(105,registration['run_cutoff']-time.time()-5))
            folder=ROOT/'logs'/name
            result=json.loads((folder/'result.json').read_text())
            assert code==0 and result['status']=='COMPLETED',result.get('reason')
            assert result['policy_sha256_before']==result['policy_sha256_after']
            assert hashes()==job['model_sha256']
            assert all(result['counters'][key]==0 for key in ('critic_iterations','critic_optimizer_ops','actor_updates','target_updates','exploration_calls'))
            windows,bins=network(folder/'down.log',60,with_seconds=True)
            assert all(window['complete_window'] for window in windows)
            with (folder/'control.csv').open() as stream:controls=list(csv.DictReader(stream))
            assert controls and all(row['state_id']==row['action_id'] and int(row['return_code'])==0 for row in controls)
            periods=np.diff([float(row['time_us']) for row in controls])/1000
            period_stats=dict(p50=float(np.percentile(periods,50)),p95=float(np.percentile(periods,95)),p99=float(np.percentile(periods,99)),max=float(np.max(periods))) if len(periods) else {}
            _,epoch_base,_=mm(folder/'down.log',60,retain_delays=False)
            handoff=(float(controls[0]['time_us'])/1000-epoch_base)/1000
            assert 0<=handoff<60
            arrival=dropped=0
            base=None
            with (folder/'down.log').open() as stream:
                for line in stream:
                    if line.startswith('# base timestamp:'):base=float(line.split(':')[1]);continue
                    if line.startswith('#') or base is None:continue
                    fields=line.split()
                    if len(fields)<3:continue
                    second=(float(fields[0])-base)/1000
                    if handoff<=second<60:
                        if fields[1]=='+':arrival+=1
                        elif fields[1]=='d':dropped+=int(fields[2])
            tail=bins[50:60]
            return dict(job=job,status='COMPLETED',windows=windows,
                        tail_utilization=sum(row['departure_bytes'] for row in tail)/sum(row['service_bytes'] for row in tail),
                        handoff_s=handoff,post_handoff_drop_fraction=dropped/max(1,arrival),
                        model_sha256=hashes(),policy_sha256=result['policy_sha256_after'],
                        valid=result['counters']['valid'],control_period_ms=period_stats,wall_s=time.monotonic()-started)
        finally:
            node.cleanup(name)


def main():
    phase=sys.argv[1]
    registration=json.loads((ROOT/'registration.json').read_text())
    jobs=json.loads((ROOT/(phase+'-plan.json')).read_text())
    rows=[]
    for job in jobs:
        if (ROOT/'STOP').exists() or time.time()+110>=registration['run_cutoff']:
            result=dict(job=job,status='NOT_RUN_BUDGET')
        else:
            try:result=run(job,registration)
            except Exception as error:result=dict(job=job,status='EXECUTION_ERROR',reason=repr(error),traceback=traceback.format_exc())
        rows.append(result)
        atomic(ROOT/(phase+'-progress.json'),rows)
    atomic(ROOT/(phase+'-result.json'),rows)


if __name__=='__main__':
    main()
