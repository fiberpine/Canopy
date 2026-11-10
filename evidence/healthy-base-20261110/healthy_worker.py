"""Run one authorized stage of the fresh single-action baseline experiment."""
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import traceback

F = pathlib.Path(__file__).resolve().parent
R = F.parent
sys.path.insert(0,str(F/'code/runtime/_deps'))
import node_control as node
from common_period_metrics import network

REG = node.load(R/'registration.json')


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data))
    temp.replace(path)


def train(phase):
    import cluster_control

    cfg = dict(REG['training_config'])
    if phase=='smoke':
        run='healthy-smoke'
        cfg.update(run_id=run,model_id=run,stop_file=str(F/'logs'/(run+'-stop.json')),
                   checkpoint_interval=1000000000, timeout_s=420, actor_port_base=53100)
    assert 'cold_init_checkpoint' not in cfg
    output=cluster_control.execute(cfg,REG['smoke_s'] if phase=='smoke' else None)
    d=F/'logs'/cfg['run_id']
    samples=[]
    with (F/'logs'/(cfg['run_id']+'-cluster-samples.jsonl')).open() as stream:
        for line in stream:
            event=json.loads(line);local=event['learner']; latest=local['entries'][0]['training']
            if not latest:
                continue
            s=latest[-1]
            samples.append(dict(epoch=event['epoch'],inserted=int(s['inserted']),
                                critic=int(s['critic_iterations']),fifo=int(s['fifo_backlog']),
                                debt=event['debt'],learner_load=local['load'],learner_network=local['network'],
                                saturated_actors=sum(a.get('recent',{}).get('min_saturation',0)>=.99
                                                     for h in event['actors'] for a in h['entries']),
                                sustained_warning_actors=sum(a.get('persistent_low_window_warning',False)
                                                             for h in event['actors'] for a in h['entries'])))
    output['observations']=samples
    output['final_checkpoint']=str(F/'checkpoints'/cfg['model_id']/('step-%d'%output['final']['inserted']))
    output['initial_policy_sha256']=(F/'checkpoints'/cfg['model_id']/'step-0.policy.sha256').read_text()
    return output


def evaluate_job(job, cutoff):
    import psutil

    run=job['run']
    cfg=dict(job['condition'])
    cfg.update(method='canopy',run_id=run,group=job['group'],seconds=REG['main_test_seconds'],port=job['port'],
               seed=3,checkpoint=job['checkpoint'],backend='moving_stats',code_commit='source-manifest.json',
               control_period_ms=20,history_frames=10,use_normalizer=False,no_training=True,
               absolute_deadline_epoch=cutoff-15,timeout_s=110,
               stop_file=str(F/'logs'/(run+'-stop.json')),server_binary=REG['training_config']['server_binary'])
    path=F/'configs'/(run+'.json')
    with path.open('x') as out:json.dump(cfg,out)
    work=F/'logs'/(run+'-work');work.mkdir()
    prefix=cfg['checkpoint']
    def hashes():
        return {s:hashlib.sha256(pathlib.Path(prefix+s).read_bytes()).hexdigest()
                for s in ('.index','.data-00000-of-00001')}
    before=hashes()
    start=time.monotonic()
    with (F/'logs'/(run+'.stdout')).open('wb') as out:
        p=subprocess.Popen([sys.executable,'-B',str(F/'repair_evaluate.py'),'--config',str(path)],
                           cwd=str(work),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        node.dump(node.own_path(run),[dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),role='evaluation',index=0)])
        try:
            code=p.wait(timeout=min(110,cutoff-time.time()-5))
            d=F/'logs'/run
            report=node.load(d/'result.json')
            assert code==0 and report and report['status']=='COMPLETED', str(report and report.get('reason'))
            assert report['policy_sha256_before']==report['policy_sha256_after'] and before==hashes()
            assert all(report['counters'][k]==0 for k in ('critic_iterations','critic_optimizer_ops','actor_updates',
                                                          'target_updates','exploration_calls'))
            windows,bins=network(d/'down.log',cfg['seconds'],with_seconds=True)
            assert all(w['complete_window'] for w in windows)
            return dict(job=job,status='COMPLETED',network=windows,second_bins=bins,
                        model_sha256=before,wall_s=time.monotonic()-start)
        finally:
            node.cleanup(run)


def evaluate(phase):
    jobs=node.load(R/'plans'/(phase+'.json'));rows=[]
    cutoff=REG['train_cutoff']-2600 if phase=='positive' else REG['eval_cutoff']
    for job in jobs:
        if (R/'STOP_ALL').exists() or time.time()+105>=cutoff:
            rows.append(dict(job=job,status='NOT_RUN_BUDGET'))
        else:
            try:
                rows.append(evaluate_job(job,cutoff))
            except Exception as error:
                rows.append(dict(job=job,status='EXECUTION_ERROR',reason=repr(error)))
        atomic(R/'results'/(phase+'-status.json'),dict(stage='RUNNING',done=len(rows),total=len(jobs)))
    return dict(status='COMPLETED' if all(r['status']=='COMPLETED' for r in rows) else 'INCOMPLETE',rows=rows)


def main(phase):
    try:
        result=train(phase) if phase in ('smoke','formal') else evaluate(phase)
    except Exception as error:
        result=dict(status='FAILED',reason=repr(error),traceback=traceback.format_exc())
    atomic(R/'results'/(phase+'-result.json'),result)
    return result['status']!='COMPLETED'


if __name__=='__main__':
    sys.exit(main(sys.argv[1]))
