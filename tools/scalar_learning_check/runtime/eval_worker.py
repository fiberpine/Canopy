"""One finite sequential frozen stream per host; no retry, no optimization."""
import pathlib,os,sys,json,time,csv,traceback
R=pathlib.Path(os.environ['CANOPY_L1_ROOT']);F=R/'framework'
sys.path[:0]=[str(F),str(F/'code/runtime/_deps'),str(R/'tools/scalar_ipc_diagnostic')]
import node_control as node
import window_binding
def ipc_proof(folder):
    streams=[list(csv.DictReader((folder/n).open())) for n in ['ipc-python.csv','ipc-native.csv']];matched=0
    for source,dest in [streams,streams[::-1]]:
        sent={(r['kind'],r['sequence']):r for r in source if r['direction']=='sent'}
        for row in dest:
            if row['direction']!='received':continue
            other=sent[(row['kind'],row['sequence'])];assert row['payload']==other['payload'] and row['crc']==other['crc'];matched+=1
    stats=json.loads((folder/'ipc-python-stats.json').read_text());assert stats['incoming']['rejected']==0
    return dict(status='PASS',matched=matched)
def main():
    registration=json.loads((R/'eval-registration.json').read_text());master=json.loads((R/'registration.json').read_text());tasks=registration['tasks'];completed=[]
    status=R/'audit/evaluation-status.json';assert not status.exists()
    try:
        for task in tasks:
            assert not (R/'audit/STOP_EVALUATION').exists()
            assert time.time()+120<master['deadline_epoch'],'Evaluation time reserve exhausted'
            cfg=task['config'];rid=cfg['run_id'];(R/'audit/lease-heartbeat').touch()
            node.dump(status,dict(stage='RUNNING',current=rid,completed=len(completed),total=len(tasks),epoch=time.time()))
            import subprocess
            subprocess.run(['sudo','-n','/sbin/ip','tcp_metrics','flush','all'],check=True,timeout=10)
            node.start_evaluation(cfg)
            while True:
                s=node.status(rid)['entries'][0]
                if s.get('result'):break
                assert s['alive'] and time.time()<master['deadline_epoch'],s
                time.sleep(1)
            assert s['result']['status']=='COMPLETED',s['result']
            result=node.evaluation_summary(rid)
            if cfg['method']!='cubic':
                result['ipc']=ipc_proof(F/'logs'/rid)
                result['window_diagnostics']=window_binding.analyze(F/'logs'/rid)
            result['stage_specification']=task['stage_specification'];result['actual_checkpoint_step']=task.get('actual_checkpoint_step')
            node.cleanup(rid);completed.append(result);node.dump(R/'audit/evaluation-results.json',completed)
        node.dump(status,dict(stage='COMPLETE',completed=len(completed),total=len(tasks),epoch=time.time()))
    except Exception:
        node.dump(status,dict(stage='FAILED',completed=len(completed),total=len(tasks),traceback=traceback.format_exc(),epoch=time.time()))
        if 'rid' in locals():node.cleanup(rid)
        raise
if __name__=='__main__':main()
