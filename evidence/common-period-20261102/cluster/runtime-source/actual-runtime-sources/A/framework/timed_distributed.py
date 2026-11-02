"""Observe read waits and existing inference calls; no additional policy or RNG calls."""
import pathlib,sys,argparse,json,time,csv
R=pathlib.Path(__file__).resolve().parent
import distributed as d
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
ap.add_argument('--role',choices=['learner','actor'],required=True);ap.add_argument('--index',type=int,default=0)
args=ap.parse_args();cfg=json.loads(pathlib.Path(args.config).read_text())
if cfg.get('backend')=='moving_stats':
 import moving_stats_backend,agent_v2
 moving_stats_backend.install(agent_v2)
if args.role=='learner':d.learner(cfg)
else:
    import evaluate
    if cfg.get("connection_reset_seconds"):
        import connection_reset
        connection_reset.install(evaluate,cfg,args.index)
    f=(R/'logs'/(cfg['run_id']+'-actor%d-timing.csv'%args.index)).open('w',buffering=262144)
    writer=csv.writer(f);writer.writerow(['time_epoch','kind','duration_ms','state_id'])
    original_read=evaluate.StateReader.read
    def read(self):
        started=time.monotonic()
        # Initial TCP handshake/slow-start can exceed the short steady-state watchdog.
        # No transitions or actions are fabricated while waiting for the first real state.
        initial_deadline=started+60
        while True:
            try:
                value=original_read(self);break
            except RuntimeError as error:
                retry=(self.last_id is None and str(error)=='no valid state within15s'
                    and not self.stop.is_set() and time.monotonic()+15<=initial_deadline+.1)
                if not retry:raise
        writer.writerow([time.time(),'read_wait',(time.monotonic()-started)*1000,self.last_id])
        return value
    evaluate.StateReader.read=read
    original_action=d.Agent.get_concrete_action
    def action(self,*a,**kw):
        started=time.monotonic();value=original_action(self,*a,**kw)
        writer.writerow([time.time(),'inference_rpc',(time.monotonic()-started)*1000,''])
        return value
    d.Agent.get_concrete_action=action
    original_main=evaluate.main
    def main():
        try:return original_main()
        finally:f.flush();f.close()
    evaluate.main=main
    d.actor(cfg,args.index)
