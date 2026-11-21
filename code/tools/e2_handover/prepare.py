"""Prepare an isolated E2 workspace from an explicitly reviewed finite plan; never run TCP."""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework',type=pathlib.Path,required=True)
    parser.add_argument('--workspace',type=pathlib.Path,required=True)
    parser.add_argument('--plan',type=pathlib.Path,required=True)
    args=parser.parse_args()
    framework=args.framework.resolve();workspace=args.workspace.resolve()
    if os.name!='posix':parser.error('Live experiment preparation requires the documented Linux runtime')
    if workspace.exists():parser.error('Use a new workspace; existing runs are never overwritten')
    if framework==workspace or framework in workspace.parents:parser.error('Workspace must be outside source framework')
    required=['evaluate.py','code/orca/rl-module/agent.py','code/orca/params_base_eval.json',
              'code/runtime/_deps','configs/traces','resource_cost.py','mahimahi_metrics.py']
    for name in required:
        if not (framework/name).exists():parser.error('Missing framework input: '+name)
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    now=time.time();deadline=plan.get('deadline_epoch',0)
    if not now<deadline<=now+9000:parser.error('Plan requires a future absolute deadline within150 minutes')
    configs=plan.get('pilot',[])+plan.get('formal',[])
    if not configs:parser.error('Plan contains no explicitly selected runs')
    if len({c['run_id'] for c in configs})!=len(configs):parser.error('Duplicate run IDs')
    for c in configs:
        if c.get('method')!='orca' or c.get('no_training') is not True:parser.error('Frozen Orca only')
        if c.get('e2_arm') not in ('A0','A1','A2','A3','A4'):parser.error('Unknown intervention')
        if not 10<c.get('seconds',0)<=60:parser.error('Run duration must be11..60 seconds')
        if not c.get('code_commit'):parser.error('Explicit model/framework source identity required')
        if '/' in c['run_id'] or '\\' in c['run_id']:parser.error('Run ID must be a single path component')
        if not pathlib.Path(c['server_binary']).is_file():parser.error('Missing explicit normal-start native')
        if not pathlib.Path(c['checkpoint']+'.index').is_file():parser.error('Missing source checkpoint index')
        if not pathlib.Path(c['checkpoint']+'.data-00000-of-00001').is_file():parser.error('Missing source checkpoint tensor')
        trace=framework/'configs/traces'/c['trace']
        if hashlib.sha256(trace.read_bytes()).hexdigest()!=c['trace_sha256']:parser.error('Trace SHA mismatch')
    source=(framework/'evaluate.py').read_text()
    for anchor in ["self.last_id=rid;self.last_raw=vals;return value+b'\\0'",
                   's=state(env.reset());hist=np.zeros(70,np.float32);hist[-7:]=s',
                   'h1=np.concatenate((hist[7:],s1)).astype(np.float32)']:
        if source.count(anchor)!=1:parser.error('Unsupported evaluator revision; audit source hooks first')
    workspace.mkdir(parents=True)
    target=workspace/'framework';target.mkdir();(target/'logs').mkdir();(target/'configs').mkdir()
    (target/'code').symlink_to(framework/'code')
    (target/'configs/traces').symlink_to(framework/'configs/traces')
    shutil.copyfile(framework/'evaluate.py',target/'evaluate.py')
    for name in ['handover_runner.py','e2_support.py','node_driver.py']:
        shutil.copyfile(pathlib.Path(__file__).parent/name,workspace/name)
    (workspace/'audit').mkdir()
    (workspace/'plan.json').write_text(json.dumps(plan,indent=2))
    (workspace/'prepared.json').write_text(json.dumps(dict(
        status='PREPARED_NO_TCP',framework=str(framework),workspace=str(workspace),
        runs=len(configs),deadline_epoch=deadline,
        source_evaluator_sha256=hashlib.sha256((framework/'evaluate.py').read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(status='PREPARED_NO_TCP',runs=len(configs),workspace=str(workspace))))


if __name__=='__main__':
    main()
