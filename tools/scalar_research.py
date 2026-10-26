#!/usr/bin/env python3
"""Explicit isolated scalar research entrypoint. No work starts in plan/prepare.

These are source-adapted, performance-failing research protocols, not replacements
for the frozen baseline. Original models and frameworks are never overwritten.
"""
import argparse,copy,json,os,pathlib,shutil,subprocess,sys,time,uuid
P=pathlib.Path(__file__).resolve().parents[1]
VARIANTS=['orca-control','orca-moving','canopy-artifact']

def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2))

def configuration(variant,framework):
 x=json.loads((P/'templates'/('scalar-'+variant+'-train.json')).read_text())
 x['server_binary']=str(framework/'native/orca-server-normal-start')
 return x

def prepare(directory,variant):
 f=directory.resolve()
 if f.exists():raise ValueError('Refuse to overwrite an existing research directory: '+str(f))
 if ' ' in str(f):raise ValueError('Existing SSH harness requires paths without spaces')
 assets=P/'frameworks/baseline'
 for n in ['code/orca/rl-module/agent.py','code/canopy/rl-module/agent_v2.py','configs/traces']:
  if not (assets/n).exists():raise ValueError('Missing preserved runtime asset: '+n)
 f.mkdir(parents=True)
 for n in ['logs','configs','audit','checkpoints','metrics','plots','report']:(f/n).mkdir()
 for p in (P/'frameworks/canopy_single').glob('*.py'):shutil.copy2(str(p),str(f/p.name))
 shutil.copytree(str(P/'frameworks/canopy_single/native'),str(f/'native'))
 (f/'code').symlink_to((assets/'code').resolve(),target_is_directory=True)
 (f/'configs/traces').symlink_to((assets/'configs/traces').resolve(),target_is_directory=True)
 write(f/'configs/train.example.json',configuration(variant,f))
 write(f/'scalar-research.json',dict(variant=variant,scope='Research, not a performance-passing baseline',prepared_from=str(P)))
 print(json.dumps(dict(status='PREPARED_NO_TRAINING',framework=str(f),variant=variant)))

def build(f):
 s=f/'native/src';target=f/'native/orca-server-normal-start'
 cmd=['g++','-std=c++11','-O2','-pthread','-I',str(s),str(s/'orca-server-normal-start.cc'),str(s/'flow.cc'),'-o',str(target)]
 subprocess.check_call(cmd);print(json.dumps(dict(status='BUILT_NOT_TCP_TESTED',binary=str(target))))

def train(args,f):
 c=json.loads((f/'configs/train.example.json').read_text())
 if not 0<args.seconds<=3600 or not 1<=args.actors<=24:raise ValueError('Finite research budget: seconds<=3600,1..24actors')
 if not pathlib.Path(c['server_binary']).is_file():raise ValueError('Build the explicit normal-start native first')
 cluster=json.loads(args.cluster.read_text())
 if len(cluster['nodes'])<2:raise ValueError('Require learner and at least one actor location')
 if any('PRIVATE_IP' in n['ip'] or n['ip']=='PRIVATE_HOST' for n in cluster['nodes']):raise ValueError('Fill your own cluster endpoints')
 now=time.time();run='scalar-'+json.loads((f/'scalar-research.json').read_text())['variant']+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
 cap=int(args.seconds)+1200
 cluster.update(start_epoch=now,deadline_epoch=now+cap,authorization='Explicit finite scalar_research train command; no automatic continuation')
 authority=f/'configs'/(run+'-authorization.json');write(authority,cluster)
 os.environ['ORCA_AUTHORIZATION_FILE']=str(authority)
 for name,key in [('ORCA_SSH_KEY','ssh_key_path'),('ORCA_KNOWN_HOSTS','known_hosts_path')]:
  if cluster.get(key):os.environ[name]=cluster[key]
 hosts=[x['ip'] for x in cluster['nodes'][1:]]
 c['conditions']=[copy.deepcopy(c['conditions'][i%len(c['conditions'])]) for i in range(args.actors)]
 c.update(run_id=run,model_id=run,num_actors=args.actors,actor_hosts=[i%len(hosts) for i in range(args.actors)],
  learner_endpoint=cluster['nodes'][0]['ip']+':'+str(args.learner_port),actor_port_base=args.actor_port,
  stop_file=str(f/'logs'/(run+'-stop.json')),absolute_deadline_epoch=now+cap-60,timeout_s=cap-120,
  evaluation_reserve_s=90,capacity_only=False,sample_budget=None)
 if args.smoke:c.update(portable_smoke_training=True,warmup=512,minimum_enqueued_per_actor=64,checkpoint_interval=100000000)
 write(f/'configs'/(run+'-input.json'),c)
 sys.path.insert(0,str(f));import cluster_control
 result=cluster_control.execute(c,duration=args.seconds)
 print(json.dumps(dict(status=result['status'],run_id=run,result=str(f/'metrics'/(run+'-cluster-result.json')),meaning='Run completion is not performance acceptance; smoke is not convergence')))

def evaluate(args,f):
 c=json.loads((f/'configs/train.example.json').read_text());condition=c['conditions'][args.scenario]
 if not 0<args.seconds<=120:raise ValueError('Frozen research evaluation is limited to120seconds')
 if not args.checkpoint.with_suffix('.index').is_file():raise ValueError('Checkpoint prefix/index missing')
 run='scalar-eval-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
 cfg=dict(condition,method=c['policy_family'],run_id=run,group='scalar-research',checkpoint=str(args.checkpoint.resolve()),
  backend='moving_stats',seconds=args.seconds,port=args.port,seed=3,no_training=True,
  code_commit='scalar-research-publication-20261026',absolute_deadline_epoch=time.time()+args.seconds+300,
  timeout_s=args.seconds+180,control_period_ms=20,server_binary=c['server_binary'],require_normal_cubic_start=True)
 path=f/'configs'/(run+'.json');write(path,cfg)
 subprocess.check_call([sys.executable,'-B',str(f/'repair_evaluate.py'),'--config',str(path)],cwd=str(f))
 sys.path.insert(0,str(f));import node_control
 summary=node_control.evaluation_summary(run)
 print(json.dumps(dict(status=summary['result']['status'],metrics=str(f/'metrics'/(run+'-summary.json')))))

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command')
 for name in ['plan','prepare']:
  q=sub.add_parser(name);q.add_argument('--directory',type=pathlib.Path,required=True);q.add_argument('--variant',choices=VARIANTS,required=True)
 for name in ['build','train','evaluate']:
  q=sub.add_parser(name);q.add_argument('--framework',type=pathlib.Path,required=True)
  if name=='train':
   q.add_argument('--cluster',type=pathlib.Path,required=True);q.add_argument('--seconds',type=float,required=True);q.add_argument('--actors',type=int,default=24)
   q.add_argument('--learner-port',type=int,default=31920);q.add_argument('--actor-port',type=int,default=35000);q.add_argument('--smoke',action='store_true')
  if name=='evaluate':
   q.add_argument('--checkpoint',type=pathlib.Path,required=True);q.add_argument('--scenario',type=int,default=0);q.add_argument('--seconds',type=float,default=60);q.add_argument('--port',type=int,default=34000)
 args=p.parse_args()
 if args.command=='plan':print(json.dumps(configuration(args.variant,args.directory.resolve()),indent=2));return
 if args.command=='prepare':prepare(args.directory,args.variant);return
 if args.command not in ['build','train','evaluate']:p.print_help();return
 f=args.framework.resolve()
 if not (f/'scalar-research.json').is_file():raise ValueError('Use an isolated directory prepared by this tool')
 if args.command=='build':build(f)
 elif args.command=='train':train(args,f)
 else:evaluate(args,f)

if __name__=='__main__':main()
