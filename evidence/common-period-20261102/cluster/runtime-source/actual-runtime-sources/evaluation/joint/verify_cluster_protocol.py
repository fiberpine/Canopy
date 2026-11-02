
import json,pathlib,time,numpy as np
from common_period import PERIODS,encoded_period,discount
from common_period_tf import Model
from common_period_transport import Policy,snapshot_bytes
from common_period_calibration import calibrate
R=pathlib.Path(__file__).resolve().parent.parent;cp='/home/experiment/orca-release-current/models/mtpjoint/model'
reward=json.loads((pathlib.Path('/home/experiment/orca-common-period-20261102/delivery-a0-v2')/'reward-v1.json').read_text())
states=np.load('/home/experiment/orca-two-track-run-20261025-2030/innovation-package/data/actor1/telemetry-s0-r0-m80-audit.npz')['s0'];model=Model(cp,0.,{str(t):1/(1-.995**(t/20.)) for t in PERIODS},reward_config=reward)
policy=Policy(cp,model.metadata['reward_checksum']);first=snapshot_bytes(model.snapshot(window_only=True));policy.load(first)
errors=[];initial=[]
for i,t in enumerate(PERIODS):
 s=states[i];a=np.array([[0,encoded_period(t)]],np.float32);expected=float(model.session.run(model.mu,{model.s:s[None,:],model.action:a})[0,0]);actual=policy.window(s,t);errors.append(abs(expected-actual));initial.append(expected)
rows=[];g=discount(1.);r=.3*(1-g)/.005
for actor in range(24):
 for block,t in enumerate(PERIODS):
  conn='%d-c%d'%(actor,int(block==4));s=states[actor]
  rows += [dict(actor=actor,block=block,period_ms=t,connection=conn,next_connection=conn,duration=1.,reward=r,discount=g,kind='continuing',s0=s,s1=s) for _ in range(40)]
audit=calibrate(model,rows);policy.load(snapshot_bytes(model.snapshot(window_only=True)))
unchanged=all(policy.window(states[i],t)==initial[i] for i,t in enumerate(PERIODS))
s=states[np.arange(512)%len(states)].astype(np.float32);a=np.column_stack([np.zeros(512),np.resize([encoded_period(t) for t in PERIODS],512)]).astype(np.float32)
b=dict(s0=s,s1=s[::-1].copy(),action=a,reward=np.zeros((512,1),np.float32),discount=np.full((512,1),.98,np.float32),bootstrap=np.ones((512,1),np.float32));noise=np.zeros((512,1),np.float32)
for i in range(20):model.train_batch(b,noise,i%2==1)
start=time.monotonic()
for i in range(200):model.train_batch(b,noise,i%2==1)
speed=200/(time.monotonic()-start);policy.load(snapshot_bytes(model.snapshot(window_only=True)))
newerrors=[]
for i,t in enumerate(PERIODS):
 p=np.array([[0,encoded_period(t)]],np.float32);expected=float(model.session.run(model.mu,{model.s:states[i:i+1],model.action:p})[0,0]);newerrors.append(abs(expected-policy.window(states[i],t)))
try:policy.load(first);reject=False
except ValueError:reject=True
out=dict(status='PASS' if max(errors+newerrors)==0 and unchanged and reject and speed>=96 else 'FAIL',initial_policy_error=max(errors),published_policy_error=max(newerrors),calibration_window_unchanged=unchanged,old_snapshot_rejected=reject,new_complete_iterations_per_s=speed,synthetic_calibration_blocks=120,synthetic_calibration_fit=audit['fit'],synthetic_calibration_check=audit['independent_check'],new_TCP=0,formal_credit=0,candidate_checkpoint_saved=False)
(R/'protocol-result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out));model.session.close();policy.session.close()
