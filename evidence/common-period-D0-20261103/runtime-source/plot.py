import json,pathlib,sys,os
E=pathlib.Path(__file__).parent;sys.path.insert(0,str(E.parent/'common-period-cluster-20261102/plot-deps'));os.environ['MPLCONFIGDIR']=str(E/'mpl-config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=json.loads((E/'results.json').read_text());raw=json.loads((E/'learner-results.json').read_text());P=E/'report/figures';P.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');t=[str(p['period_ms']) for p in r['S1']['period_ledger']]
for axis,key,label in [(axes[0],'inserted','Cumulative real inserted rows'),(axes[1],'physical_seconds','Cumulative real physical exposure (s)')]:axis.bar(t,[p[key] for p in r['S1']['period_ledger']],color='#287b8e');axis.set_title(label);axis.set_xlabel('Executed period (ms)');axis.grid(axis='y',alpha=.2)
names=['actor_only','selector_5_two_fetch','bridge_wire_only'];vals=[[x['process_cpu_ms_per_call'] for x in r['S5']['records'] if x['variant']==n] for n in names];axes[2].bar(['Actor','Selector 5','Wire'],[np.mean(v) for v in vals],color=['#287b8e','#e49f42','#b7bec8']);axes[2].errorbar(range(3),[np.mean(v) for v in vals],yerr=[np.std(v,ddof=1) for v in vals],fmt='none',color='black',capsize=4);axes[2].set_title('CPU ms/call: six within-process blocks');axes[2].grid(axis='y',alpha=.2);fig.suptitle('D0 frozen diagnosis: counts, physical exposure, and inference cost are distinct')
fig.savefig(P/'ledger-and-cpu.png',dpi=180);plt.close(fig)
pairs=[x for x in raw['S1']['actor_gradients'] if 'pair_ms' in x];periods=[20,40,80,120,160];fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for axis,key in zip(axes,['shared_fc1','shared_fc2','period_input']):
 a=np.eye(5)
 for i,t1 in enumerate(periods):
  for j,t2 in enumerate(periods):
   if i<j:a[i,j]=a[j,i]=np.mean([p['by_layer'][key] for p in pairs if p['pair_ms']==[t1,t2] and p['by_layer'][key] is not None])
 im=axis.imshow(a,vmin=-1,vmax=1,cmap='RdBu');axis.set_xticks(range(5),periods);axis.set_yticks(range(5),periods);axis.set_title(key)
 for i in range(5):
  for j in range(5):axis.text(j,i,'%.2f'%a[i,j],ha='center',va='center',color='white' if abs(a[i,j])>.5 else 'black',fontsize=8)
fig.colorbar(im,ax=axes,label='Mean gradient cosine');fig.suptitle('Same retained training states; not missing deployment failure states\nPeriod-input signs also reflect the signed t encoding')
fig.savefig(P/'gradient-directions.png',dpi=180);plt.close(fig)
print('Two standalone PNG figures generated')
