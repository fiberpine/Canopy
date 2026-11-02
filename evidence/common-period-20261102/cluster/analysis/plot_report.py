import json,os,pathlib,sys
E=pathlib.Path(__file__).parent;sys.path.insert(0,str(E/'plot-deps'));os.environ['MPLCONFIGDIR']=str(E/'mpl-config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=E/'report/figures';P.mkdir(parents=True,exist_ok=True)
data=json.loads(json.loads((E/'A-audit-learner.json').read_text())['output'])['learner'];history=data['gradient_history'];x=[h['critic_iterations'] for h in history]
fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for t in ['20','40','80','120','160']:
 for axis,key,scale in [(ax[0],'rl_gradient_norm',1),(ax[1],'regularizer_gradient_norm',1),(ax[2],'saturation',100)]:axis.plot(x,[h['periods'][t][key]*scale for h in history],lw=.8,label=t+' ms')
for axis,title in zip(ax,['RL gradient norm','Preactivation regularizer gradient norm','Noiseless |window action| > .99 (%)']):axis.set_title(title);axis.set_xlabel('Complete critic iterations');axis.grid(alpha=.2)
ax[0].set_yscale('log');ax[1].set_yscale('log');ax[2].axhline(5,color='black',ls=':',lw=1);ax[2].legend(fontsize=8)
fig.suptitle('A training diagnostics: changing balanced minibatches, not frozen network acceptance')
for ext in ['png','svg']:fig.savefig(P/('training-gradients.'+ext),dpi=180)
plt.close(fig)
gates=E/'A-gates.json'
if gates.exists():
 g=json.loads(gates.read_text());a={(r['window'],r['group']):r for r in g['aggregates']};groups=[n for n in ['A20','A40','A80','A120','A160','fixed80','fixed120','fixed160','H0','official-orca','CUBIC'] if ('trim10',n) in a]
 fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained');spec=[('utilization',100,'Utilization (%)'),('queue_mean_ms',1,'Mean queue (ms)'),('queue_p95_ms',1,'Mean of run P95 queue (ms)'),('all_owned_cpu_percent_one_core',1,'Run-owned CPU (one core = 100%)')]
 for axis,(key,scale,title) in zip(axes.flat,spec):
  vals=[a['trim10',n][key]*scale if a['trim10',n][key] is not None else np.nan for n in groups]
  axis.bar(groups,vals,color=['#27828e' if n.startswith('A') else '#b2bdc9' for n in groups]);axis.set_title(title);axis.tick_params(axis='x',labelrotation=40);axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
 fig.suptitle('Frozen A / external references: condition-equal trim10 means\n'+('Complete 12 conditions x 2 independent runs' if g['status']!='INCOMPLETE' else 'INCOMPLETE: differing condition subsets, no group comparison yet'))
 for ext in ['png','svg']:fig.savefig(P/('frozen-summary.'+ext),dpi=180)
 plt.close(fig)
print('Figures generated in '+str(P))
