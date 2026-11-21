"""Plot saved fixed-budget TF capability results; performs no optimization."""
import argparse
import csv
import json
from pathlib import Path


def plot(evidence,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(evidence);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    series=[('matched-high-period-lr-capability-tf.json','Concurrent, residual lr=1e-3','#c9822e'),
            ('claude-capability-tf.json','Concurrent, residual lr=1e-4','#3b70b5'),
            ('staged-production-capability.json','Common first 1000, residual lr=1e-4','#15876b')]
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),sharex=True,sharey=True)
    rows=[];threshold=None
    for filename,label,color in series:
        record=json.loads((root/filename).read_text());threshold=record['maximum_advantage_mae']
        by_sign={r['sign']:r for r in record['critic_tasks']}
        for r,sign in enumerate((1.,-1.)):
            trace=by_sign[sign]['trajectory'];x=[v['updates'] for v in trace]
            for c in range(2):
                y=[v['advantage_mae'][c] for v in trace]
                if filename=='staged-production-capability.json':
                    # Do not interpolate across the unobserved 1000-update
                    # release: it would visually imply learning before release.
                    early=[i for i,a in enumerate(x) if a<1000]
                    late=[i for i,a in enumerate(x) if a>1000]
                    axes[r,c].plot([x[i] for i in early],[y[i] for i in early],color=color,lw=1.6,marker='.',ms=3)
                    axes[r,c].plot([x[i] for i in late],[y[i] for i in late],color=color,lw=1.6,marker='.',ms=3,label=label)
                else:axes[r,c].plot(x,y,color=color,lw=1.6,marker='.',ms=3,label=label)
                rows.extend([dict(series=label,sign=sign,critic=c+1,updates=a,advantage_mae=b) for a,b in zip(x,y)])
    for r in range(2):
        for c in range(2):
            ax=axes[r,c];ax.axhline(threshold,color='#344054',ls='--',lw=1.2,label='Registered MAE gate')
            ax.axvspan(0,1000,color='#e7ecf2',alpha=.5,zorder=0)
            ax.set_yscale('log');ax.set_ylim(2e-4,.06);ax.set_xlim(0,10000)
            ax.grid(axis='y',which='major',color='#d6dce4',lw=.6)
            ax.set_title(('Positive' if r==0 else 'Negative')+' known advantage | critic '+str(c+1),fontsize=10)
            ax.spines[['top','right']].set_visible(False)
            if c==0:ax.set_ylabel('Advantage MAE')
            if r==1:ax.set_xlabel('Complete critic iterations')
    fig.suptitle('Small period advantages: fixed 10,000-iteration budget',fontsize=13,y=.99)
    fig.text(.5,.945,'Known single-decision values with the same future policy; synthetic tasks, no TCP measurements',ha='center',fontsize=9,color='#475467')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,-.005))
    fig.tight_layout(rect=(0,.09,1,.92))
    for suffix in ('png','svg','pdf'):
        path=out/('value-resolution.'+suffix);fig.savefig(path,dpi=170,bbox_inches='tight')
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    with (out/'value-resolution.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plot(a.evidence,a.output)
