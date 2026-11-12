"""Plot observed repetitions only; label the interrupted condition explicitly."""
import argparse
import json
from pathlib import Path


def main(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties,fontManager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'svg.fonttype':'path','axes.unicode_minus':False})
    data=json.loads((folder/'observed-performance-assessment.json').read_bytes())
    assert data['complete_runs']==95 and data['candidate_valid_runs']==71 and data['H0_valid_runs']==24
    cells=data['cells'];hosts=json.loads((folder/'frozen-final-latest.json').read_bytes())
    reference={r['job']['condition']['condition']:next(w for w in r['windows'] if w['window']=='trim10')
               for h in hosts for r in h['rows'] if r['job']['group']=='H0-reference'}
    bws=(12,24,36,42,48,72);rtts=(20,50,100,200)
    maxqueue=max([c['available_repeat_means']['queue_mean_ms'] for c in cells.values()]+[c['queue_mean_ms'] for c in reference.values()])
    fig,axes=plt.subplots(2,2,figsize=(12,9.5),layout='constrained')
    for col in range(2):
        for row,key in enumerate(('utilization','queue_mean_ms')):
            grid=np.array([[(cells['G%d-R%d'%(bw,rtt)]['available_repeat_means'] if col==0 else reference['G%d-R%d'%(bw,rtt)])[key]
                            for rtt in rtts] for bw in bws])
            if row==0:grid=grid*100
            vmax=100 if row==0 else maxqueue
            ax=axes[row,col];im=ax.imshow(grid,origin='lower',vmin=0,vmax=vmax,aspect='auto',cmap='YlGnBu' if row==0 else 'YlOrRd')
            ax.set(xticks=range(4),xticklabels=rtts,yticks=range(6),yticklabels=bws,xlabel='基础RTT（ms）',ylabel='带宽（Mbps）',
                   title=('新端点：已完成重复均值' if col==0 else 'H0：每格1次')+('，利用率%' if row==0 else '，平均排队ms'))
            for i,bw in enumerate(bws):
                for j,rtt in enumerate(rtts):
                    label='%.1f'%grid[i,j]
                    if col==0 and cells['G%d-R%d'%(bw,rtt)]['valid_repeats']==2:label+='†'
                    ax.text(j,i,label,ha='center',va='center',color='white' if grid[i,j]>.55*vmax else '#17202a',fontsize=11)
            fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('保留的完整冻结结果：71次新模型＋24次H0参照',fontsize=15)
    fig.text(.01,-.025,'† G12-R100仅2次，其他新模型格各3次；中断run不填0、不外推。每机一流、同native、trim10；H0单重复不能说明稳定性。',fontsize=10)
    out=folder/'plots';out.mkdir(exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(out/('frozen-available-grid24.'+suffix),dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
