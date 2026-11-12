"""Local report plots. Never imports or changes the TF1 training environment."""
from pathlib import Path
import json
import sys

import argparse
p=argparse.ArgumentParser()
p.add_argument('folder',type=Path)
H=p.parse_args().folder
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager

OUT=H/'plots';OUT.mkdir(exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    fontManager.addfont(str(font))
    plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
def load(name):
    if name=='training-analysis-input.json' and not (H/name).exists():
        from zipfile import ZipFile
        with ZipFile(H/'training-small-analysis.zip') as z:return json.loads(z.read('analysis-input.json'))
    return json.loads((H/name).read_bytes())


def save(fig,name):
    for ext in ('png','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)


def training():
    data=load('training-analysis-input.json')
    rows=data['training'];x=np.array([float(r['inserted'])/1e6 for r in rows]);grad=np.array([float(r['actor_grad']) for r in rows])
    finite=np.isfinite(grad)
    positive=grad[finite&(grad>0)]
    zero_floor=min(1e-14,float(positive.min())*.1) if len(positive) else 1e-14
    hist=data['history'];hx=np.array([r['inserted']/1e6 for r in hist])
    actions=np.full((24,len(hist)),np.nan)
    for j,r in enumerate(hist):
        for a in r['actors']:actions[a['index'],j]=a['recent'].get('action_mean',np.nan)
    fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained',sharex=True,gridspec_kw={'height_ratios':[1,1,2]})
    axes[0].plot(x[finite],np.where(grad[finite]==0,zero_floor,grad[finite]),color='#2563a5',lw=1)
    zero=finite&(grad==0);axes[0].scatter(x[zero],np.full(zero.sum(),zero_floor),s=5,color='#ce5a40',label='原记录为0（显示于%.1e）'%zero_floor)
    axes[0].set(yscale='log',ylabel='Actor梯度范数',title='完整训练轨迹：学习梯度、更新欠额与部署动作')
    axes[0].legend(fontsize=9)
    debt=np.array([np.nan if r['debt'] is None else r['debt'] for r in hist]);axes[1].plot(hx,debt,color='#249c81',lw=1)
    axes[1].set(ylabel='已记录更新欠额',ylim=(0,None))
    # Preserve actual sample coordinates; coordinator spacing is not uniform.
    keep=np.r_[np.diff(hx)!=0,True];cx=hx[keep];values=actions[:,keep]
    edges=np.r_[cx[0],(cx[:-1]+cx[1:])/2,cx[-1]]
    image=axes[2].pcolormesh(edges,np.arange(25)-.5,values,cmap='RdBu_r',vmin=-1,vmax=1,shading='flat',rasterized=True)
    axes[2].set(xlabel='已入库真实经验（百万条）',ylabel='Actor编号',yticks=np.arange(0,24,4))
    fig.colorbar(image,ax=axes[2],label='近200条经验的无噪动作均值')
    fig.text(.01,-.025,'梯度为既有诊断批记录，不代表每步；动作图为训练中近200条经验，不等于冻结性能。保持原数据，不据此挑选中间模型。',fontsize=10)
    save(fig,'training-trajectory')


def fixed_probe():
    data=load('fixed-probe-analysis.json')
    assert data['status']=='COMPLETED'
    rows=data['rows'];x=np.array([r['step']/1e6 for r in rows])
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained',sharex=True)
    ax=axes[0]
    for key,label,color in [('minimum_boundary','最小动作比例','#2563a5'),('maximum_boundary','最大动作比例','#ce5a40'),('exact','精确±1比例','#555555')]:
        ax.plot(x,[100*r['policy'][key] for r in rows],marker='o',ms=3,label=label,color=color)
    ax.set(ylabel='固定512状态占比（%）',ylim=(-3,103),title='同一历史真实探针：训练批恢复，不等于旧状态策略已健康')
    ax.legend(fontsize=10)
    ax=axes[1]
    ax.plot(x,[r['policy']['mean'] for r in rows],label='当前权重＋当前BN',color='#17202a',marker='o',ms=3)
    rr=[r for r in rows if 'current_weights_previous_BN_policy' in r]
    ax.plot([r['step']/1e6 for r in rr],[r['current_weights_previous_BN_policy']['mean'] for r in rr],label='当前权重＋上个保存点BN',color='#2563a5',ls='--',marker='s',ms=3)
    ax.plot([r['step']/1e6 for r in rr],[r['previous_weights_current_BN_policy']['mean'] for r in rr],label='上个保存点权重＋当前BN',color='#249c81',ls=':',marker='^',ms=3)
    ax.set(xlabel='保存点入库经验（百万条）',ylabel='固定探针动作均值',ylim=(-1.08,1.08))
    ax.legend(fontsize=9)
    fig.text(.01,-.035,'旧失败run保留的真实状态，按actor固定配额；不是新连接性能样本。函数交换破坏协变，只用于诊断，不能把效应归为在线唯一因果。',fontsize=10)
    save(fig,'fixed-probe-and-BN')


def resources():
    data=load('training-analysis-input.json');series=data['resources']
    start=load('training-complete.json')['result']['started_epoch']
    def busy(before,after):
        delta={k:after[k]-before[k] for k in before}
        total=sum(v for k,v in delta.items() if k not in ('guest','guest_nice'))
        return 100*sum(delta.get(k,0) for k in ('user','nice','system','irq','softirq'))/total if total>0 else np.nan
    times=[];host=[];peak=[];process=[]
    for a,b in zip(series[:-1],series[1:]):
        aa=[a['learner']]+a['actor_hosts'];bb=[b['learner']]+b['actor_hosts']
        times.append((b['epoch']-start)/60)
        host.append([busy(x['cpu_times'],y['cpu_times']) for x,y in zip(aa,bb)])
        peak.append(max(busy(x,y) for x,y in zip(aa[0]['cpu_times_percpu'],bb[0]['cpu_times_percpu'])))
        pa=next((p for p in aa[0]['processes'] if p['role']=='learner'),None);pb=next((p for p in bb[0]['processes'] if p['role']=='learner'),None)
        dt=bb[0]['epoch']-aa[0]['epoch'];cores=len(bb[0]['cpu_times_percpu'])
        valid=pa and pb and pa['pid']==pb['pid'] and dt>0 and all(k in p['cpu'] for p in (pa,pb) for k in ('user','system'))
        process.append(100*sum(pb['cpu'][k]-pa['cpu'][k] for k in ('user','system'))/dt/cores if valid else np.nan)
    host=np.asarray(host);fig,axes=plt.subplots(2,1,figsize=(12,7.5),layout='constrained',sharex=True)
    axes[0].plot(times,host[:,0],label='Learner整机忙碌',color='#2563a5',lw=1)
    axes[0].plot(times,process,label='Learner进程CPU折算整机',color='#249c81',lw=1)
    axes[0].plot(times,peak,label='最忙逻辑CPU区间值',color='#888888',lw=.6,alpha=.7)
    axes[0].set(ylabel='CPU（%）',ylim=(0,100),title='正式训练资源：实际区间CPU计数差，按8个逻辑CPU解释')
    axes[0].legend(fontsize=9)
    for i in range(4):axes[1].plot(times,host[:,i+1],label='Actor机%d'%(i+1),lw=1)
    axes[1].set(xlabel='训练墙钟（分钟）',ylabel='整机忙碌CPU（%）',ylim=(0,100));axes[1].legend(fontsize=9,ncol=4)
    for name in ('during-run-diagnostic-registration.json','during-run-diagnostic-r2-registration.json','during-run-diagnostic-r3-registration.json'):
        if (H/name).exists():
            r=load(name)
            for ax in axes:ax.axvspan((r['start_epoch']-start)/60,(r['end_epoch']-start)/60,color='#555555',alpha=.15)
    fig.text(.01,-.03,'CPU忙碌不含idle/iowait/steal。本轮正式训练期间没有额外TF诊断；这是训练资源曲线，不是部署CPU收益或扩容准入证据。',fontsize=10)
    save(fig,'training-CPU')


def frozen():
    hosts=load('frozen-final-latest.json');rows=[r for h in hosts for r in h['rows']]
    assert len(rows)==96 and all(r['status']=='COMPLETED' for r in rows if r['job']['group']=='onlinebn-final6m')
    conditions=load('health-registration.json')['conditions'];bws=sorted({c['nominal_mbps'] for c in conditions});rtts=sorted({c['base_rtt_ms'] for c in conditions})
    groups=('onlinebn-final6m','H0-reference');metrics=('utilization','queue_mean_ms')
    fig,axes=plt.subplots(2,2,figsize=(12,9.5),layout='constrained')
    for col,group in enumerate(groups):
        for row,key in enumerate(metrics):
            grid=np.full((len(bws),len(rtts)),np.nan)
            for i,bw in enumerate(bws):
                for j,rtt in enumerate(rtts):
                    matches=[r for r in rows if r['status']=='COMPLETED' and r['job']['group']==group and r['job']['condition']['nominal_mbps']==bw and r['job']['condition']['base_rtt_ms']==rtt]
                    values=[next(w for w in r['windows'] if w['window']=='trim10')[key] for r in matches]
                    assert len(values)==3 if group==groups[0] else len(values) in (0,1)
                    if values:grid[i,j]=np.mean(values)*(100 if key=='utilization' else 1)
            ax=axes[row,col];vmax=100 if row==0 else max(1,max(next(w for w in r['windows'] if w['window']=='trim10')[key] for r in rows if r['status']=='COMPLETED'))
            cmap=plt.get_cmap('YlGnBu' if row==0 else 'YlOrRd').copy();cmap.set_bad('#e1e6eb')
            im=ax.imshow(np.ma.masked_invalid(grid),origin='lower',cmap=cmap,vmin=0,vmax=vmax,aspect='auto')
            ax.set(xticks=np.arange(len(rtts)),xticklabels=rtts,yticks=np.arange(len(bws)),yticklabels=bws,xlabel='基础RTT（ms）',ylabel='带宽（Mbps）',title=('新终点：每格3次均值' if col==0 else 'H0参照：有效23格各1次')+('，利用率%' if row==0 else '，平均排队ms'))
            for i in range(len(bws)):
                for j in range(len(rtts)):ax.text(j,i,'%.1f'%grid[i,j] if np.isfinite(grid[i,j]) else 'N/A',ha='center',va='center',color='white' if grid[i,j]>.55*vmax else '#17202a',fontsize=11)
            fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('最终冻结性能：24条件、同native、每机单流，trim10',fontsize=15)
    fig.text(.01,-.03,'单个训练seed。H0的G12-R200中途无反馈失败，保留N/A、不补0或重测；参照不是全24总体。健康门按新模型3次中位数另判。',fontsize=10)
    save(fig,'frozen-grid24')


if __name__=='__main__':
    if (H/'training-analysis-input.json').exists() or (H/'training-small-analysis.zip').exists():training();resources()
    if (H/'fixed-probe-analysis.json').exists():fixed_probe()
    if (H/'frozen-final-latest.json').exists():
        hosts=load('frozen-final-latest.json')
        if all(h['finished'] for h in hosts) and all(r['status']=='COMPLETED' for h in hosts for r in h['rows'] if r['job']['group']=='onlinebn-final6m'):frozen()
    print(json.dumps(dict(plots=[p.name for p in OUT.glob('*.png')],matplotlib=matplotlib.__version__)))
