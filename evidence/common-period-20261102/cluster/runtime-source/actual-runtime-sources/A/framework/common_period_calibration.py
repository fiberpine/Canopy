"""Fixed-budget fit/check calibration from isolated same-connection blocks."""
import numpy as np
from common_period import calibration,bias_correction,center,encoded_period

def calibrate(model,rows):
    groups={}
    for row in rows:
        if row.get('block') is not None:groups.setdefault((row['actor'],row['block'],row['connection']),[]).append(row)
    if len(groups)!=120:raise ValueError('Require24 actors x5 distinct calibration blocks')
    blocks=[]
    for key,part in groups.items():
        periods={r['period_ms'] for r in part}
        if len(periods)!=1:raise ValueError('Mixed period in calibration block')
        blocks.append(dict(connection=key[2],end_connection=part[-1]['next_connection'],period_ms=part[0]['period_ms'],duration=[r['duration'] for r in part],reward=[r['reward'] for r in part]))
    params=calibration(blocks);model.initialize_centering(params['K'],params['n0'])
    records=[]
    for (actor,block,connection),part in groups.items():
        elapsed=np.r_[0.,np.cumsum([r['duration'] for r in part])]
        starts=np.flatnonzero(elapsed[:-1]<20.)
        ends=np.searchsorted(elapsed,elapsed[starts]+20.,side='left')-1
        valid=ends<len(part);starts,ends=starts[valid],ends[valid]
        if not len(starts):raise ValueError('No legal >=20s suffix')
        def value(states):
            s=np.asarray(states,np.float32);a=np.zeros((len(s),2),np.float32);a[:,1]=encoded_period(part[0]['period_ms'])
            a[:,0]=model.session.run(model.mu,{model.s:s,model.action:a})[:,0]
            q1,q2=model.session.run(model.q,{model.s:s,model.action:a});return ((q1+q2)*.5).ravel()
        predicted=value([part[i]['s0'] for i in starts]);tails=value([part[j]['s1'] for j in ends])
        rewards=np.asarray([center(r['reward'],r['discount'],r['kind'],params['K']) for r in part])
        for index,(i,j) in enumerate(zip(starts,ends)):
            dt=elapsed[j+1]-elapsed[i];suffix=.995**(dt/.02)
            weights=.995**((elapsed[i:j+1]-elapsed[i])/.02)
            label=float(np.dot(weights,rewards[i:j+1])+suffix*tails[index])
            records.append(dict(actor=actor,block=int(block),period_ms=part[0]['period_ms'],duration=float(dt),discount=float(suffix),label=label,predicted=float(predicted[index]),tail=float(suffix*tails[index])))
    fit=[r for r in records if r['actor']%2==0];check=[r for r in records if r['actor']%2]
    offset=bias_correction([r['label'] for r in fit],[r['predicted'] for r in fit],[r['discount'] for r in fit])
    model.calibrate_bias_once(np.float32(offset))
    def summarize(rs):
        errors=np.asarray([r['label']-r['predicted']-offset*(1-r['discount']) for r in rs])
        return dict(points=len(rs),residual_mean=float(errors.mean()),residual_std=float(errors.std()),tail_abs_mean=float(np.mean([abs(r['tail']) for r in rs])),tail_abs_max=float(max(abs(r['tail']) for r in rs)))
    return dict(params=params,global_bias_offset=offset,fit=summarize(fit),independent_check=summarize(check),records=records,stopping_gate=False,limitation='Finite same-policy suffix plus frozen tail; check not an A stopping gate; no universal critic-correctness claim')
