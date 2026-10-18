"""Reservoirs of real transitions per actor and executed-period neighborhood.

This changes replay retention/sampling only. No new environment credit, actor
schedule, reward, target or action change. Caller owns the replay RLock.
"""
import numpy as np

class PeriodBank:
 def __init__(self,cfg):
  self.actors=int(cfg['num_actors']);self.periods=np.asarray(cfg['timing_calibration_periods_ms'],np.float64)
  self.per_bin=int(cfg['period_bank_rows_per_bin']);self.capacity=self.actors*len(self.periods)*self.per_bin
  self.retention=cfg.get('period_bank_retention','reservoir')
  assert self.retention in ('reservoir','recent'), 'Explicit real-experience retention mode'
  assert cfg.get('joint_control') and cfg['sampler']=='balanced_actor'
  assert self.per_bin>=cfg['timing_min_retained_per_period'] and self.capacity<cfg['memsize']-cfg['warmup']
  self.centers=2*np.log(self.periods/cfg['mtp_min_ms'])/np.log(cfg['mtp_max_ms']/cfg['mtp_min_ms'])-1
  self.rng=np.random.RandomState(cfg['seed']+813007)
  self.draw_rng=np.random.RandomState(cfg['seed']+813008)
  self.seen=np.zeros((self.actors,len(self.periods)),np.int64);self.sampled=np.zeros_like(self.seen);self.cursor=np.zeros(self.actors,np.int64)
  self.arrays=[np.zeros((self.capacity,d),np.float32) for d in [72,2,1,72,1,1,1]]
  self.birth=np.full(self.capacity,-1,np.int64)
 def bucket(self,action):return int(np.argmin(np.abs(self.centers-float(action))))
 def base(self,actor,bucket):return (actor*len(self.periods)+bucket)*self.per_bin
 def store(self,actor,row,birth):
  assert 0<=actor<self.actors and len(row)==7 and birth>=0
  bucket=self.bucket(row[1][1]);n=int(self.seen[actor,bucket]);self.seen[actor,bucket]+=1
  j=n%self.per_bin if self.retention=='recent' else (n if n<self.per_bin else int(self.rng.randint(n+1)))
  if j>=self.per_bin:return
  ix=self.base(actor,bucket)+j
  for array,value in zip(self.arrays,row):array[ix]=value
  self.birth[ix]=birth
 def draw(self,actor,n):
  available=np.flatnonzero(self.seen[actor]>0);assert len(available),'No real bank rows for actor'
  indices=[]
  for _ in range(n):
   bucket=int(available[int(self.cursor[actor])%len(available)]);self.cursor[actor]+=1
   j=int(self.draw_rng.randint(min(int(self.seen[actor,bucket]),self.per_bin)));indices.append(self.base(actor,bucket)+j);self.sampled[actor,bucket]+=1
  ix=np.asarray(indices,np.int64);assert np.all(self.birth[ix]>=0),'Unfilled bank slots are never experience'
  return [x[ix].copy() for x in self.arrays],self.birth[ix].copy()
 def draw_actors(self,actors):
  actors=np.asarray(actors,np.int64)
  # Replay supplies ascending actor chunks. The usual full-coverage path avoids
  # per-row Python work; partial initial coverage uses only filled reservoirs.
  assert np.all(actors[:-1]<=actors[1:])
  if np.any(self.seen[np.unique(actors)]==0):
   output=[[] for _ in self.arrays];birth=[]
   for actor in np.unique(actors):
    data,ids=self.draw(int(actor),int(np.sum(actors==actor)))
    for target,x in zip(output,data):target.append(x)
    birth.append(ids)
   return [np.concatenate(x) for x in output],np.concatenate(birth)
  counts=np.bincount(actors,minlength=self.actors);starts=np.repeat(np.cumsum(counts)-counts,counts)
  local=np.arange(len(actors))-starts;buckets=(self.cursor[actors]+local)%len(self.periods)
  valid=np.minimum(self.seen[actors,buckets],self.per_bin)
  j=(self.draw_rng.random_sample(len(actors))*valid).astype(np.int64)
  ix=(actors*len(self.periods)+buckets)*self.per_bin+j
  self.cursor+=counts;np.add.at(self.sampled,(actors,buckets),1)
  assert np.all(self.birth[ix]>=0)
  return [x[ix].copy() for x in self.arrays],self.birth[ix].copy()
 def snapshot(self):
  filled=np.minimum(self.seen,self.per_bin)
  return dict(capacity=self.capacity,valid_rows=int(filled.sum()),rows_per_bin=self.per_bin,retention=self.retention,period_centers_ms=self.periods.tolist(),seen_by_actor_period=self.seen.tolist(),retained_by_actor_period=filled.tolist(),sampled_by_actor_period=self.sampled.tolist(),minimum_valid_birth=int(self.birth[self.birth>=0].min()) if np.any(self.birth>=0) else None,definition='Already validated real transitions per actor/period: uniform reservoir or last-N ring, explicitly configured. Bank/FIFO copies do not count as new samples, warmup or physical update credit. Periods are nearest normalized/log-duration neighborhoods, not exact action bins.')
 def save(self,path):
  valid=self.birth>=0;keys=['s0','a','reward','s1','terminal','duration','discount']
  payload={k:v for k,v in zip(keys,self.arrays)}
  payload.update(birth=self.birth,seen=self.seen,sampled=self.sampled,cursor=self.cursor,retention=np.array(self.retention))
  np.savez(str(path),**payload)
