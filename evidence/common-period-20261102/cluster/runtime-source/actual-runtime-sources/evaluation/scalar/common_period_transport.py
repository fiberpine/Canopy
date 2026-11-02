"""Versioned binary state/snapshot envelopes, no TCP control implementation."""
import base64,hashlib,io,json
import numpy as np
from common_period import SCHEMA

def pack_row(row):
    out=dict(row)
    for name in ('s0','s1','action'):
        if out.get(name) is not None:
            out[name]=base64.b64encode(np.asarray(out[name],np.float32).tobytes()).decode('ascii')
    return out

def unpack_row(row):
    out=dict(row)
    for name,size in [('s0',149),('s1',149),('action',2)]:
        if out.get(name) is not None:
            x=np.frombuffer(base64.b64decode(out[name],validate=True),np.float32)
            if x.shape!=(size,) or not np.isfinite(x).all():raise ValueError('Invalid boundary array '+name)
            out[name]=x.copy()
    return out

def snapshot_bytes(snapshot):
    values=dict(snapshot['values'])
    meta={k:v for k,v in snapshot.items() if k!='values'}
    values['__metadata__']=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),np.uint8)
    stream=io.BytesIO();np.savez_compressed(stream,**values)
    return stream.getvalue()

def decode_snapshot(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        meta=json.loads(z['__metadata__'].tobytes().decode())
        if meta['schema']!=SCHEMA:raise ValueError('Snapshot schema mismatch')
        values={n:z[n].copy() for n in z.files if n!='__metadata__'}
    if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite policy snapshot')
    return dict(meta,values=values)

class Policy:
    """Actor-only local TF inference; immutable moving stats between snapshots."""
    def __init__(self,checkpoint,reward_checksum):
        import tensorflow as tf
        from common_period_tf import ONLINE,Model
        self.tf=tf;self.graph=tf.Graph();self.version=0;self.reward_checksum=reward_checksum
        with self.graph.as_default():
            reader=tf.train.NewCheckpointReader(checkpoint);self.v={}
            for n in sorted(reader.get_variable_to_shape_map()):
                if n.startswith(ONLINE) and '/Adam' not in n:
                    self.v[n]=tf.get_variable(n,initializer=reader.get_tensor(n),trainable=False)
            self.v[ONLINE+'period_w']=tf.get_variable(ONLINE+'period_w',[1,256],initializer=tf.zeros_initializer(),trainable=False)
            self.s=tf.placeholder(tf.float32,[None,149]);self.t=tf.placeholder(tf.float32,[None,1]);self.mu,_=Model.actor(self,self.s,self.t,ONLINE)
            self.assign_inputs={n:tf.placeholder(tf.float32,v.shape) for n,v in self.v.items()}
            self.assigns=[tf.assign(v,self.assign_inputs[n]) for n,v in self.v.items()]
            init=tf.global_variables_initializer()
        self.session=tf.Session(graph=self.graph,config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1));self.session.run(init)

    def load(self,blob):
        s=decode_snapshot(blob)
        if s['reward_checksum']!=self.reward_checksum or s['version']<self.version:raise ValueError('Snapshot identity/regression')
        if set(s['values'])!=set(self.v):raise ValueError('Incomplete published actor')
        for n,x in s['values'].items():
            if list(x.shape)!=self.v[n].shape.as_list():raise ValueError('Policy shape mismatch')
        self.session.run(self.assigns,{self.assign_inputs[n]:x for n,x in s['values'].items()})
        self.version=int(s['version'])

    def window(self,state,period):
        from common_period import encoded_period
        if self.version<=0:raise ValueError('No published policy loaded')
        return float(self.session.run(self.mu,{self.s:np.asarray(state,np.float32)[None,:],self.t:[[encoded_period(period)]]})[0,0])
