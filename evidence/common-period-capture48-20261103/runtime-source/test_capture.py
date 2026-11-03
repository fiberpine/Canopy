import importlib.util,sys,types,tempfile,pathlib,json,unittest,numpy as np
from unittest.mock import patch
from capture_core import interval,window_checks,finite_suffix
HERE=pathlib.Path(__file__).parent
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,str(HERE/file));x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);return x
class CaptureContracts(unittest.TestCase):
    def test_reward_and_center(self):
        r=np.ones(15);r[1]=6000000;r[2]=1;r[3]=.02;r[7]=0;r[0]=50;r[8]=20;r[14]=20
        x=interval(r,100);self.assertAlmostEqual(x['reward'],.5);self.assertAlmostEqual(x['centered'],0)
    def test_tolerance_no_relaxation(self):
        limit=dict(utilization_absolute=.03,mean_queue_absolute_ms=2,mean_queue_relative=.2,p95_absolute_ms=3,p95_relative=.2);old=dict(utilization=.9,queue_mean_ms=10,queue_p95_ms=20);new=dict(utilization=.86,queue_mean_ms=10,queue_p95_ms=20);self.assertFalse(window_checks(old,new,limit)[0]['passed'])
    def test_suffix_tail_and_no_cross_end(self):
        d=np.full(1200,.02);x=finite_suffix(np.full(1200,.5),d,0,100,10,200);self.assertGreaterEqual(x['H'],20);self.assertAlmostEqual(x['Q_label'],x['Q_tail'],places=8)
        with self.assertRaises(ValueError):finite_suffix(np.full(5,.5),d[:5],0,100,0,0)
    def test_driver_events_identical_and_capture_aligned(self):
        original=module('original','original_deployment.py');capture=module('capture','common_period_deployment.py')
        class End(Exception):pass
        fake={n:types.ModuleType(n) for n in ['joint_mtp','decision_clock_v21','observation_noise_v14','timing_context_v17','feedback_context_v31','evaluate']}
        fake['joint_mtp'].encode_state=lambda h,dt,plan,*args:np.r_[h,np.full(79,dt+plan)].astype(np.float32);fake['joint_mtp'].mtp_ms=lambda a,c:80.
        fake['decision_clock_v21'].validate=lambda c:None;fake['decision_clock_v21'].planned_us=lambda a,c:80000;fake['decision_clock_v21'].wire=lambda i,a,c:('wire:%d:%s'%(i,None if a is None else a.tolist())).encode()
        class Obs:
            def __init__(self,c):self.last_multiplier=1
            def apply(self,x):return x
        class Ctx:
            def __init__(self,c):pass
            def update(self,*a):pass
            def observe(self,*a):pass
        fake['observation_noise_v14'].Observation=Obs;fake['timing_context_v17'].Context=Ctx;fake['feedback_context_v31'].Feedback=Ctx;fake['evaluate'].EndOfRun=End
        class Stop:
            def is_set(self):return False
        import time
        outputs=[]
        with patch.dict(sys.modules,fake):
            for driver in [original,capture]:
                events=[]
                class Reader:
                    last_id=657;last_raw=np.ones(15);expected_interval_s=0
                    server=types.SimpleNamespace(poll=lambda:0,created=time.time()-100)
                reader=Reader();reader.last_raw[3]=.02
                class Env:
                    index=0
                    def reset(self):events.append('reset');return np.ones(7)
                    def get_state(self,**kw):
                        self.index+=1
                        if self.index>3:raise End()
                        reader.last_id+=1;reader.last_raw=np.ones(15)*self.index;reader.last_raw[3]=.02;events.append('feedback');return np.ones(7)*self.index,0,.5,True
                def act(s):events.append(('infer',s.tolist()));return np.array([.1,.3333333],np.float32)
                writer=types.SimpleNamespace(write=lambda x:events.append(('write',x)))
                with tempfile.TemporaryDirectory() as d:
                    root=pathlib.Path(d);count=driver.drive(Env(),reader,writer,act,dict(no_training=True,capture_raw_state=True,seconds=60,absolute_deadline_epoch=time.time()+100),Stop(),None,root)
                    self.assertEqual(count,3)
                    if driver is capture:
                        self.assertFalse((root/'captured-boundaries.npz').exists());capture.flush_captures()
                        with np.load(str(root/'captured-boundaries.npz'),allow_pickle=False) as z:
                            self.assertEqual(z['s0'].shape,(3,149));self.assertEqual(z['raw_next'].shape,(3,15));np.testing.assert_array_equal(z['s0'][1:],z['s1'][:-1]);np.testing.assert_array_equal(z['next_state_id'],[658,659,660])
                outputs.append(events)
        self.assertEqual(outputs[0],outputs[1])
if __name__=='__main__':unittest.main()
