"""Admission boundary checks, including the real portable CLI configuration."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frameworks/baseline'))
from baseline_admission import validate_ports,cleanup_report,drain_after_failure


class BaselineAdmission(unittest.TestCase):
    def test_cleanup_waits_for_transient_listener_but_rejects_persistent_listener(self):
        import baseline_admission as contracts
        fail=dict(status='FAIL',remaining=[],listeners=['LISTEN :10000'],accessible_shm_keys=[])
        good=dict(status='PASS',remaining=[],listeners=[],accessible_shm_keys=[])
        with mock.patch.object(contracts,'_cleanup_snapshot',side_effect=[fail,good]),mock.patch('time.sleep'):
            report=cleanup_report({},10000,[],None)
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(len(report['checks']),2)
        with mock.patch.object(contracts,'_cleanup_snapshot',return_value=fail):
            report=cleanup_report({},10000,[],None,timeout_seconds=0)
        self.assertEqual(report['status'],'FAIL')

    def test_failed_actor_records_are_forwarded_until_learner_finishes(self):
        forwarded={};calls=[0]
        def collect():
            calls[0]+=1
            return [dict(index=0,producer=dict(generated=5,enqueued=5),result=dict(status='FAILED'))]+([dict(index=1,producer=dict(generated=7,enqueued=7))] if calls[0]>1 else [])
        with mock.patch('time.sleep'):
            report=drain_after_failure(2,collect,lambda i,p:forwarded.update({i:p}),lambda:len(forwarded)==2)
        self.assertEqual(report['status'],'DRAIN_RECORDED_RUN_STILL_FAILED')
        self.assertEqual(set(forwarded),{0,1})

    def test_failure_drain_is_bounded_when_a_producer_is_missing(self):
        report=drain_after_failure(1,lambda:[],lambda i,p:None,lambda:False,timeout_seconds=0)
        self.assertEqual(report['status'],'DRAIN_INCOMPLETE_RUN_FAILED')
    def test_port_plan_rejects_learner_overlap_ephemeral_and_exhaustion(self):
        self.assertEqual(validate_ports(10000,24,20000,(32768,60999),1200,20)['maximum_connections'],62)
        for values in ((23000,24,25000,(32768,60999),1800,20),
                       (53000,24,25000,(32768,60999),1200,None),
                       (65000,24,25000,(32768,60999),1200,20)):
            with self.assertRaises(ValueError):validate_ports(*values)

    def test_real_cli_selects_bn_only_and_preserves_learning_recipe(self):
        spec=importlib.util.spec_from_file_location('tested_baseline_cli',ROOT/'run.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            package=Path(directory);(package/'templates').mkdir()
            (package/'frameworks/baseline').mkdir(parents=True)
            source=(ROOT/'templates/baseline-train.json').read_text()
            (package/'templates/baseline-train.json').write_text(source)
            # The production module remains on sys.path; only package output is temporary.
            module.P=package
            original=Path.read_text
            def reading(path,*args,**kwargs):
                if str(path).replace('\\','/').endswith('/proc/sys/net/ipv4/ip_local_port_range'):
                    return '32768 60999'
                return original(path,*args,**kwargs)
            args=['run.py','train','--variant','baseline','--run-id','probe','--actors','24',
                  '--samples','225000','--replay-probe','--target-bn-consistent',
                  '--checkpoint-interval','100000','--performance-guard-policy','warn',
                  '--port','10000','--learner-port','20000']
            with mock.patch.object(sys,'argv',args),mock.patch.object(Path,'read_text',reading),mock.patch.object(module,'call',side_effect=RuntimeError('intercepted before launch')):
                with self.assertRaisesRegex(RuntimeError,'intercepted before launch'):module.main()
            cfg=json.loads((package/'frameworks/baseline/configs/probe-input.json').read_text())
            old=json.loads(source)
            for key in ('sampler','warmup','update_ratio','LOSS_TYPE','reward_mode','memsize','exploration_std','control_period_ms'):
                self.assertEqual(cfg[key],old[key])
            self.assertTrue(cfg['target_bn_consistent'])
            self.assertTrue(cfg['capacity_replay_probe'])
            self.assertNotIn('episode_seconds',cfg)
            self.assertEqual(cfg['performance_guard_policy'],'warn')
            self.assertEqual(cfg['checkpoint_interval'],100000)
            self.assertEqual(cfg['sample_budget'],225000)
            self.assertEqual(cfg['steps_per_actor']*cfg['num_actors'],225000)
