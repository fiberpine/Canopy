"""Cross-check the delivered entrypoint against actual accepted run evidence."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]


class CurrentBaseline(unittest.TestCase):
    def test_learning_template_matches_executed_recipe(self):
        actual=json.loads((ROOT/'evidence/baseline-online-bn-repair-20261112/training-complete.json').read_bytes())['config']
        template=json.loads((ROOT/'templates/baseline-train.json').read_bytes())
        keys=('seed','num_actors','conditions','memsize','sampler','warmup','update_ratio','LOSS_TYPE','reward_mode',
            'control_period_ms','exploration_std','tf_threads','checkpoint_interval','target_bn_consistent',
            'independent_critic_adam_v34','unique_critic_targets','unique_online_bn_updates','minimum_enqueued_per_actor',
            'require_coverage_gate','performance_guard_policy','sample_budget')
        for key in keys:self.assertEqual(template[key],actual[key],key)

    def test_selected_payload_hashes_and_actual_capture_are_preserved(self):
        current=json.loads((ROOT/'CURRENT_BASELINE.json').read_bytes())
        for name,info in current['models_and_natives'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),info['sha256'],name)
        deployment=json.loads((ROOT/'evidence/baseline-health-validation-20261112/frozen-deployment.json').read_bytes())
        for name,sha in deployment['helpers'][0]['helper_sha256'].items():
            if name=='paired_frozen_worker.py':continue
            self.assertEqual(hashlib.sha256((ROOT/'frameworks/baseline'/name).read_bytes()).hexdigest(),sha)

    def test_default_evaluation_selects_the_current_model_and_protocol(self):
        spec=importlib.util.spec_from_file_location('current_cli',ROOT/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as temporary:
            m.P=Path(temporary);(m.P/'templates').mkdir();(m.P/'frameworks/baseline').mkdir(parents=True)
            (m.P/'templates/baseline-eval.json').write_bytes((ROOT/'templates/baseline-eval.json').read_bytes())
            with mock.patch.object(sys,'argv',['run.py','evaluate','--variant','baseline','--run-id','current']),mock.patch.object(m,'call',side_effect=RuntimeError('before TCP')) as call:
                with self.assertRaisesRegex(RuntimeError,'before TCP'):m.main()
            cfg=json.loads((m.P/'frameworks/baseline/configs/current.json').read_bytes())
            self.assertIn('healthy-baseline-seed3',cfg['checkpoint'])
            self.assertEqual(cfg['baseline_native_startup'],'natural-cubic')
            self.assertTrue(cfg['no_training']);self.assertTrue(cfg['require_natural_handoff'])
            self.assertEqual(Path(call.call_args[0][0][0]).name,'capture_frozen.py')

    def test_postcondition_accepts_observed_handoff_and_rejects_optimization(self):
        spec=importlib.util.spec_from_file_location('postcondition',ROOT/'frameworks/baseline/frozen_startup_contract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        evidence=json.loads((ROOT/'evidence/baseline-health-validation-20261112/natural-completed-source-and-handoff-audit.json').read_bytes())
        actual=next(r for h in evidence['proof'] for r in h['records'] if r['job']['run']=='natural-00')
        with tempfile.TemporaryDirectory() as temporary:
            f=Path(temporary);policy='631683b939596cabc85262e2294123c462913bf6b836dff10b84501644ad60e0'
            result=dict(status='COMPLETED',policy_sha256_before=policy,policy_sha256_after=policy,counters=actual['counters'])
            (f/'result.json').write_text(json.dumps(result))
            (f/'server.log').write_text('ORCA_STARTUP_HANDOFF '+' '.join('%s=%s'%x for x in actual['startup'].items())+'\n')
            with (f/'control.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(actual['first_controls'][0]));writer.writeheader();writer.writerows(actual['first_controls'])
            base=float(next(x.split(':',1)[1] for x in actual['headers'] if x.startswith('# base timestamp:')))
            # Synthetic complete tail for the protocol unit check, not network evidence.
            (f/'down.log').write_text('\n'.join(actual['headers'])+'\n%d # 1504\n'%int(base+60000))
            self.assertEqual(m.verify(f)['status'],'PASS')
            result['counters']['actor_updates']=1;(f/'result.json').write_text(json.dumps(result))
            with self.assertRaises(AssertionError):m.verify(f)


if __name__=='__main__':unittest.main()
