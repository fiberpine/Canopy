"""A delivered policy must be selectable without replacing preserved defaults."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]


class FrozenPolicySelection(unittest.TestCase):
    def module(self):
        spec=importlib.util.spec_from_file_location('frozen_selection_cli',ROOT/'run.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_explicit_frozen_policy_and_native_do_not_change_template(self):
        m=self.module();original=(ROOT/'templates/baseline-eval.json').read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            m.P=Path(temporary);(m.P/'templates').mkdir()
            (m.P/'frameworks/baseline').mkdir(parents=True)
            (m.P/'templates/baseline-eval.json').write_bytes(original)
            prefix=m.P/'models/new-policy/model';native=m.P/'runtime/orca-server-cloexec'
            argv=['run.py','evaluate','--variant','baseline','--run-id','selected','--checkpoint',str(prefix),'--server-binary',str(native)]
            with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'call',side_effect=RuntimeError('before TCP launch')):
                with self.assertRaisesRegex(RuntimeError,'before TCP launch'):m.main()
            cfg=json.loads((m.P/'frameworks/baseline/configs/selected.json').read_text())
            self.assertEqual(cfg['checkpoint'],str(prefix.resolve()))
            self.assertEqual(cfg['server_binary'],str(native.resolve()))
            self.assertIs(cfg['no_training'],True)
            self.assertEqual(cfg['control_period_ms'],20)
            self.assertEqual((m.P/'templates/baseline-eval.json').read_bytes(),original)

    def test_frozen_checkpoint_flag_cannot_silently_warm_start_training(self):
        m=self.module()
        with mock.patch.object(sys,'argv',['run.py','train','--variant','baseline','--checkpoint','weights/model']):
            with self.assertRaisesRegex(ValueError,'frozen evaluation'):m.main()


if __name__=='__main__':unittest.main()
