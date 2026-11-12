"""The reusable CLI must select the whole audited contract explicitly."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class ScalarContractCLI(unittest.TestCase):
    def _configuration(self, online=False):
        spec = importlib.util.spec_from_file_location('contract_cli', ROOT/'run.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        sys.path.insert(0, str(ROOT/'frameworks/baseline'))
        with tempfile.TemporaryDirectory() as temporary:
            m.P = Path(temporary)
            (m.P/'templates').mkdir()
            (m.P/'frameworks/baseline').mkdir(parents=True)
            old = json.loads((ROOT/'templates/baseline-train.json').read_text())
            (m.P/'templates/baseline-train.json').write_text(json.dumps(old))
            read = Path.read_text
            def reading(path, *a, **kw):
                if str(path).replace('\\', '/').endswith('/proc/sys/net/ipv4/ip_local_port_range'):
                    return '32768 60999'
                return read(path, *a, **kw)
            argv = ['run.py', 'train', '--variant', 'baseline', '--run-id', 'contract',
                    '--actors', '24', '--scalar-update-contract', '--port', '10000',
                    '--learner-port', '20000', '--server-binary', str(m.P/'native')]
            if online:argv.append('--unique-online-bn-updates')
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(Path, 'read_text', reading), mock.patch.object(m, 'call', side_effect=RuntimeError('before any launch')):
                with self.assertRaisesRegex(RuntimeError, 'before any launch'):
                    m.main()
            cfg = json.loads((m.P/'frameworks/baseline/configs/contract-input.json').read_text())
            for flag in ('target_bn_consistent', 'independent_critic_adam_v34', 'unique_critic_targets', 'verify_owned_cleanup'):
                self.assertIs(cfg[flag], True)
            for key in ('sampler', 'warmup', 'update_ratio', 'LOSS_TYPE', 'reward_mode', 'memsize', 'exploration_std', 'control_period_ms'):
                self.assertEqual(cfg[key], old[key])
            self.assertEqual(cfg['server_binary'], str((m.P/'native').resolve()))
            self.assertNotIn('episode_seconds', cfg)
            self.assertEqual(cfg.get('unique_online_bn_updates',False),online)

    def test_explicit_contract_selects_repairs_and_preserves_recipe(self):
        self._configuration()

    def test_online_bn_fix_requires_explicit_flag_and_preserves_recipe(self):
        self._configuration(True)


if __name__ == '__main__':
    unittest.main()
