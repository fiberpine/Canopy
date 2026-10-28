"""Stdlib tests for the limited-learning-check tooling (no TF, no TCP, no servers)."""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/scalar_learning_check'))
sys.path.insert(0, str(ROOT / 'tools/scalar_ipc_diagnostic'))
import adjudicate_learning as al  # noqa: E402
import learning_monitor as lm  # noqa: E402
import make_learning_registration as mlr  # noqa: E402
import window_binding as wb  # noqa: E402


class MonitorTest(unittest.TestCase):
    def test_self_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = lm.self_test(tmp)
        self.assertEqual(r['first_trigger_s'], {'S3_STOP': 40.0})


class AdjudicatorTest(unittest.TestCase):
    def test_missing_or_bad_tail_blocks_long_training(self):
        rows,cells=al._synthetic_rows('good')
        plan=dict(evaluation=dict(cells=cells),tail_guard=True)
        self.assertNotEqual(al.adjudicate(rows,plan,dict(stop=False,rules=[]))['decision'],'GO_LONG_TRAINING')
        for row in rows:row.update(queue_p95_ms=40.0,queue_p99_ms=60.0)
        self.assertEqual(al.adjudicate(rows,plan,dict(stop=False,rules=[]))['decision'],'GO_LONG_TRAINING')
        for row in rows:
            if row['checkpoint']=='1200000':row['queue_p99_ms']=1000.0
        self.assertNotEqual(al.adjudicate(rows,plan,dict(stop=False,rules=[]))['decision'],'GO_LONG_TRAINING')

    def test_all_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = al.self_test(tmp)
        self.assertEqual({k: v[0] for k, v in r.items()},
                         dict(anchoring='ANCHORING_LIMITED', policy='POLICY_LIMITED', good='GO_LONG_TRAINING',
                              static='LONG_RTT_STATIC', collapse='COLLAPSE_REPRODUCED'))


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.plan = mlr.build('TEST')

    def test_training_changes_only_validated_components_and_budget(self):
        cfg = self.plan['training_config']
        self.assertEqual(cfg['exploration_process'], 'zero_sum_offset')
        self.assertEqual(cfg['exploration_offset_rho'], 0.0)
        self.assertEqual(cfg['scalar_window_initialization'], 'small_uniform')
        self.assertTrue(cfg['verified_ipc_v2'] and cfg['orca_actor_loss_moving_stats'])
        self.assertEqual((cfg['sample_budget'], cfg['checkpoint_interval']), (1200000, 100000))
        self.assertEqual(len(cfg['conditions']), 24)
        for key in ('reward_mode', 'update_ratio', 'warmup', 'memsize', 'sampler', 'exploration_std', 'LOSS_TYPE'):
            self.assertNotIn(key, self.plan['differences_from_historical_moving_template'])

    def test_evaluation_jobs(self):
        jobs = self.plan['evaluation']['jobs']
        self.assertEqual(len(jobs), 110)
        self.assertEqual(sum(j['checkpoint'] == 1200000 and j['arm'] == 'A0' for j in jobs), 20)
        a4g = {j['checkpoint'] for j in jobs if j['arm'] == 'A4g' and j['checkpoint'] != 'old'}
        self.assertEqual(a4g, {400000, 1200000})
        self.assertEqual(sum(j['arm'] == 'CUBIC' for j in jobs), 10)
        self.assertEqual(sum(j['checkpoint'] == 'old' for j in jobs), 20)


class WindowBindingTest(unittest.TestCase):
    def test_rtt_proxy_separates_jitter_from_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = wb.self_test(tmp)
        self.assertEqual(r['capped']['trim10']['binding_rtt_fraction'], 0.0)
        self.assertLess(r['jitter']['trim10']['binding_fraction'], 0.6)
        self.assertGreater(r['jitter']['trim10']['binding_rtt_fraction'], 0.9)


class ActorSpeedTest(unittest.TestCase):
    """Synthetic constant-gradient illustration, not universal scale independence."""
    def test_speed_is_independent_of_gradient_magnitude(self):
        import actor_speed_sim as sim
        big = sim.first(sim.run(c=0.02, seed=3, steps=120), .9)
        tiny = sim.first(sim.run(c=0.001, seed=3, steps=120), .9)
        frozen = sim.first(sim.run(c=0.02, seed=3, steps=120, bn_mode='frozen'), .9)
        self.assertTrue(55 <= big <= 110 and 55 <= tiny <= 110 and 55 <= frozen <= 110, (big, tiny, frozen))
        self.assertLessEqual(abs(big - tiny), 15)


class ValueHorizonTest(unittest.TestCase):
    """Soft-target propagation model (exact on a chain) and its fit to the published R3/R5 critic records."""
    def test_model_identities(self):
        import value_horizon as vh
        r = vh.self_test()
        self.assertAlmostEqual(r['lambda_recovered'], 7.5, places=5)
        self.assertAlmostEqual(r['poisson_approx'], r['chain_exact'], delta=.02 * r['chain_exact'])

    def test_published_records(self):
        import value_horizon as vh
        out = vh.analyze_evidence(ROOT / 'evidence/scalar-first-updates-20261027', reward_mean=0.706)
        s = out['summary']
        self.assertEqual(s['independent_critic_phases'], 3)
        lo, hi = s['implied_reward_mean_range']
        self.assertTrue(.55 < lo <= hi < .65, (lo, hi))
        self.assertTrue(all(1500 <= u <= 2500 for u in s['rtt20_first_nonpositive']), s['rtt20_first_nonpositive'])
        self.assertEqual(s['predicted_sign_change_tau001']['200'], 20000.0)


class ExplorationResponseTest(unittest.TestCase):
    def test_series_intercepts_remove_between_connection_bias(self):
        import numpy as np
        import exploration_response as er
        rng = np.random.RandomState(23)
        series = []
        for level in (-20., 20.):
            x = level + rng.randn(400)
            y = 100 * level + .3 * (x - level)
            series.append((x, y))
        out = er.response(series, max_lag=0)
        self.assertAlmostEqual(out['beta'][0], .3, places=10)
        self.assertIn('HAC', out['standard_errors'])

    def test_hac_preserves_class_time_gaps_and_connections(self):
        import numpy as np
        import exploration_response as er
        x = np.array([-2., 1., 3., -1., 2., -3.])
        y = np.array([1., 4., -2., 3., 0., 2.])[:, None]
        # Two separate connections: gaps must not be compressed and the boundary
        # must not manufacture a lagged covariance.
        segments = [(0, 8, np.array([0, 2, 7])), (3, 7, np.array([0, 1, 6]))]
        beta, se = er._ols(x, y, segments, hac_lags=2)
        xc = np.concatenate([x[:3]-x[:3].mean(), x[3:]-x[3:].mean()])
        yc = np.concatenate([y[:3, 0]-y[:3].mean(), y[3:, 0]-y[3:].mean()])
        scores = xc * (yc - beta[0] * xc)
        meat = float(scores @ scores)
        # Only times 0 and 2 of connection 1, and 0 and 1 of connection 2,
        # are within the bandwidth. Bartlett weights are 1/3 and 2/3.
        meat += 2 * (scores[0]*scores[1]/3 + 2*scores[3]*scores[4]/3)
        self.assertAlmostEqual(se[0], np.sqrt(meat) / (xc @ xc), places=12)

    def test_self_test_recovers_lags_and_class_signs(self):
        import exploration_response as er
        r = er.self_test()
        self.assertGreater(r['split_beta1']['free'], .04)
        self.assertLess(r['split_beta1']['penalized'], -.04)

    def test_states_csv_and_npz_inputs(self):
        import csv
        import numpy as np
        import exploration_response as er
        rng = np.random.RandomState(7)
        conds = [dict(base_rtt_ms=r) for r in (20, 50, 100, 200)]
        with tempfile.TemporaryDirectory() as tmp:
            arrays = dict(o=[], r=[], rtt=[], cls=[], series_id=[])
            for i, cond in enumerate(conds):
                n = 600
                o = .2 / np.sqrt(2) * rng.randn(n)
                pi = rng.uniform(-.01, .01, n)
                act = pi + o - np.concatenate([[0.], o[:-1]])
                delay = np.where(rng.rand(n) < .5, 1.0, .7)
                cls = np.concatenate([[er.UNKNOWN], np.where(delay[:-1] < .999, er.PENALIZED, er.FREE)])
                rew = .7 + .02 * rng.randn(n)
                rew[1:] += np.where(cls[:-1] == er.FREE, .05, -.05) * o[:-1] * (cls[:-1] != er.UNKNOWN)
                d = pathlib.Path(tmp) / 'logs' / ('run-actor%d' % i)
                d.mkdir(parents=True)
                with (d / 'states.csv').open('w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['action', 'action_no_noise', 'reward', 'delay_term'])
                    w.writerows(zip(act, pi, rew, delay))
                for k, v in (('o', er.offsets(act, pi)), ('r', rew), ('rtt', np.full(n, cond['base_rtt_ms'])), ('cls', cls),
                             ('series_id', np.full(n, i))):
                    arrays[k].append(v)
            res = er.analyze(tmp, 'run', dict(conditions=conds), max_lag=5)
            self.assertTrue(res['offset_reconstruction_ok'])
            self.assertEqual(res['by_rtt']['20']['class_counts']['unknown'], 1)
            path = pathlib.Path(tmp) / 'series.npz'
            np.savez(str(path), **{k: np.concatenate(v) for k, v in arrays.items()})
            res2 = er.analyze_npz(path, max_lag=5)
        for out in (res, res2):
            for r in ('20', '200'):
                self.assertGreater(out['by_rtt'][r]['free']['beta'][1], 0)
                self.assertLess(out['by_rtt'][r]['penalized']['beta'][1], 0)


if __name__ == '__main__':
    unittest.main()
