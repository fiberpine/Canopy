import copy
import unittest
from baseline_reproduction_check import aggregate,METRICS


def record(run,condition,value):
    metrics={k:0. for k in METRICS};metrics.update(utilization=value,queue_mean_ms=32.32)
    return dict(tag='candidate',repeat=0,host_index=0,
                result=dict(cfg=dict(run_id=run,condition=condition,seconds=60),
                            status='COMPLETED',wall_s=62,policy_sha256_before='abc',
                            policy_sha256_after='abc',counters={k:0 for k in (
                                'critic_iterations','critic_optimizer_ops','actor_updates',
                                'target_updates','exploration_calls')}),
                windows=[dict(metrics,condition=condition,window=w,end_s=60,
                              complete_window=True) for w in ('full','trim10')])


class FrozenAggregationTests(unittest.TestCase):
    def test_failed_short_duplicate_and_drift_cannot_be_counted_as_valid(self):
        original=[record('a','one',.99),record('b','two',.97)]
        self.assertTrue(aggregate(original,['one','two'],1)['registered_coarse_pass'])
        for mutate in (
                lambda r:r[0]['result'].update(wall_s=15),
                lambda r:r[0]['result'].update(policy_sha256_after='changed'),
                lambda r:r[0]['result']['counters'].update(actor_updates=1),
                lambda r:r[0]['windows'][0].update(queue_mean_ms=float('nan')),
                lambda r:r.append(copy.deepcopy(r[0]))):
            rows=copy.deepcopy(original);mutate(rows)
            with self.assertRaises(ValueError):aggregate(rows,['one','two'],1)

    def test_missing_condition_is_not_a_smaller_successful_grid(self):
        with self.assertRaises(ValueError):aggregate([record('a','one',.99)],['one','two'],1)

    def test_repeated_runs_average_within_condition(self):
        rows=[record('a','one',.8),record('b','one',1.),
              record('c','two',.98),record('d','two',1.)]
        result=aggregate(rows,['one','two'],2)
        self.assertAlmostEqual(result['trim10']['condition_equal']['utilization'],.945)
        self.assertFalse(result['registered_coarse_pass'])


if __name__=='__main__':unittest.main()
