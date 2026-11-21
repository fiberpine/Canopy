import copy
import csv
import pathlib
import unittest
import comprehensive_audit as audit


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (audit.ROOT / (audit.BASE + 'all-network-windows.csv')).open(newline='') as f:
            cls.rows = list(csv.DictReader(f))

    def test_recomputed_endpoint(self):
        result = audit.run()
        x = result['endpoint']['condition_equal']
        self.assertAlmostEqual(x['stopped_trim10']['utilization'], .446636598, places=8)
        self.assertAlmostEqual(x['archived_trim10']['utilization'], .979047218, places=8)
        self.assertEqual(result['timing']['G12_R200_slow_cycles'], 264)
        self.assertEqual(result['timing']['G12_R200_sample_wait_dominant'], 264)

    def test_duplicate_not_a_repeat(self):
        rows = copy.deepcopy(self.rows)
        rows[0] = rows[1]
        with self.assertRaises(ValueError):
            audit.summarize(rows)

    def test_truncated_not_complete(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['complete_window'] = 'False'
        with self.assertRaises(ValueError):
            audit.summarize(rows)

    def test_missing_window_rejected(self):
        with self.assertRaises(ValueError):
            audit.summarize(self.rows[:-1])

    def test_nonfinite_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['utilization'] = 'nan'
        with self.assertRaises(ValueError):
            audit.summarize(rows)

    def test_changed_condition_set_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['condition'] = 'not-the-matched-condition'
        with self.assertRaises(ValueError):
            audit.summarize(rows)

    def test_inherited_reward_is_not_monotone_in_delay_during_negative_utility(self):
        for values in audit.reward_source_fixtures().values():
            r = values['higher_delay_less_negative_reward']
            self.assertAlmostEqual(r['srtt40_reward'], -.125)
            self.assertAlmostEqual(r['srtt80_reward'], -.0625)


if __name__ == '__main__':
    unittest.main()
