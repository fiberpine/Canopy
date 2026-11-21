import pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'frameworks/mtpjoint'))
from common_period_metrics import network

class PacketWindows(unittest.TestCase):
    def test_mahimahi_drop_count_and_window_quantiles(self):
        # Mahimahi d records carry packet count followed by byte count.
        text='''# base timestamp: 1000
1000 + 1000
1001 + 500
1002 + 500
1003 d 2 1000
1004 # 1000
1005 - 1000 2
12000 # 1000
12001 - 500 6
61000 # 1000
'''
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'down.log';p.write_text(text);full,start,trim=network(p)
        self.assertEqual(full['utilization'],.75)
        self.assertEqual(full['drop_fraction'],2/3)
        self.assertEqual(full['dropped_bytes'],1000)
        self.assertEqual(full['arrival_bytes'],2000)
        self.assertEqual(full['queue_mean_ms'],4)
        self.assertAlmostEqual(full['queue_p95_ms'],5.8)
        self.assertEqual(start['queue_p95_ms'],2)
        self.assertEqual(trim['queue_p95_ms'],6)
        self.assertTrue(all(w['complete_window'] for w in [full,start,trim]))

if __name__=='__main__':unittest.main()
