import pathlib,sys,json,csv,unittest
D=pathlib.Path(__file__).resolve().parents[1]/'evidence/common-period-capture48-20261103'
sys.path.insert(0,str(D/'runtime-source'))
from test_capture import CaptureContracts
class CaptureEvidence(unittest.TestCase):
 def test_registered_counts_and_downgrades(self):
  reg=json.loads((D/'registration.json').read_text());m=json.loads((D/'captured-run-metrics.json').read_text());r=json.loads((D/'results.json').read_text());self.assertEqual(len(reg['jobs']),48);self.assertEqual(len({x['run'] for x in m}),48);self.assertTrue(all(x['status']=='COMPLETED' for x in m));cells={}
  for x in m:cells.setdefault((x['job']['condition'],x['job']['period_ms']),[]).append(x['instrumentation_pass'])
  self.assertEqual(len(cells),24);self.assertTrue(all(len(v)==2 for v in cells.values()));self.assertEqual(sum(not all(v) for v in cells.values()),r['downgraded_cells']);self.assertEqual(sum(x['capture_checks']['rows'] for x in m),r['captured_transitions']);self.assertEqual(r['optimizer_operations'],0)
