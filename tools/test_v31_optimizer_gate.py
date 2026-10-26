import pathlib,sys,unittest,types
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'frameworks/mtpjoint'))
import independent_critic_v34 as fix
class OptimizerGateTest(unittest.TestCase):
 def test_requires_explicit_boolean(self):
  for bad in ['false',1,None]:
   with self.assertRaises(ValueError):fix.enabled({fix.FLAG:bad})
 def test_disabled_install_preserves_methods(self):
  class A:
   def build_critic_train_op_huber(self):return 'original-configured'
   def build_critic_train_op(self):return 'original-mse'
  before=A.build_critic_train_op;fix.install(types.SimpleNamespace(Agent=A),{})
  self.assertIs(A.build_critic_train_op,before)
 def test_same_process_can_return_to_legacy(self):
  class A:
   CDQ=True
   def build_critic_train_op_huber(self):return 'original-configured'
   def build_critic_train_op(self):return 'original-mse'
  m=types.SimpleNamespace(Agent=A);fix.install(m,{fix.FLAG:True});fix.install(m,{fix.FLAG:False})
  self.assertEqual(A().build_critic_train_op(),'original-mse');self.assertEqual(A().build_critic_train_op_huber(),'original-configured')
if __name__=='__main__':unittest.main()
