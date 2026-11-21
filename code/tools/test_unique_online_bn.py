import importlib.util
from pathlib import Path
import unittest

p=Path(__file__).resolve().parents[1]/'frameworks/baseline/unique_online_bn.py'
s=importlib.util.spec_from_file_location('unique_online_bn',p);module=importlib.util.module_from_spec(s);s.loader.exec_module(module)


class Op:
    def __init__(self,name,kind='VariableV2',inputs=(),controls=()):
        self.name=name;self.type=kind;self.inputs=[Tensor(x) for x in inputs];self.control_inputs=list(controls)


class Tensor:
    def __init__(self,op):self.op=op;self.name=op.name+':0'


def entries():
    result=[]
    for layer in ('bn1','bn2'):
        for stat in ('moving_mean','moving_variance'):
            var=Op('abstract_actor/abstract_actor/'+layer+'/'+stat)
            for prefix in ('abstract_actor/','verifiable_predictor/abstract_actor/'):
                update=Op(prefix+layer+'/'+stat+'/write','AssignSub',(Op('ref','RefSwitch',(var,)),))
                result.append(Tensor(Op(prefix+layer+'/'+stat+'/Merge','Merge',(update,))))
    return result


class UniqueOnlineBN(unittest.TestCase):
    def test_eight_fetchable_wrappers_become_four(self):
        values=entries();chosen,audit=module.select_online_updates(values)
        self.assertEqual((len(chosen),audit['original_entries']),(4,8))
        self.assertTrue(all(x in values and x.op.type=='Merge' for x in chosen))
        self.assertEqual(len({x['destination'] for x in audit['statistics']}),4)

    def test_choice_is_canonical_even_if_collection_reversed(self):
        values=entries();a,_=module.select_online_updates(values);b,_=module.select_online_updates(values[::-1])
        self.assertEqual([x.name for x in a],[x.name for x in b])
        self.assertTrue(all(x.name.startswith('abstract_actor/') for x in a))

    def test_missing_statistic_rejected(self):
        with self.assertRaises(ValueError):module.select_online_updates(entries()[:-2])

    def test_wrapper_without_writer_rejected(self):
        with self.assertRaises(ValueError):module.select_online_updates(entries()+[Tensor(Op('empty','NoOp'))])

    def test_multiple_destinations_rejected(self):
        values=entries();extra=Tensor(Op('ambiguous','NoOp',controls=(values[0].op,values[2].op)))
        with self.assertRaises(ValueError):module.select_online_updates(values+[extra])

    def test_target_statistics_do_not_count_as_online(self):
        var=Op('target_actor/abstract_actor/bn1/moving_mean');update=Op('target/write','AssignSub',(var,))
        with self.assertRaises(ValueError):module.select_online_updates(entries()+[Tensor(update)])


if __name__=='__main__':unittest.main()
