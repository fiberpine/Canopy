"""Read full training checkpoint; export exactly the actor/BN for frozen deployment only."""
import argparse,pathlib,json,hashlib,os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
source=tf.train.NewCheckpointReader(a.source);names=sorted(n for n in source.get_variable_to_shape_map() if n.startswith('actor/') and '/Adam' not in n)
assert names and not pathlib.Path(a.output+'.index').exists()
graph=tf.Graph()
with graph.as_default():
    variables=[tf.get_variable(n,initializer=source.get_tensor(n)) for n in names]
    init=tf.global_variables_initializer();saver=tf.train.Saver(var_list=variables)
with tf.Session(graph=graph,config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)) as sess:
    sess.run(init);saver.save(sess,a.output,write_meta_graph=False)
reader=tf.train.NewCheckpointReader(a.output)
assert all(np.array_equal(source.get_tensor(n),reader.get_tensor(n)) for n in names)
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path(a.output).parent.glob(pathlib.Path(a.output).name+'.*') if p.suffix=='.index' or '.data-' in p.name}
receipt=dict(source_full_checkpoint=a.source,export_checkpoint=a.output,actor_and_BN_variables=len(names),all_exported_values_equal=True,files=files,purpose='Actor-only frozen deployment; never a full training resume checkpoint')
pathlib.Path(a.output+'.deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
