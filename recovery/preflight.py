"""Verify the restored host without launching TCP or training."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from frameworks.joint_healthy.configuration import configuration, runtime_preflight
from frameworks.joint_healthy.identity import implementation_identity, check_admission, validate_native
from frameworks.joint_healthy.model import Model
import numpy as np
import tensorflow as tf


def main():
    cfg = configuration('continuous-v4')
    environment = runtime_preflight()
    validate_native()
    manifest = json.loads((ROOT/'CURRENT_BASELINE.json').read_text())
    for name, expected in manifest['models_and_natives'].items():
        actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual != expected['sha256']:
            raise ValueError('Baseline integrity mismatch: '+name)
    receipt = json.loads((ROOT/'recovery/evidence/initialization-admission.json').read_text())
    check_admission(receipt, cfg)
    tf.set_random_seed(3)
    model = Model(cfg, False)
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as sess:
        sess.run(tf.global_variables_initializer())
        model.initialize_baseline(sess,cfg['baseline_checkpoint'])
        prediction = sess.run(model.action,{model.s:np.zeros((1,82),np.float32)})
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite restored model output')
        policy = model.digest(sess)
    print(json.dumps(dict(status='PASS_RESTORED_HOST_NO_TCP',environment=environment,
        tensorflow=tf.__version__,policy_sha256=policy,source=implementation_identity(True))))


if __name__=='__main__':
    main()
