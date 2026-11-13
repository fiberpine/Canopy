#!/usr/bin/env python3
"""Independent healthy-seed3 joint candidate. No command runs TCP by default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'build-native':
        return subprocess.call([sys.executable, str(ROOT / 'tools/build_joint_healthy_native.py')] + sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        del sys.argv[1]
        from frameworks.joint_healthy.backend import worker_main
        worker_main(); return
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='command')
    p = sub.add_parser('plan'); p.add_argument('--profile', default='joint'); p.add_argument('--output')
    p = sub.add_parser('verify-model'); p.add_argument('--output', required=True)
    sub.add_parser('verify-offline')
    sub.add_parser('identity')
    p = sub.add_parser('build-native'); p.add_argument('build_args', nargs=argparse.REMAINDER)
    for name in ('evaluate', 'train'):
        p = sub.add_parser(name); p.add_argument('--profile', default='g1' if name == 'evaluate' else 'joint')
        p.add_argument('--config'); p.add_argument('--authorization', required=True)
        p.add_argument('--run-id', required=True); p.add_argument('--seed', type=int)
        p.add_argument('--checkpoint'); p.add_argument('--scenario', type=int, default=0)
        p.add_argument('--seconds', type=float, default=60); p.add_argument('--seconds-per-actor', type=float, default=600)
        p.add_argument('--override-period-ms', type=float); p.add_argument('--decision-cost', type=float)
        p.add_argument('--release-joint', action='store_true'); p.add_argument('--gate-receipt')
    args = ap.parse_args()
    if args.command is None: ap.print_help(); return 0
    from frameworks.joint_healthy.configuration import configuration, checked_authorization
    from frameworks.joint_healthy.core import validate, period_action
    if args.command == 'plan':
        cfg = configuration(args.profile)
        if args.output:
            path = Path(args.output)
            if path.exists(): raise ValueError('Existing configuration must be preserved')
            path.write_text(json.dumps(cfg, indent=2) + '\n')
        print(json.dumps(cfg, indent=2)); return 0
    if args.command == 'verify-model':
        from tools.verify_joint_healthy_model import verify
        verify(args.output); return 0
    if args.command == 'verify-offline':
        return subprocess.call([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests/joint_healthy'), '-p', 'test_*.py'])
    if args.command == 'identity':
        from frameworks.joint_healthy.identity import implementation_identity
        print(json.dumps(implementation_identity(), indent=2)); return 0
    if args.command == 'build-native':
        return subprocess.call([sys.executable, str(ROOT / 'tools/build_joint_healthy_native.py')] + args.build_args)
    cfg = json.loads(Path(args.config).read_text().replace('@PACKAGE@',str(ROOT).replace('\\','/'))) if args.config else configuration(args.profile)
    cfg['run_id'] = args.run_id
    if args.seed is not None: cfg['seed'] = args.seed
    if args.decision_cost is not None: cfg['decision_cost'] = args.decision_cost
    validate(cfg)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', args.run_id): raise ValueError('Run ID must be a single safe path component')
    auth = checked_authorization(args.authorization)
    cfg['deadline_epoch'] = auth['deadline_epoch']
    if os.name != 'posix': raise RuntimeError('TCP commands require the prepared Linux/custom kernel; offline verification is available here')
    from frameworks.joint_healthy.configuration import runtime_preflight
    cfg['runtime_preflight'] = runtime_preflight()
    if args.command == 'train':
        if args.override_period_ms is not None or args.checkpoint:
            raise ValueError('Training starts from the declared baseline; no resume or inference override')
        cfg['physical_seconds_per_actor'] = args.seconds_per_actor
        if not 0 < args.seconds_per_actor < float('inf'): raise ValueError('Finite physical budget required')
        cfg['release_joint'] = args.release_joint
        if args.release_joint:
            if not args.gate_receipt: raise ValueError('G1/G2/G0 admission receipt required before actor release')
            receipt = json.loads(Path(args.gate_receipt).read_text())
            from frameworks.joint_healthy.identity import check_admission
            check_admission(receipt,cfg)
            cfg['gate_receipt_sha256'] = hashlib.sha256(Path(args.gate_receipt).read_bytes()).hexdigest()
        from frameworks.joint_healthy.controller import execute
        print(json.dumps(execute(cfg, auth), indent=2)); return 0
    if args.release_joint: raise ValueError('Evaluation cannot enable optimization')
    if not 10 < args.seconds < float('inf') or time.time() + args.seconds + 120 >= auth['deadline_epoch']:
        raise ValueError('Frozen evaluation needs a complete window and cleanup reserve')
    import numpy as np
    import tensorflow as tf
    from frameworks.joint_healthy.model import Model
    from frameworks.joint_healthy.flow import run_flow, dump, R
    from frameworks.joint_healthy.audit import audit_flow
    condition = cfg['conditions'][args.scenario]
    tf.set_random_seed(cfg['seed']); model = Model(cfg, training=False)
    stop = threading.Event()
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)) as sess:
        sess.run(tf.global_variables_initializer())
        if args.checkpoint: model.restore(sess, args.checkpoint)
        else:
            if cfg['profile'] not in ('g0', 'g1', 'g2', 'g3'):
                raise ValueError('A joint/fixed candidate needs its own trained checkpoint; baseline initialization is only a frozen G0/G1/G2 diagnostic')
            model.initialize_baseline(sess, cfg['baseline_checkpoint'])
        before = model.digest(sess)
        def action(state):
            value = sess.run(model.action, {model.s: state[None, :]})[0]
            if args.override_period_ms is not None: value[1] = period_action(args.override_period_ms)
            return value, dict(explored=False, checkpoint=args.checkpoint, diagnostic_override_ms=args.override_period_ms)
        result = run_flow(cfg, condition, action, stop, frozen_seconds=args.seconds)
        after = model.digest(sess)
        if before != after: raise ValueError('Frozen policy/BN changed')
    proof = audit_flow(result['folder'], args.seconds)
    summary = dict(result=result, execution=proof, policy_sha256=before,
                   counters=dict(critic_iterations=0, critic_optimizer_ops=0, actor_updates=0, target_updates=0, exploration_calls=0),
                   profile=cfg['profile'], clock_mode=cfg['clock_mode'], rate_mode=cfg['rate_mode'], decision_cost=cfg['decision_cost'],
                   same_weight_override_is_not_independent_fixed_training=args.override_period_ms is not None)
    dump(Path(result['folder']) / 'frozen-summary.json', summary); print(json.dumps(summary, indent=2)); return 0


if __name__ == '__main__': sys.exit(main())
