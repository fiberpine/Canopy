"""NumPy reproduction of the v4 known-advantage critic task (no TensorFlow).

Reads the migrated seed3 critics directly from the TF checkpoint (tfckpt.py),
rebuilds model.py::_critic + continuous.critic_residual (critic_np.py) and
replays tools/check_joint_continuous_capability.py's critic task: the same two
real states, 256 log-uniform durations each, full-batch Adam, 10000 updates,
endpoint verdict with the unchanged margin/4 MAE gate. Residual layers use a
NumPy seed, not TF seed 3, so results are a statistical match, not bitwise.

  python3 tools/joint_numpy_reproduction/capability_matrix.py --residual-lr 1e-4 \
      --critic critic2 --sign -1 --level registered_168_169 --output cell.json

level=migrated_initial_q keeps the same known advantages but on the critics'
own initial 20ms level instead of the arbitrary 168/169 labels.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import tfckpt, critic_np as C


def cell(residual_lr, scope, sign, level, updates=10000, init_seed=0):
    ck = tfckpt.read(str(ROOT / 'models/healthy-baseline-seed3/training/step-6013196'))
    src = np.load(str(ROOT / 'evidence/joint-cost-repair-20261116/real-frozen-probes.npz'))['states'].astype(np.float64)
    idx = [int(np.argmin(src[:, 79])), int(np.argmax(src[:, 79]))]
    states = src[idx].copy(); states[:, 66] = 1.
    queries = np.array([20., 40., 80., 160.]); qs = np.repeat(states, 4, 0)
    qa = np.column_stack([np.zeros(8), C.encode(np.tile(queries, 2))])
    gate = C.COST * (C.GAMMA - C.GAMMA ** 2) / (1 - C.GAMMA) / 4
    critic = C.Critic(ck, scope, np.random.RandomState(init_seed))
    ss, a, labels, _ = C.task(states, sign); truth = C.truth(sign)
    if level == 'migrated_initial_q':
        q0, _, _ = critic.forward(ss, np.column_stack([a[:, 0], -np.ones(len(a))]))
        labels = q0 + (labels - np.repeat([168., 169.], 256)[:, None])
    elif level != 'registered_168_169': raise ValueError(level)
    late = []
    for i in range(updates + 1):
        if i >= updates // 2 and i % 250 == 0:
            q, _, _ = critic.forward(qs, qa); q = q.reshape(2, 4)
            late.append(float(np.mean(np.abs((q - q[:, :1])[:, 1:] - truth[1:]))))
        if i < updates: critic.step(ss, a, labels, 1e-3, residual_lr)
    q, _, _ = critic.forward(qs, qa); q = q.reshape(2, 4); adv = q - q[:, :1]
    mae = float(np.mean(np.abs(adv[:, 1:] - truth[1:])))
    active = np.abs(truth) >= gate * 4 * .999
    ranking = float(np.mean(np.sign(adv[:, active]) == np.sign(truth[active]))) if sign else None
    return dict(critic_period_residual_lr=residual_lr, critic=scope, sign=sign, label_level=level, updates=updates,
                advantage_mae=mae, ranking=ranking, pass_gate=bool(mae <= gate and (ranking is None or ranking >= .95)),
                predicted_mean_advantages=adv.mean(0).tolist(), known_advantages=truth.tolist(),
                late_window_min=min(late), late_window_max=max(late),
                late_window_fraction_within_gate=float(np.mean(np.array(late) <= gate)),
                implementation='numpy_reproduction_not_tf_graph')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--residual-lr', type=float, default=None); p.add_argument('--critic', default='critic')
    p.add_argument('--sign', type=float, required=True); p.add_argument('--level', default='registered_168_169')
    p.add_argument('--updates', type=int, default=10000); p.add_argument('--output', required=True)
    args = p.parse_args()
    result = cell(args.residual_lr, args.critic, args.sign, args.level, args.updates)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result))
