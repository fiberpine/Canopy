"""Real balanced replay, independent calibration holdout and explicit phase gate."""
import math
import random
import numpy as np
from .core import PhysicalCredit, PERIODS, CALIBRATION, FORWARD_ONLY, JOINT


class Replay:
    def __init__(self, cfg):
        self.cfg = cfg; self.size = int(cfg['memsize']); self.actors = int(cfg['num_actors'])
        self.batch_size = int(cfg.get('batch_size', 512)); self.rng = random.Random(cfg['seed'])
        self.credit = PhysicalCredit(self.actors, cfg['warmup'], cfg['minimum_enqueued_per_actor'],
                                     cfg.get('warmup_physical_seconds', 245.76), cfg['updates_per_base20ms'])
        self.arrays = [np.zeros((self.size, n), np.float64 if i == 5 else np.float32) for i, n in enumerate((82, 2, 1, 82, 1, 1, 1))]
        self.actor_ids = np.full(self.size, -1, np.int32)
        self.period_ids = np.full(self.size, -1, np.int32)
        self.birth = np.zeros(self.size, np.int64); self.pools = [set() for _ in range(self.actors)]
        self.coverage = np.zeros((self.actors, 5), np.int64)
        self.last = {}; self.sample_calls = 0; self.sampled = np.zeros(self.actors, np.int64)
        self.holdout = [[] for _ in range(5)]; self.holdout_seen = np.zeros(5, np.int64)
        self.short_returns = [[] for _ in range(5)]; self.short_seen = np.zeros(5, np.int64)
        self.pending_returns = {}

    def append(self, values, actor, episode, sequence):
        if len(values) != 7: raise ValueError('SMDP replay needs seven tensors')
        row = [np.asarray(x, np.float64 if i == 5 else np.float32).reshape(n) for i, (x, n) in enumerate(zip(values, (82, 2, 1, 82, 1, 1, 1)))]
        if not all(np.isfinite(x).all() for x in row): raise ValueError('Nonfinite transition')
        if np.max(np.abs(row[1])) > 1.00001 or row[4][0] not in (0., 1.): raise ValueError('Action/terminal range')
        dt, discount = float(row[5][0]), float(row[6][0])
        if dt <= 0 or not math.isclose(discount, .995 ** (dt / .02), rel_tol=2e-5, abs_tol=1e-8):
            raise ValueError('Real duration/discount mismatch')
        actor, episode, sequence = int(actor), int(episode), int(sequence)
        if not 0 <= actor < self.actors or episode < 0 or sequence < 1: raise ValueError('Transition identity')
        if actor in self.last:
            ep, seq, terminal = self.last[actor]
            if episode == ep:
                if sequence != seq + 1 or terminal: raise ValueError('Gap or transition after terminal')
            elif episode != ep + 1 or sequence != 1 or not terminal:
                raise ValueError('Cross-connection transition or missing episode terminal')
        elif sequence != 1 or episode != 0: raise ValueError('First actor sequence must start at episode0/seq1')
        self.last[actor] = (episode, sequence, bool(row[4][0]))
        planned = 20. * 8. ** ((float(row[1][1]) + 1.) / 2.)
        period = min(range(5), key=lambda i: abs(math.log(planned / PERIODS[i])))
        index = self.credit.inserted % self.size
        old = self.actor_ids[index]
        if old >= 0:
            self.coverage[old, self.period_ids[index]] -= 1
            self.pools[old].discard(index)
        self.credit.insert(actor, dt); birth = self.credit.inserted
        self.actor_ids[index], self.period_ids[index], self.birth[index] = actor, period, birth
        for array, value in zip(self.arrays, row): array[index] = value
        sequence_holdout = (sequence - 1) % 257 < 8
        if sequence_holdout:
            if (sequence - 1) % 257 == 0: self.pending_returns[actor] = []
            block = self.pending_returns[actor]; block.append(row)
            if len(block) == 8 or bool(row[4][0]):
                start_period = min(range(5), key=lambda i: abs(math.log((20.*8.**((float(block[0][1][1])+1.)/2.))/PERIODS[i])))
                self.short_seen[start_period] += 1; bucket = self.short_returns[start_period]
                cap = self.cfg.get('short_return_blocks_per_period', 64)
                if len(bucket) < cap: bucket.append(block)
                else:
                    candidate = self.rng.randrange(int(self.short_seen[start_period]))
                    if candidate < cap: bucket[candidate] = block
                del self.pending_returns[actor]
        if birth % 17 == 0 or sequence_holdout:
            self.holdout_seen[period] += 1; bucket = self.holdout[period]
            cap = self.cfg.get('holdout_rows_per_period', 256)
            if len(bucket) < cap: bucket.append(row)
            else:
                candidate = self.rng.randrange(int(self.holdout_seen[period]))
                if candidate < cap: bucket[candidate] = row
            # A held-out slot must never be reachable by the optimizer sampler.
            self.period_ids[index] = -1
            self.actor_ids[index] = -1
        else:
            self.coverage[actor, period] += 1; self.pools[actor].add(index)

    def sample(self):
        base, remainder = divmod(self.batch_size, self.actors)
        extra = {(self.sample_calls * remainder + j) % self.actors for j in range(remainder)}
        indices = []
        for actor, pool in enumerate(self.pools):
            quota = base + int(actor in extra)
            if len(pool) < quota: raise ValueError('Insufficient retained actor samples')
            selected = self.rng.sample(sorted(pool), quota); indices.extend(selected); self.sampled[actor] += quota
        self.sample_calls += 1
        index = np.asarray(indices, dtype=np.int64)
        assert np.all(self.birth[index] % 17 != 0)
        return [array[index].copy() for array in self.arrays]

    def holdout_batch(self, period):
        rows = self.holdout[period]
        if not rows: return None
        return [np.stack([row[i] for row in rows]) for i in range(7)]

    def short_return_batch(self, period):
        blocks = self.short_returns[period]
        if not blocks: return None
        first = [np.stack([b[0][i] for b in blocks]) for i in range(7)]
        last = [np.stack([b[-1][i] for b in blocks]) for i in range(7)]
        rewards = []; continuation = []
        for block in blocks:
            total = 0.; weight = 1.
            for row in block:
                total += weight * float(row[2][0]); weight *= float(row[6][0])*(1.-float(row[4][0]))
            rewards.append([total]); continuation.append([weight])
        return first, last, np.array(rewards), np.array(continuation)

    def snapshot(self):
        n = min(self.size, self.credit.inserted)
        return dict(zip(('s0', 'actions', 'rewards', 's1', 'terminal', 'duration', 'discount'), [x[:n] for x in self.arrays]),
                    actor_ids=self.actor_ids[:n], period_ids=self.period_ids[:n], birth=self.birth[:n])


class CalibrationGate:
    def __init__(self, cfg):
        self.cfg = cfg; self.phase = CALIBRATION; self.phase_started = 0; self.history = []

    def assess(self, iterations, coverage, holdout_reports, short_reports=None):
        minimum = self.cfg.get('calibration_updates', 5000) if self.phase == CALIBRATION else self.cfg.get('forward_calibration_updates', 1000)
        reasons = []
        if iterations - self.phase_started < minimum: reasons.append('minimum_critic_calibration')
        required = [PERIODS.index(float(self.cfg['fixed_mtp_ms']))] if self.cfg.get('fixed_mtp_ms') is not None else list(range(5))
        if np.any(np.asarray(coverage)[:, required] < self.cfg.get('calibration_rows_per_actor_period', 128)):
            reasons.append('retained_real_period_coverage')
        report = {x['period']: x for x in holdout_reports}
        reference_period = required[0] if len(required) == 1 else 0
        reference = report.get(reference_period, {}).get('normalized_td_rmse', float('inf'))
        for period in required:
            item = report.get(period, {})
            error = item.get('normalized_td_rmse', float('inf'))
            if item.get('rows', 0) < self.cfg.get('minimum_holdout_per_period', 16): reasons.append('holdout_coverage_%d' % period)
            if not math.isfinite(error) or error > min(self.cfg.get('maximum_normalized_td_rmse', .25),
                                                      max(.05, reference * self.cfg.get('relative_td_limit', 2.))):
                reasons.append('holdout_value_error_%d' % period)
        short = {x['period']: x for x in (short_reports or [])}
        short_reference = short.get(reference_period, {}).get('normalized_return_rmse', float('inf'))
        for period in required:
            item = short.get(period, {})
            error = item.get('normalized_return_rmse', float('inf'))
            if item.get('blocks', 0) < self.cfg.get('minimum_short_return_blocks', 4): reasons.append('short_return_coverage_%d' % period)
            if not math.isfinite(error) or error > min(self.cfg.get('maximum_normalized_short_return_rmse', .35), max(.05, short_reference*2.)):
                reasons.append('short_return_value_error_%d' % period)
        decision = dict(phase=self.phase, iterations=iterations, admitted=not reasons, reasons=reasons,
                        holdout=holdout_reports, short_returns=short_reports or [], coverage=np.asarray(coverage).tolist())
        self.history.append(decision)
        if not reasons and self.phase < JOINT:
            self.phase += 1; self.phase_started = iterations
        return decision
