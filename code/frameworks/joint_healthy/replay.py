"""Real balanced replay, independent calibration holdout and explicit phase gate."""
import math
import random
import numpy as np
from .core import PhysicalCredit, PERIODS, CALIBRATION, FORWARD_ONLY, JOINT, bootstrap_terminal, heldout_position, period_bucket


class IndexPool:
    """Uniform sampling without sorting the entire actor pool on every update."""
    def __init__(self):
        self.values = []; self.positions = {}

    def __len__(self): return len(self.values)

    def __iter__(self): return iter(self.values)

    def add(self, index):
        if index not in self.positions:
            self.positions[index] = len(self.values); self.values.append(index)

    def discard(self, index):
        position = self.positions.pop(index, None)
        if position is None: return
        last = self.values.pop()
        if position < len(self.values):
            self.values[position] = last; self.positions[last] = position

    def sample(self, rng, quota): return rng.sample(self.values, quota)


class Replay:
    def __init__(self, cfg):
        self.cfg = cfg; self.size = int(cfg['memsize']); self.actors = int(cfg['num_actors'])
        self.batch_size = int(cfg.get('batch_size', 512)); self.rng = random.Random(cfg['seed'])
        self.credit = PhysicalCredit(self.actors, cfg['warmup'], cfg['minimum_enqueued_per_actor'],
                                     cfg.get('warmup_physical_seconds', 245.76), cfg['updates_per_base20ms'])
        self.arrays = [np.zeros((self.size, n), np.float64 if i == 5 else np.float32) for i, n in enumerate((82, 2, 1, 82, 1, 1, 1))]
        self.actor_ids = np.full(self.size, -1, np.int32)
        self.period_ids = np.full(self.size, -1, np.int32)
        self.birth = np.zeros(self.size, np.int64); self.pools = [IndexPool() for _ in range(self.actors)]
        self.next_in_episode = np.full(self.size, -1, np.int32)
        self.prev_in_episode = np.full(self.size, -1, np.int32)
        self.last_index = {}
        self.teacher_actions = np.zeros((self.size,1),np.float32) if float(cfg.get('teacher_anchor_weight',0))>0 else None
        self.coverage = np.zeros((self.actors, 5), np.int64)
        self.last = {}; self.sample_calls = 0; self.sampled = np.zeros(self.actors, np.int64)
        self.holdout = [[] for _ in range(5)]; self.holdout_seen = np.zeros(5, np.int64)
        self.short_returns = [[] for _ in range(5)]; self.short_seen = np.zeros(5, np.int64)
        self.pending_returns = {}

    def append(self, values, actor, episode, sequence, teacher_action=None):
        if len(values) != 7: raise ValueError('SMDP replay needs seven tensors')
        row = [np.asarray(x, np.float64 if i == 5 else np.float32).reshape(n) for i, (x, n) in enumerate(zip(values, (82, 2, 1, 82, 1, 1, 1)))]
        if not all(np.isfinite(x).all() for x in row): raise ValueError('Nonfinite transition')
        if self.teacher_actions is not None:
            if teacher_action is None or not math.isfinite(float(teacher_action)) or abs(float(teacher_action))>1.00001:
                raise ValueError('A finite same-state teacher action is required')
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
        period = period_bucket(self.cfg, planned)
        index = self.credit.inserted % self.size
        old = self.actor_ids[index]
        if old >= 0:
            self.coverage[old, self.period_ids[index]] -= 1
            self.pools[old].discard(index)
        prev_slot = int(self.prev_in_episode[index])
        if prev_slot >= 0: self.next_in_episode[prev_slot] = -1
        next_slot = int(self.next_in_episode[index])
        if next_slot >= 0: self.prev_in_episode[next_slot] = -1
        self.prev_in_episode[index] = -1; self.next_in_episode[index] = -1
        self.credit.insert(actor, dt); birth = self.credit.inserted
        self.actor_ids[index], self.period_ids[index], self.birth[index] = actor, period, birth
        for array, value in zip(self.arrays, row): array[index] = value
        if self.teacher_actions is not None:self.teacher_actions[index,0]=float(teacher_action)
        position = heldout_position(self.cfg, actor, episode, sequence)
        sequence_holdout = position < 8
        if sequence_holdout:
            if position == 0: self.pending_returns[actor] = []
            # A block that starts before this physical connection is never
            # continued across its reset. Its rows still enter one-step holdout.
            block = self.pending_returns.get(actor)
            if block is not None: block.append(row)
            if block is not None and (len(block) == 8 or bool(row[4][0])):
                start_period = period_bucket(self.cfg, 20.*8.**((float(block[0][1][1])+1.)/2.))
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
            # A held-out slot must never be reachable by the optimizer sampler or n-step rollout.
            self.period_ids[index] = -1
            self.actor_ids[index] = -1
            self.last_index.pop(actor, None)
        else:
            if actor in self.last_index:
                prev_actor_idx, prev_ep, prev_term = self.last_index[actor]
                if episode == prev_ep and not prev_term and prev_actor_idx != index:
                    self.next_in_episode[prev_actor_idx] = index
                    self.prev_in_episode[index] = prev_actor_idx
            self.last_index[actor] = (index, episode, bool(row[4][0]))
            self.coverage[actor, period] += 1; self.pools[actor].add(index)

    def sample(self, n_step=1):
        base, remainder = divmod(self.batch_size, self.actors)
        extra = {(self.sample_calls * remainder + j) % self.actors for j in range(remainder)}
        quotas = [base + int(actor in extra) for actor in range(self.actors)]
        for actor, (pool, quota) in enumerate(zip(self.pools, quotas)):
            if len(pool) < quota:
                raise ValueError('Insufficient retained actor samples: actor=%d retained=%d required=%d' % (actor, len(pool), quota))
        indices = []
        for actor, (pool, quota) in enumerate(zip(self.pools, quotas)):
            selected = pool.sample(self.rng, quota); indices.extend(selected); self.sampled[actor] += quota
        self.sample_calls += 1
        index = np.asarray(indices, dtype=np.int64)
        assert np.all(self.birth[index] % 17 != 0)
        batch=[array[index].copy() for array in self.arrays]
        if n_step > 1:
            for row_i, start in enumerate(index):
                total_r = float(batch[2][row_i, 0])
                total_d = float(batch[6][row_i, 0])
                cur = int(start)
                last_term = bool(bootstrap_terminal(self.cfg, batch[4][row_i])[0])
                for _ in range(n_step - 1):
                    if last_term: break
                    nxt = int(self.next_in_episode[cur])
                    if nxt < 0: break
                    assert self.actor_ids[nxt] >= 0 and self.birth[nxt] % 17 != 0
                    total_r += total_d * float(self.arrays[2][nxt, 0])
                    total_d *= float(self.arrays[6][nxt, 0])
                    cur = nxt
                    last_term = bool(bootstrap_terminal(self.cfg, self.arrays[4][cur])[0])
                batch[2][row_i, 0] = total_r
                batch[3][row_i] = self.arrays[3][cur]
                batch[4][row_i] = self.arrays[4][cur]
                batch[6][row_i, 0] = total_d
        if self.teacher_actions is not None:batch.append(self.teacher_actions[index].copy())
        return batch

    def sample_dual(self, n_step=8):
        """Return both exact 1-step and n-step SMDP columns for the same minibatch indices."""
        base, remainder = divmod(self.batch_size, self.actors)
        extra = {(self.sample_calls * remainder + j) % self.actors for j in range(remainder)}
        quotas = [base + int(actor in extra) for actor in range(self.actors)]
        for actor, (pool, quota) in enumerate(zip(self.pools, quotas)):
            if len(pool) < quota:
                raise ValueError('Insufficient retained actor samples: actor=%d retained=%d required=%d' % (actor, len(pool), quota))
        indices = []
        for actor, (pool, quota) in enumerate(zip(self.pools, quotas)):
            selected = pool.sample(self.rng, quota); indices.extend(selected); self.sampled[actor] += quota
        self.sample_calls += 1
        index = np.asarray(indices, dtype=np.int64)
        assert np.all(self.birth[index] % 17 != 0)
        one_step = [array[index].copy() for array in self.arrays]
        ret_n = one_step[2].copy()
        s_n = one_step[3].copy()
        term_n = one_step[4].copy()
        disc_n = one_step[6].copy()
        if n_step > 1:
            for row_i, start in enumerate(index):
                total_r = float(ret_n[row_i, 0])
                total_d = float(disc_n[row_i, 0])
                cur = int(start)
                last_term = bool(bootstrap_terminal(self.cfg, term_n[row_i])[0])
                for _ in range(n_step - 1):
                    if last_term: break
                    nxt = int(self.next_in_episode[cur])
                    if nxt < 0: break
                    assert self.actor_ids[nxt] >= 0 and self.birth[nxt] % 17 != 0
                    total_r += total_d * float(self.arrays[2][nxt, 0])
                    total_d *= float(self.arrays[6][nxt, 0])
                    cur = nxt
                    last_term = bool(bootstrap_terminal(self.cfg, self.arrays[4][cur])[0])
                ret_n[row_i, 0] = total_r
                s_n[row_i] = self.arrays[3][cur]
                term_n[row_i] = self.arrays[4][cur]
                disc_n[row_i, 0] = total_d
        return one_step, (ret_n, s_n, term_n, disc_n)

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
                total += weight * float(row[2][0]); weight *= float(row[6][0])*(1.-float(bootstrap_terminal(self.cfg,row[4])[0]))
            rewards.append([total]); continuation.append([weight])
        return first, last, np.array(rewards), np.array(continuation)

    def snapshot(self):
        n = min(self.size, self.credit.inserted)
        result=dict(zip(('s0', 'actions', 'rewards', 's1', 'terminal', 'duration', 'discount'), [x[:n] for x in self.arrays]),
                    actor_ids=self.actor_ids[:n], period_ids=self.period_ids[:n], birth=self.birth[:n])
        if self.teacher_actions is not None:result['teacher_actions']=self.teacher_actions[:n]
        return result


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
