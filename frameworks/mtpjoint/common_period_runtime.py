"""A0-verifiable boundary/replay/update/snapshot integration.

No new clock and no TCP reset is invented here. Callers provide the existing
native boundary, its actual interval reward and a same-connection successor.
Calibration and training connections have distinct IDs and data paths.
"""
import json
import pathlib
import numpy as np
from common_period import Replay, Credit, discount, checksum, REWARD, encoded_period, PERIODS


class Runtime:
    def __init__(self, model, reward_config, seed=3, active_periods=PERIODS):
        self.model = model
        self.reward_config = reward_config
        self.replay = Replay(checksum(reward_config), active_periods=active_periods)
        self.credit = Credit()
        self.rng = np.random.RandomState(seed)
        self.calibration = []
        self.known_snapshots = {}
        self.training_connections = set()
        self.calibration_connections = set()
        self.gradient_history = []

    def publish(self, window_only=False):
        snapshot = self.model.snapshot(window_only=True) if window_only else self.model.snapshot()
        if snapshot['reward_checksum'] != self.replay.reward_checksum:
            raise ValueError('Policy/replay reward contract mismatch')
        self.known_snapshots[snapshot['version']] = {k:v for k,v in snapshot.items() if k!='values'}
        return snapshot

    def boundary(self, actor, connection, next_connection, state, action, reward,
                 next_state, dt, period, version, kind='continuing', behavior='policy',
                 calibration=False, apply=1):
        if version not in self.known_snapshots:
            raise ValueError('Experience references an unpublished policy')
        if calibration:
            if connection in self.training_connections:
                raise ValueError('Calibration must use a separate connection')
            self.calibration_connections.add(connection)
        else:
            if connection in self.calibration_connections:
                raise ValueError('Calibration connection cannot enter replay')
            self.training_connections.add(connection)
        row=dict(actor=actor,connection=connection,next_connection=next_connection,s0=state,s1=next_state,
                 action=action,reward=reward,duration=dt,discount=discount(dt),period_ms=period,
                 snapshot=version,kind=kind,behavior=behavior,apply=apply,calibration=calibration,
                 reward_version=REWARD,reward_checksum=self.replay.reward_checksum)
        # Use exactly the same validation in both paths, with calibration kept
        # outside the real replay and therefore outside its exposure ledger.
        if calibration:
            isolated=Replay(self.replay.reward_checksum,capacity=1,active_periods=self.replay.active_periods)
            checked=dict(row,calibration=False)
            if isolated.append(checked):
                self.calibration.append(dict(isolated.rows[0],calibration=True))
                return True
            return False
        return self.replay.append(row)

    def update_once(self):
        actor = self.credit.next(self.replay)
        batch = self.replay.sample(self.rng,self.model.metadata['K'])
        noise = self.rng.normal(0,.1,(len(batch['s0']),1)).astype(np.float32)
        self.model.train_batch(batch,noise,actor)
        self.credit.record(actor)
        if self.credit.critic % 100 == 0:
            self.gradient_history.append(dict(critic_iterations=self.credit.critic,
                periods=(self.model.diagnostics(batch['s0']) if self.replay.active_periods == PERIODS
                         else self.model.diagnostics(batch['s0'], self.replay.active_periods))))
        return actor

    def evaluate_b0(self, updates, registered_cap):
        if self.replay.active_periods != PERIODS:
            raise ValueError('B0 requires a separately registered multi-period family')
        if updates != registered_cap or updates <= 0:
            raise ValueError('B0 must use its exact preregistered finite update budget')
        if self.credit.b0:
            raise ValueError('Do not automatically repeat B0 policy evaluation')
        with self.model.lock:
            for _ in range(updates):
                batch=self.replay.sample(self.rng,self.model.metadata['K'])
                self.model.session.run(self.model.b0_step,self.model.feed(batch))
                self.credit.b0 += 1
        return self.credit.b0

    def ledger(self):
        return dict(inserted=self.replay.inserted,real_training_seconds=self.replay.seconds,
                    inserted_by_actor=self.replay.by_actor.tolist(),sampled_by_actor=self.replay.sampled_by_actor.tolist(),
                    retained_actor_period=self.replay.retained_coverage.tolist(),
                    warmup_seconds=self.credit.warmup_seconds,issued=self.credit.limit(self.replay),
                    critic_iterations=self.credit.critic,critic_optimizer_ops=2*self.credit.critic,
                    actor_updates=self.credit.actor,N_updates=self.credit.n,target_updates=self.credit.target,
                    calibration_intervals=len(self.calibration),calibration_credit=0,
                    B0_updates=self.credit.b0,complete_tcp_resume=False)

    def save(self,prefix,*,compress_replay=True):
        if pathlib.Path(prefix+'.replay.npz').exists():raise ValueError('Never overwrite replay')
        rows=self.replay.rows
        fields=['s0','s1','action','reward','duration','discount','period_ms','snapshot','actor','kind',
                'connection','next_connection','behavior','bootstrap','apply','reward_version','reward_checksum']
        string_fields=('connection','next_connection','kind','behavior','reward_version','reward_checksum')
        arrays={key:np.asarray([str(row[key]) if key in string_fields else row[key] for row in rows]) for key in fields}
        write_replay = np.savez_compressed if compress_replay else np.savez
        write_replay(prefix+'.replay.npz',**arrays)
        return self.model.save(prefix,self.ledger(),dict(reward_version=REWARD,
            reward_checksum=self.replay.reward_checksum,raw_rewards=True,rows=len(rows),
            optimizer_rng=self.rng.get_state()[1].tolist(),rng_position=self.rng.get_state()[2]))


def require_a0(preregistration,results):
    """The old run.py train command is never an entrypoint for this scheme."""
    if preregistration.get('status')!='FROZEN':raise ValueError('Registration not frozen')
    missing=[r['name'] for r in results['checks'] if r.get('required',True) and r['status']!='PASS']
    if missing:raise ValueError('Required A0 checks not passed: '+', '.join(missing))
