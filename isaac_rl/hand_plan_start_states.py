"""Measured post-hit q/qd bank plus a small collision-clear random-start mix."""
from __future__ import annotations

import numpy as np
from amortized_opt.constants import N_DOF, G1_UPPER_BODY_JOINTS, ready_joint_pos
from isaac_rl.hand_plan_ranker import cartesian_tips


class PostHitStateBank:
    def __init__(self, lower, upper, clearance, *, per_combination=32, tip_geometry=None):
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.clearance = clearance
        self.per_combination = int(per_combination)
        self.tip_geometry = tip_geometry
        self.states, self.seen = {}, {}

    def valid(self, q, qd, *, random=False):
        q, qd = np.asarray(q), np.asarray(qd)
        okay = (np.isfinite(q).all(1) & np.isfinite(qd).all(1)
            & (q >= self.lower).all(1) & (q <= self.upper).all(1)
            & (np.abs(q[:, 1:3]) <= .08).all(1) & (np.abs(qd) <= 12.).all(1))
        ids = np.flatnonzero(okay)
        if len(ids):
            okay[ids] &= np.asarray(self.clearance(q[ids])) >= (.003 if random else -.005)
        return okay

    def add(self, states, rng):
        if not states:
            return dict(proposed=0, accepted=0, bank_size=sum(map(len, self.states.values())), combinations=len(self.states))
        q = np.asarray([s['q'] for s in states], np.float32)
        qd = np.asarray([s['qd'] for s in states], np.float32)
        okay = self.valid(q, qd)
        added = 0
        for state, valid in zip(states, okay):
            if not valid or state.get('kind') != 'post_hit':
                continue
            key = (int(state['source_hand']), int(state['source_drum']))
            if key == (0, 6):
                continue
            tips = np.asarray(state['tips'], np.float32)
            if tips.shape != (2, 3) or not np.isfinite(tips).all():
                continue
            bucket = self.states.setdefault(key, [])
            self.seen[key] = self.seen.get(key, 0)+1
            if len(bucket) < self.per_combination:
                bucket.append(state)
            else:
                slot = int(rng.integers(self.seen[key]))
                if slot < len(bucket):
                    bucket[slot] = state
            added += 1
        return dict(proposed=len(states), accepted=added, bank_size=sum(map(len, self.states.values())),
                    combinations=len(self.states), counts={f'{h}:{d}': len(v) for (h,d),v in self.states.items()})

    def sample(self, rng, count):
        if not self.states:
            raise ValueError('Bootstrap real successful post-hit states before training')
        keys = sorted(self.states)
        # Explicit quotas retain a small ready control and random pose share.
        ready_count = max(1, round(count*.05)) if count >= 4 else 0
        random_count = max(1, round(count*.15)) if count >= 4 else 0
        post_count = count-ready_count-random_count
        modes = ['post_hit']*post_count+['random']*random_count+['ready']*ready_count
        rng.shuffle(modes)
        rows, retries, fallbacks = [], 0, 0
        noise_std = np.asarray([.40 if n == 'waist_yaw_joint' else .025 if n.startswith('waist_') else .25
                                for n in G1_UPPER_BODY_JOINTS])
        source_codes = np.arange(post_count) % len(keys); rng.shuffle(source_codes)
        post_index = 0
        ready = np.asarray(ready_joint_pos(), np.float32)
        for mode in modes:
            if mode == 'post_hit':
                key = keys[int(source_codes[post_index])]; post_index += 1
                bucket = self.states[key]
                # Exact full measured state, including both arms and waist.
                rows.append(dict(bucket[int(rng.integers(len(bucket)))]))
                continue
            q, qd = ready.copy(), np.zeros(N_DOF, np.float32)
            kind = mode
            if mode == 'random':
                accepted = False
                for _ in range(128):
                    retries += 1
                    key = keys[int(rng.integers(len(keys)))]
                    bucket = self.states[key]
                    base = ready if rng.random() < .5 else np.asarray(bucket[int(rng.integers(len(bucket)))]['q'])
                    trial = np.clip(base+rng.normal(size=N_DOF)*noise_std,
                                    self.lower+.001, self.upper-.001).astype(np.float32)
                    trial[1:3] = trial[1:3].clip(-.08, .08)
                    velocity = rng.normal(0, .3, N_DOF).astype(np.float32)
                    velocity[:3] *= .5
                    if self.valid(trial[None], velocity[None], random=True)[0]:
                        q, qd, accepted = trial, velocity, True
                        break
                if not accepted:
                    kind = 'ready_fallback'; fallbacks += 1
            rows.append(dict(q=q.tolist(), qd=qd.tolist(), tips=cartesian_tips(q, self.tip_geometry).tolist(),
                             initial_drums=[-1, -1], kind=kind))
        poses = np.asarray([s['q'] for s in rows]); velocities = np.asarray([s['qd'] for s in rows])
        return rows, dict(post_hit_count=post_count, random_count=random_count-fallbacks,
            ready_count=ready_count, ready_fallback_count=fallbacks, random_proposals=retries,
            bank_combinations=len(keys), bank_size=sum(map(len, self.states.values())),
            waist_yaw_std_rad=float(poses[:, 0].std()),
            waist_pitch_std_rad=float(poses[:, 1].std()), waist_roll_std_rad=float(poses[:, 2].std()),
            arm_position_std_mean_rad=float(poses[:, 3:].std(0).mean()),
            velocity_rms_rad_s=float(np.sqrt(np.mean(velocities**2))))

    def state_dict(self):
        return dict(states=self.states, seen=self.seen, per_combination=self.per_combination)

    def load_state_dict(self, data):
        self.states, self.seen = data['states'], data['seen']
        self.per_combination = data['per_combination']
