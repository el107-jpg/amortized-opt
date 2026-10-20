"""Four-beat, bounded candidate search and direct-return Q distillation."""
from __future__ import annotations

from dataclasses import replace
from itertools import groupby, product

import numpy as np
import torch

from amortized_opt.constants import N_DOF, PELVIS_IN_ENV, STICK_TIP_SIGMA
from amortized_opt.fk import stick_contact_offset
from isaac_rl.arm_assignment import AssignmentQueue


def hit_groups(clip):
    hits = sorted(clip.hits, key=lambda h: (h.time, h.drum))
    groups = [list(rows) for _, rows in groupby(hits, key=lambda h: h.time)]
    if not groups or any(len(g) > 2 for g in groups):
        raise ValueError("search requires nonempty rhythms with at most two simultaneous hits")
    return groups


def egregious_crossovers(groups, actions, kit, yaw, *, margin=.2):
    """Reject only severely crossed simultaneous targets in the torso frame."""
    for group, action in zip(groups, actions):
        if len(group) == 2:
            left = kit.positions[group[action].drum]-PELVIS_IN_ENV
            right = kit.positions[group[1-action].drum]-PELVIS_IN_ENV
            dy = -np.sin(yaw)*(left[0]-right[0])+np.cos(yaw)*(left[1]-right[1])
            if dy < -margin:
                return True
    return False


def candidate_sequences(clip, kit, policy_cfg, rng, q0, *, limit=16, crossover_margin=.2):
    """Exhaustive for <=4 decisions; otherwise at most 16 full-phrase candidates.

    Never truncate a dense rhythm to four decisions. Single hits on either arm
    are permitted, including left-floor and right-hihat assignments.
    """
    groups = hit_groups(clip)
    n = len(groups)
    queue = AssignmentQueue(clip, kit, policy_cfg)
    teacher = []
    while queue.next_group < len(queue.groups):
        try:
            action = queue.teacher_action()
        except ValueError as error:
            if str(error) != "graph teacher cannot assign this group":
                raise
            # The graph is only a candidate seed. Its hand restrictions (or
            # duplicate drum names) must not reject an unrestricted search
            # problem. Keep both physical strikes and seed a deterministic
            # assignment; complete rollouts determine its value.
            action = queue.next_group % 2
        teacher.append(action); queue.commit(action)
    allowed = [[a for a in (0, 1) if not egregious_crossovers([g], [a], kit, float(q0[0]), margin=crossover_margin)]
               for g in groups]
    teacher = [a if a in opts else opts[0] for a, opts in zip(teacher, allowed)]
    free = [g for g, opts in enumerate(allowed) if len(opts) == 2]
    if 2**len(free) <= limit:
        pool = list(product(*allowed))
    else:
        pool = [tuple(teacher), tuple(1-a for a in teacher)]
        for i in rng.permutation(free)[:max(0, limit//4)]:
            row = list(teacher); row[i] = 1-row[i]; pool.append(tuple(row))
        pool.extend(tuple(opts[int(rng.integers(len(opts)))] for opts in allowed) for _ in range(128))
    accepted, seen, rejected = [], set(), 0
    for row in pool:
        if row in seen:
            continue
        seen.add(row)
        if egregious_crossovers(groups, row, kit, float(q0[0]), margin=crossover_margin):
            rejected += 1
            continue
        accepted.append(row)
        if len(accepted) == limit:
            break
    if not accepted:
        raise ValueError("crossover screen rejected every candidate")
    return groups, np.asarray(accepted, dtype=np.int64), rejected


def pack_candidates(clips, kit, policy_cfg, poses, rng, *, limit=16, rebound_samples=2,
                    max_groups=64, device="cpu", efficiency_range=(.2, .8)):
    """Use identical drum-specific bounce draws for competing assignments."""
    problems = [candidate_sequences(c, kit, policy_cfg, rng, q, limit=limit)
                for c, q in zip(clips, poses)]
    G = max(len(g) for g, _, _ in problems)
    if G > max_groups:
        raise ValueError(f"phrase has {G} hit groups, capacity {max_groups}; refusing to drop hits")
    q0, drums, speeds, times, valid, eta, owners, first, actions = [], [], [], [], [], [], [], [], []
    for owner, (groups, candidates, _) in enumerate(problems):
        draws = rng.uniform(*efficiency_range, size=(rebound_samples, G, 2))
        for row in candidates:
            target = np.full((G, 2), -1, np.int64); velocity = np.zeros((G, 2), np.float32)
            event_times = np.full(G, groups[-1][0].time, np.float32)
            mask = np.arange(G) < len(groups)
            for g, group in enumerate(groups):
                event_times[g] = group[0].time
                for j, hit in enumerate(group):
                    hand = int((row[g]+j) % 2)
                    target[g, hand] = hit.drum
                    velocity[g, hand] = 1+2*hit.velocity
            for bounce in range(rebound_samples):
                efficiencies = np.ones((G, 2), np.float32)
                for g, group in enumerate(groups):
                    for j in range(len(group)):
                        efficiencies[g, int((row[g]+j) % 2)] = draws[bounce, g, j]
                q0.append(poses[owner]); drums.append(target); speeds.append(velocity)
                times.append(event_times); valid.append(mask); eta.append(efficiencies)
                owners.append(owner); first.append(row[0]); actions.append(row)
    t = lambda x, dtype: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    return dict(q0=t(q0, torch.float32), v0=torch.zeros(len(q0), N_DOF, device=device),
        drums=t(drums, torch.long), times=t(times, torch.float32), speeds=t(speeds, torch.float32),
        valid=t(valid, torch.bool), efficiency=t(eta, torch.float32),
        owner=np.array(owners), first_action=np.array(first), sequences=actions,
        problems=problems, rebound_samples=rebound_samples)


def direct_q_targets(scores, packed, count, requested=None):
    """Best sampled continuation per first action, averaged across rebound draws.

    The target contains no target-network value or Bellman bootstrap.
    """
    values = scores.detach().cpu().numpy()
    target = np.full((count, 2), -1000., np.float32)
    best_rows = []
    for owner in range(count):
        indexes = np.flatnonzero(packed["owner"] == owner)
        grouped = {}
        for index in indexes:
            key = tuple(packed["sequences"][index])
            grouped.setdefault(key, []).append(int(index))
        candidates = [(np.mean(values[ids]), ids) for ids in grouped.values()]
        for score, ids in candidates:
            action = int(packed["first_action"][ids[0]])
            target[owner, action] = max(target[owner, action], score)
        # Retain all rebound trials of the best complete assignment sequence.
        best = max(candidates, key=lambda item: item[0])[1]
        best_rows.extend(best)
        if requested is not None:
            n = float(requested[indexes[0]])
            finite = target[owner] > -1000
            target[owner, finite] /= max(n, 1)
            target[owner, ~finite] = min(target[owner, finite])-10
    return target, np.array(best_rows)


def assignment_observations(clips, kit, cfg, q, fk, device):
    Q = torch.as_tensor(q, device=device, dtype=torch.float32)
    tips = []
    for hand in ("left", "right"):
        R, p = fk.frame(Q, hand+"_wrist")
        offset = Q.new_tensor(stick_contact_offset(hand, STICK_TIP_SIGMA))
        tips.append(p+(R @ offset[:, None]).squeeze(-1))
    tip = torch.stack(tips, 1).detach().cpu().numpy()
    observations = []
    for b, clip in enumerate(clips):
        # All events in this finite phrase must be visible even at slow tempi.
        pc = replace(cfg, lookahead_s=max(cfg.lookahead_s, clip.hits[-1].time+.01))
        queue = AssignmentQueue(clip, kit, pc)
        observations.append(queue.observation(0, q[b], np.zeros(N_DOF), tip[b]))
    return torch.as_tensor(np.stack(observations), device=device)
