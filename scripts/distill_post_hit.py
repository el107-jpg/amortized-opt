#!/usr/bin/env python
"""Distill KIND_POST_HIT (Phase B) rows from rhythm_idle / existing post_hit parents.

LEGACY. Training now harvests on-policy post-hit inits from the MPC roller
(plant state after a real hit, next goal from clip refill). This script's
synthetic tip-displace + IK path is kept for offline experiments only.

    python scripts/distill_post_hit.py \\
        --pool-a /path/to/a_f.npz --ckpt runs/.../best.pt \\
        --out /path/to/post_hit.npz --n 2000 --device cuda:0

Parents are KIND_RHYTHM_IDLE (+ KIND_POST_HIT when present in --pool-a or
--pool-b).  Merge the result into training via ``train_phase2.py --post-hit-cache``.
Unlock when rolling eval is at least 30/81 accepted (``stage_a_pass``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A
from amortized_opt.constants import G1_UPPER_BODY_JOINTS, PELVIS_IN_ENV
from amortized_opt.fk import ChainFK
from amortized_opt.ik import arm_joint_indices, correct_tip, solve_tip_velocity, tip_position
from amortized_opt.limits import isaac_velocity_limits, urdf_position_limits
from amortized_opt.phase2_data import (
    KIND_POST_HIT,
    KIND_RHYTHM_IDLE,
    Phase2Pool,
    Phase2ScenarioConfig,
    STRIKE_FIELDS,
    _empty_strike,
    _strike_row,
)
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig, mass_matrix
from amortized_opt.rhythm import HANDS, DrumKit, default_rcc_dir, load_rcc_pool
from amortized_opt.rhythm_chain_data import (
    SCHEDULE_FIELDS,
    SCHEDULE_SCHEMA,
    TIP_MODE_AROUND,
    TIP_MODE_CYMBAL_BELOW,
    TIP_MODE_HIT,
    refill_lookahead_from_clip,
    sample_post_hit_tip,
)
from amortized_opt.rnea import BatchedRNEA
from amortized_opt.sdf import SdfPenalty, q0_collision_clear
from amortized_opt.tokenizer import ContextInputs
from amortized_opt.strike_seed import accelerating_path_numpy
from amortized_opt.waist_aim import aim_yaw, waist_reference_block
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice, waist_slice
from amortized_opt.warmstart_data import CONTEXT_FIELDS, bucket_for


_LONG_FIELDS = frozenset({"goal_drum", "future_drum"})
_BOOL_FIELDS = frozenset(
    {"drum_valid", "goal_valid", "future_valid", "idle_valid", "stick_valid", "aim_valid"}
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-a", required=True, help="Phase A (+ optional B) npz")
    p.add_argument("--pool-b", default=None, help="extra post_hit parents to mix in")
    p.add_argument("--ckpt", required=True, help="phase-2 (or usable phase-1) checkpoint")
    p.add_argument("--out", required=True, help="output post_hit npz")
    p.add_argument("--n", type=int, default=2000, help="accepted post_hit rows to emit")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen", type=int, default=0, help="generation tag written on rows")
    p.add_argument("--hit-mode-frac", type=float, default=0.70)
    p.add_argument("--around-mode-frac", type=float, default=0.20)
    p.add_argument("--ik-residual-max", type=float, default=0.04,
                   help="drop row if tip residual after correct_tip exceeds this (m)")
    p.add_argument("--clip-dir", default=None)
    p.add_argument("--max-attempts", type=int, default=None,
                   help="cap parent draws (default 8 x --n)")
    return p.parse_args(argv)


def _load_pool(path: Path, device) -> Phase2Pool:
    z = np.load(path, allow_pickle=True)
    buckets: dict[int, dict[str, np.ndarray]] = {}
    for k in z.files:
        if k.startswith("_"):
            continue
        b, name = k.split("/", 1)
        buckets.setdefault(int(b), {})[name] = z[k]
    # Infer a minimal scenario cfg from stored shapes.
    any_b = next(iter(buckets))
    H = int(any_b)
    nf = int(buckets[any_b]["future_valid"].shape[-1]) if "future_valid" in buckets[any_b] else 1
    cfg = Phase2ScenarioConfig(
        horizon=max(buckets), buckets=tuple(sorted(buckets)),
        num_future=nf, eval_like_frac=0.0, cold_start_frac=0.0, pair_frac=0.0,
        hit_time_jitter_s=0.05,
    )
    return Phase2Pool(buckets, cfg=cfg, device=device)


def _parent_indices(pool: Phase2Pool, rng: np.random.Generator, n: int):
    """Sample (bucket, row) from A+B parent kinds."""
    want = {KIND_RHYTHM_IDLE, KIND_POST_HIT}
    choices = []
    for b in pool.buckets:
        for k in want:
            idx = pool.indices_of_kind(b, k)
            if idx.numel():
                choices.append((b, idx.cpu().numpy()))
    if not choices:
        raise SystemExit("no KIND_RHYTHM_IDLE / KIND_POST_HIT parents in pool")
    # Uniform over rows across chosen kinds.
    sizes = np.array([len(ix) for _, ix in choices], float)
    p = sizes / sizes.sum()
    out = []
    for _ in range(n):
        ci = int(rng.choice(len(choices), p=p))
        b, ix = choices[ci]
        out.append((int(b), int(ix[int(rng.integers(len(ix)))])))
    return out


def _pick_mode(rng: np.random.Generator, hit_f: float, around_f: float, drum_name: str) -> str:
    u = float(rng.uniform())
    if u < hit_f:
        return TIP_MODE_HIT
    if u < hit_f + around_f:
        return TIP_MODE_AROUND
    if "cymbal" in drum_name:
        return TIP_MODE_CYMBAL_BELOW
    # Rescale leftover mass onto hit/around when not a cymbal.
    return TIP_MODE_HIT if float(rng.uniform()) < hit_f / max(hit_f + around_f, 1e-9) else TIP_MODE_AROUND


def _cone_velocity(n_out: np.ndarray, rng: np.random.Generator, *,
                   speed=(0.4, 1.2), half_angle_deg=60.0) -> np.ndarray:
    """Soft ~120° cone about outward normal."""
    n = np.asarray(n_out, float).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(ref, n); t1 /= max(float(np.linalg.norm(t1)), 1e-12)
    t2 = np.cross(n, t1)
    theta = abs(float(rng.normal(0.0, math.radians(half_angle_deg) / 2.0)))
    theta = min(theta, math.radians(half_angle_deg))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    d = np.cos(theta) * n + np.sin(theta) * (np.cos(phi) * t1 + np.sin(phi) * t2)
    d /= max(float(np.linalg.norm(d)), 1e-12)
    spd = float(rng.uniform(*speed))
    return spd * d


@torch.no_grad()
def _roll_to_hit(model, sim, rnea, q0, qd0, init, hit, active, context_fields,
                 *, device, dtype, lead_f=None):
    """Plan-follow (+ optional model refine) until first active hit fires.

    Returns ``(q, qd, init_shifted, fired_hand, ok)``.
    """
    from amortized_opt.limits import ETA_VELOCITY
    from amortized_opt.lead import lead_waypoints
    from amortized_opt.tokenizer import build_waypoint_masks

    names = G1_UPPER_BODY_JOINTS
    q_lo, q_hi = urdf_position_limits(names)
    q_lo_t = torch.as_tensor(q_lo, device=device, dtype=dtype)
    q_hi_t = torch.as_tensor(q_hi, device=device, dtype=dtype)
    step_max = ETA_VELOCITY * torch.as_tensor(
        isaac_velocity_limits(names), device=device, dtype=dtype
    ) * float(A.CONTROL_DT)

    q = q0.clone()
    qd = qd0.clone()
    plan = init.clone()
    h = hit.clone()
    act = active.clone()
    prev = q.clone()
    max_steps = int(max(1, int(h.clamp_min(0).max().item()) + 2))
    ctx = ContextInputs(**{f: context_fields[f] for f in CONTEXT_FIELDS if f in context_fields})

    for _ in range(max_steps):
        if not bool(act.any()):
            return q, qd, plan, -1, False
        # Fire when any active hand's hit has counted down to 0.
        fired = act & (h <= 0)
        if bool(fired.any()):
            # Prefer a random active fired hand.
            hands = fired[0].nonzero(as_tuple=True)[0]
            hi = int(hands[int(torch.randint(0, hands.numel(), (1,)).item())].item())
            return q, qd, plan, hi, True

        t = h.clamp_min(0).to(dtype) * float(A.CONTROL_DT)
        masks = build_waypoint_masks(
            t[:, 0], t[:, 1], max_horizon=plan.shape[1], dt=float(A.CONTROL_DT),
            active_left=act[:, 0], active_right=act[:, 1], joint_names=names,
        )
        state = torch.cat([q, qd], dim=-1)
        core = model.module if hasattr(model, "module") else model
        enc = core.encode_context(ctx)
        out = core(plan * masks.elem_mask, state, enc, masks)
        traj = out.traj.to(dtype)
        cmd = traj[:, 0] if lead_f is None else lead_waypoints(traj, lead_f)[:, 0]
        cmd = prev + torch.clamp(cmd - prev, -step_max, step_max)
        cmd = torch.clamp(cmd, q_lo_t, q_hi_t)
        Minv = torch.linalg.inv(mass_matrix(rnea, q))
        q, qd, _ = sim.step(prev, cmd, q, qd, Minv=Minv)
        prev = cmd
        plan = torch.cat([traj[:, 1:], traj[:, -1:]], dim=1)
        h = torch.where(h >= 0, h - 1, h)
        # Cheap context clock update for goal_tth if present.
        if "goal_tth" in context_fields:
            context_fields["goal_tth"] = torch.clamp(
                context_fields["goal_tth"] - float(A.CONTROL_DT), min=0.0
            )
            ctx = ContextInputs(
                **{f: context_fields[f] for f in CONTEXT_FIELDS if f in context_fields}
            )
    return q, qd, plan, -1, False


def _promote_row(
    row: dict[str, np.ndarray],
    *,
    hand_i: int,
    kit: DrumKit,
    clips,
    library: GoalPoseLibrary,
    q0: np.ndarray,
    qd0: np.ndarray,
    leftover_init: np.ndarray,
    tip_aim: np.ndarray,
    n_out: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    fk: ChainFK,
    sdf: SdfPenalty,
    ik_residual_max: float,
    gen: int,
    q_bounds,
) -> dict[str, np.ndarray] | None:
    """Build one KIND_POST_HIT host-side row, or None on drop."""
    hand = HANDS[hand_i]
    names = G1_UPPER_BODY_JOINTS
    # Tip displace + IK.
    drum = int(row["drum"][hand_i])
    if drum < 0:
        return None
    tip = sample_post_hit_tip(
        kit, drum, tip_aim, n_out, mode=mode, rng=rng,
        hit_radius=float(row["hit_radius"][hand_i]),
    )
    qt = torch.as_tensor(q0[None], dtype=torch.float64)
    aim_t = torch.as_tensor(tip[None], dtype=torch.float64)
    qc = correct_tip(
        fk, qt, aim_t, hand, steps=3, q_bounds=q_bounds, joint_names=names,
    )
    tip_err = tip_position(fk, qc, hand).detach().cpu().numpy()[0] - tip
    if float(np.linalg.norm(tip_err)) > float(ik_residual_max):
        return None
    q_new = qc.detach().cpu().numpy()[0].astype(np.float64)
    # Stick+wrist SDF on the displaced start, same screen as live Pool B /
    # Pool A q0.  Exempt the pad just struck so leftover contact is not a
    # reject; the row's own randomized kit if the parent carried one.
    q_t = torch.as_tensor(q_new[None], dtype=torch.float32)
    heads = (
        torch.as_tensor(np.asarray(row["drum_pos"])[None], dtype=torch.float32)
        if "drum_pos" in row else None
    )
    radii = (
        torch.as_tensor(np.asarray(row["drum_radius"])[None], dtype=torch.float32)
        if "drum_radius" in row else None
    )
    exempt = torch.full((1, 2), -1, dtype=torch.long)
    exempt[0, hand_i] = drum
    if not bool(q0_collision_clear(
        sdf, q_t, drum_head_centers=heads, drum_radii=radii, exempt_drum=exempt,
    )[0]):
        return None

    # Tip velocity in outward cone → arm qd; other DoFs keep plant.
    v_tip = _cone_velocity(n_out, rng)
    cols = arm_joint_indices(hand, names)
    qd_arm = solve_tip_velocity(
        fk,
        torch.as_tensor(q_new[None], dtype=torch.float64),
        hand,
        torch.as_tensor(v_tip[None], dtype=torch.float64),
        weighting="plain",
        joint_subset=cols,
        damping=1e-4,
    ).detach().cpu().numpy()[0]
    qd_new = qd0.copy()
    qd_new[cols] = qd_arm[cols]
    qd_new = qd_new + rng.normal(0.0, 0.02, size=qd_new.shape)

    # ---- promote schedule: goal ← future[0] / sched[0]; refill from clip ----
    if not bool(row["future_valid"][hand_i, 0]):
        return None
    new_drum = int(row["future_drum"][hand_i, 0])
    new_pos = np.asarray(row["future_pos"][hand_i, 0], float)
    new_vel = np.asarray(row["future_vel"][hand_i, 0], float)
    new_tth = float(row["future_gap"][hand_i, 0])
    if new_tth <= 1e-4 or new_drum < 0:
        return None
    # Timing jitter on promoted goal.
    jit = float(rng.uniform(-0.05, 0.05))
    new_tth = max(0.08, new_tth + jit)

    # Provenance / schedule refill.
    clip_i = int(row.get("clip_i", -1)) if np.ndim(row.get("clip_i", -1)) == 0 else int(row["clip_i"])
    hand_hit = np.asarray(row["hand_hit_time"], float).copy() if "hand_hit_time" in row else np.full(2, -1.0)
    # New goal's clip time ≈ prior goal time + gap (when prior was valid).
    if hand_hit[hand_i] >= 0 and bool(row["goal_valid"][hand_i]):
        hand_hit[hand_i] = float(hand_hit[hand_i]) + float(row["future_gap"][hand_i, 0])
    sched = {f: np.asarray(row[f]).copy() for f in SCHEDULE_FIELDS if f in row}
    # Shift: schedule[0] ← old schedule[1] if present; else refill.
    fut = {
        "future_drum": np.full((2, 1), -1, np.int64),
        "future_pos": np.zeros((2, 1, 3), np.float32),
        "future_vel": np.zeros((2, 1, 3), np.float32),
        "future_gap": np.zeros((2, 1), np.float32),
        "future_valid": np.zeros((2, 1), bool),
    }
    # Copy other hand's future untouched.
    for hi in range(2):
        if hi == hand_i:
            continue
        for k in ("future_drum", "future_pos", "future_vel", "future_gap", "future_valid"):
            if k in row:
                fut[k][hi] = row[k][hi, :1]

    filled = False
    if "sched_valid" in sched and sched["sched_valid"].shape[1] > 1 and bool(sched["sched_valid"][hand_i, 1]):
        fut["future_drum"][hand_i, 0] = sched["sched_drum"][hand_i, 1]
        fut["future_pos"][hand_i, 0] = sched["sched_pos"][hand_i, 1]
        fut["future_vel"][hand_i, 0] = sched["sched_vel"][hand_i, 1]
        fut["future_gap"][hand_i, 0] = max(0.05, float(sched["sched_dt"][hand_i, 1]) - float(row["future_gap"][hand_i, 0]))
        fut["future_valid"][hand_i, 0] = True
        filled = True
        # Clear consumed schedule[1]; keep schedule[0] as mirror of future.
        sched["sched_valid"][hand_i, 1] = False
    if (not filled) and clips is not None and 0 <= clip_i < len(clips):
        fields, ok = refill_lookahead_from_clip(
            kit, clips[clip_i], hand, new_goal_hit_time=float(hand_hit[hand_i]),
            n_future=1, n_schedule_extra=1,
        )
        if not ok:
            return None
        fut["future_drum"][hand_i, 0] = fields["future_drum"][0]
        fut["future_pos"][hand_i, 0] = fields["future_pos"][0]
        fut["future_vel"][hand_i, 0] = fields["future_vel"][0]
        fut["future_gap"][hand_i, 0] = fields["future_gap"][0]
        fut["future_valid"][hand_i, 0] = True
        if "sched_valid" in fields:
            for f in SCHEDULE_FIELDS:
                if f in fields and f in sched:
                    sched[f][hand_i, 1] = fields[f][0] if f != "sched_valid" else bool(fields[f][0])
        filled = True
    if not filled or not bool(fut["future_valid"][hand_i, 0]):
        return None
    # Mirror future into schedule[0].
    if "sched_valid" in sched:
        sched["sched_drum"][hand_i, 0] = fut["future_drum"][hand_i, 0]
        sched["sched_pos"][hand_i, 0] = fut["future_pos"][hand_i, 0]
        sched["sched_vel"][hand_i, 0] = fut["future_vel"][hand_i, 0]
        sched["sched_dt"][hand_i, 0] = fut["future_gap"][hand_i, 0]
        sched["sched_valid"][hand_i, 0] = True

    # Library terminal for fired arm; leftover for other arm; waist rebuild.
    if not library.has(hand, kit.names[new_drum]):
        return None
    yaw = float(q_new[names.index("waist_yaw_joint")])
    term_arm = library.select(hand, kit.names[new_drum], yaw).arm_q
    H = leftover_init.shape[0]
    dt = float(A.CONTROL_DT)
    hit_wp = int(max(2, min(H, round(new_tth / dt))))
    # Fired-arm seed retains the post-contact velocity.
    sl = arm_slice(hand, names)
    init = leftover_init.copy()
    init[:, sl] = accelerating_path_numpy(
        q_new[sl], qd_new[sl], term_arm, hit_wp, H, dt,
    )[1:]
    # Other arm: keep leftover columns.
    other = HANDS[1 - hand_i]
    osl = arm_slice(other, names)
    init[:, osl] = leftover_init[:, osl]

    # Waist rebuild from new aim.
    pelvis = np.asarray(PELVIS_IN_ENV, float)
    targets = {hand: new_pos - pelvis}
    tth = {hand: new_tth}
    futures = {
        hand: [(np.asarray(fut["future_pos"][hand_i, 0], float) - pelvis,
                float(fut["future_gap"][hand_i, 0]))]
        if fut["future_valid"][hand_i, 0] else []
    }
    # Keep other hand target if still active.
    other_i = 1 - hand_i
    if bool(row["active"][other_i]) and int(row["drum"][other_i]) >= 0:
        targets[other] = np.asarray(row["p_star"][other_i], float)  # already pelvis
        # p_star in pool is pelvis frame; aim_yaw wants pelvis.
        tth[other] = float(row["goal_tth"][other_i]) if "goal_tth" in row else hit_wp * dt
    ay = float(aim_yaw(targets, goal_tth=tth, futures=futures, yaw_current=yaw))
    aval = bool(targets)
    ws = waist_slice(names)
    waist_names = [names[c] for c in ws]
    jy = waist_names.index("waist_yaw_joint")
    sp = np.array([float(q_new[c]) for c in ws], float)
    sp[jy] = yaw
    ref = waist_reference_block(
        waist_names, yaw_start=yaw, yaw_target=float(ay) if aval else yaw,
        n_waypoints=H + 1, setpoint=sp, dt=dt, settle_wp=3,
        yaw_rate_start=float(qd_new[names.index("waist_yaw_joint")]),
        max_move_s=0.2,
    )
    init[:, ws] = ref[1:]

    # Active / hit / drum / strike / context.
    active = np.asarray(row["active"], bool).copy()
    active[hand_i] = True
    hit = np.asarray(row["hit"], np.int64).copy()
    hit[hand_i] = hit_wp
    # Other hand: shrink leftover horizon by the ticks we rolled (approx keep).
    drum_ids = np.asarray(row["drum"], np.int64).copy()
    drum_ids[hand_i] = new_drum
    start_drum = np.asarray(row["start_drum"], np.int64).copy()
    start_drum[hand_i] = drum

    strike_rows = []
    for hi, hname in enumerate(HANDS):
        if not active[hi] or int(drum_ids[hi]) < 0:
            strike_rows.append(_empty_strike())
        elif hi == hand_i:
            strike_rows.append(_strike_row(kit, new_drum, new_pos, new_vel))
        else:
            # Reuse parent strike geometry for the other hand.
            strike_rows.append({f: np.asarray(row[f][hi], float) for f in STRIKE_FIELDS})

    # Context: patch goal/future for fired hand; refresh aim / yaw_now.
    ctx = {f: np.asarray(row[f]).copy() for f in CONTEXT_FIELDS if f in row}
    ctx["goal_drum"][hand_i] = new_drum
    ctx["goal_pos"][hand_i] = new_pos
    ctx["goal_vel"][hand_i] = new_vel
    ctx["goal_tth"][hand_i] = new_tth
    ctx["goal_valid"][hand_i] = True
    for k, v in fut.items():
        ctx[k] = v.astype(ctx[k].dtype) if k in ctx else v
    ctx["yaw_now"] = np.float32(yaw)
    ctx["aim_yaw"] = np.float32(ay)
    ctx["aim_valid"] = bool(aval)

    out = {
        "q0": q_new.astype(np.float32),
        "qd0": qd_new.astype(np.float32),
        "init": init.astype(np.float32),
        "hit": hit,
        "active": active,
        "drum": drum_ids,
        "start_drum": start_drum,
        "kind": np.int64(KIND_POST_HIT),
        "clip_i": np.int64(clip_i),
        "hand_hit_time": hand_hit.astype(np.float32),
    }
    for f in STRIKE_FIELDS:
        out[f] = np.stack([np.asarray(r[f], float) for r in strike_rows]).astype(np.float32)
    for f in CONTEXT_FIELDS:
        if f not in ctx:
            continue
        v = ctx[f]
        if f in _LONG_FIELDS:
            out[f] = np.asarray(v, np.int64)
        elif f in _BOOL_FIELDS:
            out[f] = np.asarray(v, bool)
        else:
            out[f] = np.asarray(v, np.float32)
    for f in SCHEDULE_FIELDS:
        if f in sched:
            out[f] = sched[f]
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    pool = _load_pool(Path(args.pool_a), device)
    if args.pool_b:
        from amortized_opt.phase2_data import _merge
        pb = _load_pool(Path(args.pool_b), "cpu")
        host_a = {b: {k: v.detach().cpu().numpy() for k, v in d.items()}
                  for b, d in pool.data.items()}
        host_b = {b: {k: v.detach().cpu().numpy() for k, v in d.items()}
                  for b, d in pb.data.items()}
        raw = _merge([
            {"buckets": host_a, "stats": {}},
            {"buckets": host_b, "stats": {}},
        ])
        pool = Phase2Pool(raw["buckets"], cfg=pool.cfg, device=device)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(model_cfg, device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    kit = DrumKit.from_layout()
    clip_dir = args.clip_dir or default_rcc_dir("train")
    clips = load_rcc_pool(clip_dir, kit)
    library = GoalPoseLibrary()
    fk = ChainFK(device="cpu", dtype=torch.float64)
    sdf = SdfPenalty(device="cpu", dtype=torch.float32)
    q_bounds = urdf_position_limits(G1_UPPER_BODY_JOINTS)
    sim = BatchedPDSim(PDSimConfig(), device=device, dtype=torch.float32)
    rnea = BatchedRNEA(device=device, dtype=torch.float32)

    out_rec: dict[int, dict[str, list]] = {}
    stats = {"attempted": 0, "accepted": 0, "reject_roll": 0, "reject_ik": 0,
             "reject_promote": 0}
    max_attempts = int(args.max_attempts or 8 * args.n)
    t0 = time.perf_counter()
    print(f"[distill] parents={len(pool)} kinds={pool.kind_counts()} "
          f"target={args.n}", flush=True)

    while stats["accepted"] < int(args.n) and stats["attempted"] < max_attempts:
        parents = _parent_indices(pool, rng, min(args.batch, args.n - stats["accepted"]))
        for b, ri in parents:
            stats["attempted"] += 1
            d = pool.data[b]
            # Host row snapshot.
            row = {k: v[ri].detach().cpu().numpy() for k, v in d.items()}
            if not bool(row["active"].any()):
                stats["reject_roll"] += 1
                continue
            # Device batch of 1 for the roll.
            q0 = torch.as_tensor(row["q0"][None], device=device, dtype=torch.float32)
            qd0 = torch.as_tensor(row["qd0"][None], device=device, dtype=torch.float32)
            init = torch.as_tensor(row["init"][None], device=device, dtype=torch.float32)
            hit = torch.as_tensor(row["hit"][None], device=device, dtype=torch.long)
            active = torch.as_tensor(row["active"][None], device=device)
            ctx_fields = {
                f: torch.as_tensor(row[f][None], device=device)
                for f in CONTEXT_FIELDS if f in row
            }
            q, qd, plan, hi, ok = _roll_to_hit(
                model, sim, rnea, q0, qd0, init, hit, active, ctx_fields,
                device=device, dtype=torch.float32,
            )
            if not ok:
                stats["reject_roll"] += 1
                continue
            tip_aim = np.asarray(row["p_star"][hi], float) + np.asarray(PELVIS_IN_ENV, float)
            n_out = np.asarray(row["n_out"][hi], float)
            mode = _pick_mode(
                rng, args.hit_mode_frac, args.around_mode_frac, kit.names[int(row["drum"][hi])],
            )
            built = _promote_row(
                row, hand_i=hi, kit=kit, clips=clips, library=library,
                q0=q.cpu().numpy()[0], qd0=qd.cpu().numpy()[0],
                leftover_init=plan.cpu().numpy()[0],
                tip_aim=tip_aim, n_out=n_out, mode=mode, rng=rng,
                fk=fk, sdf=sdf, ik_residual_max=args.ik_residual_max,
                gen=int(args.gen), q_bounds=q_bounds,
            )
            if built is None:
                stats["reject_promote"] += 1
                continue
            # Re-bucket by new horizon.
            act_m = np.asarray(built["active"], bool)
            k_max = int(np.asarray(built["hit"])[act_m].max()) if act_m.any() else 2
            bb = bucket_for(k_max, tuple(pool.cfg.buckets))
            # Pad / trim init to bucket length.
            init_b = built["init"]
            if init_b.shape[0] < bb:
                pad = np.repeat(init_b[-1:], bb - init_b.shape[0], axis=0)
                built["init"] = np.concatenate([init_b, pad], axis=0)
            elif init_b.shape[0] > bb:
                built["init"] = init_b[:bb]
            rec = out_rec.setdefault(bb, {})
            for k, v in built.items():
                rec.setdefault(k, []).append(v)
            stats["accepted"] += 1
            if stats["accepted"] % 50 == 0:
                print(f"[distill] accepted {stats['accepted']}/{args.n} "
                      f"(attempts {stats['attempted']})", flush=True)

    if stats["accepted"] == 0:
        raise SystemExit(f"distill produced 0 rows; stats={stats}")

    packed = {
        b: {k: np.stack(v) for k, v in rec.items()}
        for b, rec in out_rec.items() if rec
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{b}/{k}": v for b, arrays in packed.items() for k, v in arrays.items()}
    from amortized_opt.warmstart_data import CONTEXT_FIELDS as _CF
    np.savez(
        out,
        _stats=np.array(json.dumps(stats)),
        _schema=np.array(json.dumps(list(_CF))),
        _schedule_schema=np.array(SCHEDULE_SCHEMA),
        **flat,
    )
    print(f"[distill] wrote {stats['accepted']} rows -> {out} in "
          f"{time.perf_counter() - t0:.1f}s; stats={stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
