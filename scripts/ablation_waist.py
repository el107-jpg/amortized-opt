#!/usr/bin/env python
"""Section 9b -- the waist-stabilisation ablation, five arms, one table.

The waist is three joints (yaw / pitch / roll) shared by both arms.  It is the
cheapest place in the whole objective to buy tip speed and azimuth, which is
exactly why it needs pricing: an unpriced torso winds up, leans, and hands the
PD servo a base motion the arms were never planned against.  Three mechanisms
were proposed for stopping that, and this script measures them against a shared
control:

1. **waist-aim IL** (``CostConfig.w_waist_aim``) -- a soft position penalty on
   the whole waist against ``waist_aim``'s own reference: yaw follows the damped
   azimuth law, pitch/roll follow zero.  It says *where* the torso should be.
   Arms at 5x and 10x the shipped ``w_waist_aim = 500``.
2. **the direct posture term** (``w_waist_upright`` + ``waist_pr_box`` +
   a heavy ``w_waist_yaw_vel``) -- a quadratic well pinning pitch/roll to the
   ready pose, a tightened joint box, and a price on yaw RATE.  It says *how
   far* the torso may lean and *how fast* it may turn, without saying where.
3. **structural removal** (``ModelConfig.frozen_joints``) -- pitch and roll are
   taken out of the action space entirely and pinned at the ready pose; the
   policy controls yaw alone.  Not a price: there is nothing left to trade.

Every arm continues from ``runs/phase1_10m/best.pt @ 18500`` with the same pool
cache, batch, horizon and buckets, so the comparison is paired.

Three stages, separately runnable because they cost wildly different amounts:

``--stage digest``   parse ``metrics.jsonl``; no GPU, works while training runs.
``--stage eval``     open-loop acceptance grid AND waist posture, both read off
                     the SAME batched plans so the two axes cannot drift apart.
``--stage closed``   closed-loop ``MPCGenerator`` episodes (the expensive one).
``--stage all``      all three.

    python scripts/ablation_waist.py --stage digest
    python scripts/ablation_waist.py --stage all --device cuda:2

Writes ``ablation_results.md`` plus a JSON sidecar with every number in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# The arms.  ``mechanism`` is what the row means, not what the flags say --
# the flags are recorded verbatim from the run's own log so the table cannot
# drift from what was actually trained.
# --------------------------------------------------------------------------- #
ARMS: tuple[tuple[str, str, str], ...] = (
    ("base", "abl_waist_base", "control -- yaw-rate price only"),
    ("aim5x", "abl_waist_aim5x", "1. waist-aim IL, 5x shipped (w=2500)"),
    ("aim10x", "abl_waist_aim10x", "1. waist-aim IL, 10x shipped (w=5000)"),
    ("direct", "abl_waist_direct", "2. posture well + tight box + yaw-rate"),
    ("freeze", "abl_waist_freeze", "3. pitch/roll out of the action space"),
)

WAIST = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
SHORT = {"waist_yaw_joint": "yaw", "waist_pitch_joint": "pitch",
         "waist_roll_joint": "roll"}


# ============================================================== stage 1 ======
def digest_run(run_dir: Path, *, tail_frac: float = 0.25) -> dict:
    """Training-time summary of one arm, straight out of ``metrics.jsonl``.

    Two things come out of here that the checkpoint evals cannot give:

    * the **acceptance trajectory** -- every rolling-grid eval, so a run that
      peaked early and decayed is distinguishable from one still climbing;
    * the **waist diagnostics under the training distribution** --
      ``d_waist_pr`` / ``d_waist_yaw`` / ``d_waist_yawd`` are per-batch maxima
      logged every 50 steps, which is a far larger sample of the waist's
      behaviour than any grid.  Summarised over the last ``tail_frac`` of the
      run, so a ramp that only bites late is not diluted by the early steps.
    """
    mpath = run_dir / "metrics.jsonl"
    if not mpath.exists():
        return {"present": False}
    train, evals, start = [], [], None
    with open(mpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line while the run is still writing
            ev = r.get("event")
            if ev == "train":
                train.append(r)
            elif ev == "eval":
                evals.append(r)
            elif ev == "start":
                start = r
    if not train:
        return {"present": False}

    last_step = int(train[-1].get("step", 0))
    max_steps = int(((start or {}).get("cfg") or {}).get("max_steps", 0)) or None
    cut = last_step * (1.0 - float(tail_frac))
    tail = [r for r in train if float(r.get("step", 0)) >= cut] or train[-1:]

    out: dict = {
        "present": True,
        "last_step": last_step,
        "max_steps": max_steps,
        "done": bool(max_steps and last_step >= max_steps - 50),
        "n_evals": len(evals),
        "tail_from_step": int(cut),
    }

    # --- the waist itself ---------------------------------------------------
    for key, tag in (("d_waist_pr", "pr"), ("d_waist_yaw", "yaw"),
                     ("d_waist_yawd", "yawd"), ("d_waist_prd", "prd")):
        v = np.array([float(r[key]) for r in tail if key in r], float)
        if v.size:
            out[f"{tag}_p50"] = float(np.percentile(v, 50))
            out[f"{tag}_p90"] = float(np.percentile(v, 90))
            out[f"{tag}_max"] = float(v.max())

    # --- did it still learn to drum ----------------------------------------
    if evals:
        acc = [(int(e["step"]), float(e["acceptance"])) for e in evals]
        best = max(acc, key=lambda t: t[1])
        out["acc_curve"] = acc
        out["best_acceptance"] = best[1]
        out["best_step"] = best[0]
        out["final_acceptance"] = acc[-1][1]
        for k in ("ok_hit", "ok_torque", "ok_collision", "ok_velocity",
                  "ok_joint_limits", "elbow_down", "contact_pos_err_mm_p50",
                  "speed_err_p50"):
            if k in evals[-1]:
                out[f"final_{k}"] = float(evals[-1][k])
    # --- what the objective actually charged --------------------------------
    for key in ("c_waist_aim", "c_waist_yaw_vel", "c_terminal", "c_collision",
                "c_torque", "cost"):
        v = np.array([float(r[key]) for r in tail if key in r], float)
        if v.size:
            out[f"{key}_p50"] = float(np.percentile(v, 50))
    return out


#: The knobs this ablation moves.  Read back off the run's own ``start`` event
#: rather than off a command line, so the table reports what the trainer
#: actually constructed -- a CLI flag that failed to reach ``CostConfig`` (the
#: exact defect ``test_waist_ablation.py`` exists to catch) would otherwise be
#: reported as if it had worked.
_WAIST_KNOBS = ("w_waist_aim", "w_waist_yaw_vel", "w_waist_upright", "waist_pr_box")


def read_config(run_dir: Path) -> dict:
    """Seed, waist ramp, waist weights and frozen columns, from ``metrics.jsonl``."""
    mpath = run_dir / "metrics.jsonl"
    if not mpath.exists():
        return {}
    with open(mpath) as f:
        try:
            r = json.loads(f.readline())
        except json.JSONDecodeError:
            return {}
    if r.get("event") != "start":
        return {}
    cfg = r.get("cfg") or {}
    cost = cfg.get("cost") or {}
    out = {"seed": cfg.get("seed"), "max_steps": cfg.get("max_steps"),
           "batch_size": cfg.get("batch_size"),
           "waist_ramp": (cfg.get("schedule") or {}).get("waist_ramp"),
           "waist_start": (cfg.get("schedule") or {}).get("waist_start")}
    out.update({k: cost.get(k) for k in _WAIST_KNOBS})
    # The freeze is a MODEL config, not a cost weight, so it is not in ``cost``.
    # The trainer announces it; picking it up here means the digest stage knows
    # about it without having to load a 129 MB checkpoint.
    log = run_dir / "train.log"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines()[:20]:
            if "frozen joints" in line:
                out["frozen_joints"] = line.split(":", 1)[1].strip()
                break
    return out


def flag_str(conf: dict) -> str:
    """The non-default waist knobs of one arm, as a compact cell."""
    bits = [f"{k}={conf[k]:g}" for k in _WAIST_KNOBS
            if conf.get(k) not in (None, 0.0)]
    return ", ".join(bits) if bits else "--"


# ============================================================== stage 2 ======
def load_model(ckpt: Path, dev):
    """Checkpoint -> model + the scenario config its tokenizer was built for.

    ``model_cfg`` carries ``frozen_joints``, so the freeze arm reconstructs with
    its pitch/roll columns still pinned -- evaluating it as a free-waist model
    would silently undo the very thing being measured.

    Architecture flags absent from the stored config are recovered from the
    state dict by :func:`~amortized_opt.model.model_config_from_checkpoint` (the
    ``_ARCH_FROM_WEIGHTS`` map this file used to carry, now the package's own),
    and the load is STRICT, so any remaining drift between the checkpoint and
    the installed package is a crash rather than a randomly-initialised layer
    scoring in the table.
    """
    import torch

    import amortized_opt as A

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    cfg = {"buckets": buckets, "num_drums": mcfg.num_drums,
           "num_future": mcfg.num_future,
           "frozen_joints": tuple(mcfg.frozen_joints)}
    return model, cfg, int(ck.get("step", -1))


def waist_stats(series: list[np.ndarray]) -> dict:
    """Excursion and travel per waist joint over a list of ``(T, 17)`` paths.

    Both statistics are needed and they say different things: a p90 of 0.4 rad
    with negligible travel is a static lean, the same p90 with large travel is a
    sway, and only the second one is a tracking problem for the servo.
    """
    import amortized_opt as A

    col = {n: A.G1_UPPER_BODY_JOINTS.index(n) for n in WAIST}
    out: dict = {"n_traj": len(series),
                 "n_samples": int(sum(len(s) for s in series))}
    if not series:
        return out
    for name, c in col.items():
        tag = SHORT[name]
        allq = np.concatenate([np.abs(s[:, c]) for s in series])
        trav = np.array([np.abs(np.diff(s[:, c])).sum() for s in series if len(s) > 1])
        out[f"{tag}_abs_p50"] = float(np.percentile(allq, 50))
        out[f"{tag}_abs_p90"] = float(np.percentile(allq, 90))
        out[f"{tag}_abs_max"] = float(allq.max())
        out[f"{tag}_travel_p50"] = float(np.percentile(trav, 50)) if trav.size else float("nan")
    pr = np.concatenate([
        np.hypot(s[:, col["waist_pitch_joint"]], s[:, col["waist_roll_joint"]])
        for s in series
    ])
    out["lean_norm_p50"] = float(np.percentile(pr, 50))
    out["lean_norm_p90"] = float(np.percentile(pr, 90))
    out["frac_lean_over_0.15"] = float((pr > 0.15).mean())
    return out


def open_loop(model, cfg, kit, library, scorer, dev, amp, *, grid, seed):
    """Acceptance AND posture off ONE set of plans.

    Scoring and the waist statistics read the same ``plans`` dict, so a row can
    never claim a posture that belongs to a different rollout than its
    acceptance -- which is the whole failure mode of measuring the two axes in
    separate scripts.
    """
    import torch

    import amortized_opt as A

    with torch.no_grad():
        cases = A.generate_grid(seed=seed, grid=grid, scorer=scorer)
        plans, wall = A.batched_model_plans(
            model, cases, device=str(dev), kit=kit, library=library,
            buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
            num_future=cfg["num_future"], iterations=1, init="quintic",
            amp_dtype=amp,
        )
    rows, _ = A.run_eval(
        A.precomputed_planner(plans, wall / max(len(cases), 1)),
        cases, scorer=scorer,
    )
    agg = A.aggregate(rows)
    n = max(agg["n"], 1)
    # Only the VALID window: the plan array is padded out to the bucket, and the
    # padding is an unpriced hold that would flatten every posture percentile.
    series = [np.asarray(plans[int(c.case_id)], float)[: int(c.N) + 1] for c in cases]
    out = {"n": agg["n"], "grid": grid, "seed": seed,
           "acceptance": agg["counts"]["accepted"]["count"] / n,
           "accepted": agg["counts"]["accepted"]["count"]}
    for flag in ("ok_hit", "ok_collision", "ok_torque", "ok_velocity",
                 "ok_joint_limits", "elev_ok"):
        if flag in agg["counts"]:
            out[flag] = agg["counts"][flag]["count"] / n
    for m in ("contact_pos_err_mm", "contact_speed_err_mps", "tau_excess_max_nm"):
        if m in agg["metrics"]:
            out[f"{m}_p50"] = agg["metrics"][m]["p50"]
    out["waist"] = waist_stats(series)
    return out


# ============================================================== stage 3 ======
def closed_loop(model, cfg, kit, scorer, dev, amp, *, clips, probe_seeds, steps,
                cmd_lead):
    """Full ``MPCGenerator`` episodes -- the posture the servo actually sees.

    Open loop scores a plan; this scores what the plant did with it.  The two
    can disagree badly, and for a waist term that is exactly the point: a soft
    price the planner is willing to pay on every tick compounds over an episode
    in a way that no single plan shows.

    Posture and strike accuracy are both read off the SAME episodes -- the waist
    statistics from ``res.q_hist`` and the timing from
    :func:`~amortized_opt.probe.hit_latency` -- rather than by running the
    episodes twice under two scripts.  A stabilisation mechanism is only worth
    anything if it holds the torso still *and* the sticks still arrive, and that
    trade is unreadable if the two halves come from different rollouts.
    """
    import amortized_opt as A
    from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator
    from amortized_opt.probe import hit_latency, mpc_model_planner, summarize_latency

    gen0 = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=cfg["buckets"][-1]))
    sampler = A.StartSampler(kit=kit, library=gen0.library, fk=gen0.fk,
                             rnea=gen0.rnea, sdf=gen0.sdf)
    series, eps, all_lat = [], [], []
    for ps in probe_seeds:
        starts = sampler.sample(len(clips), seed=int(ps))
        for ep, clip in enumerate(clips):
            gen = MPCGenerator(
                kit=kit,
                config=MPCGenConfig(horizon=cfg["buckets"][-1],
                                    cmd_lead_s=cmd_lead, sticky_goal_point=True),
            )
            plan, _st = mpc_model_planner(
                model, num_drums=cfg["num_drums"], num_future=cfg["num_future"],
                device=dev, amp_dtype=amp, iterations=1, record=None,
            )
            res = gen.run_episode(clip, starts.q[ep].cpu().numpy(),
                                  starts.qd[ep].cpu().numpy(),
                                  seed=1000 + 100 * int(ps) + ep, planner=plan,
                                  max_steps=int(steps))
            q = np.asarray(res.q_hist, float)
            series.append(q)
            lat = hit_latency(scorer, kit, res)
            all_lat.extend(lat)
            eps.append({"clip": clip.name, "probe_seed": int(ps),
                        "ticks": int(res.cmd_hist.shape[0]),
                        "scheduled": len(lat),
                        "diverged": bool(res.diverged)})
            print(f"      {clip.name} s{ps}: ticks {eps[-1]['ticks']}"
                  f" strikes {len(lat)}"
                  f"{'  DIVERGED' if eps[-1]['diverged'] else ''}", flush=True)
    out = waist_stats(series)
    out["episodes"] = eps
    out["n_diverged"] = sum(1 for e in eps if e["diverged"])
    out["latency"] = summarize_latency(all_lat)
    return out


# ============================================================== stage 4 ======
def diagnose_lean(model, cfg, kit, dev, amp, *, clip, seed, steps, cmd_lead):
    """WHERE the closed-loop waist excursion enters, and whether it RATCHETS.

    Two different questions, and a cost term can only answer the first.

    **Where.**  At every tick there are three waist states for the same instant:
    ``plan_wp1`` (what the planner asked for one control period ahead -- the
    only thing ``FusedCost`` ever prices), ``cmd`` (what the episode actually
    commanded, i.e. ``plan_wp1`` after the reference lead, the slew limiter and
    the box clamp) and ``q`` (what the plant reached).  ``cmd - plan_wp1`` is
    the lead, ``q - cmd`` is servo tracking error, and if both are small the
    excursion is the PLAN's own -- receding-horizon drift, each tick planning a
    small lean from a state that already has one.  Only that last case is
    fixable with a weight.

    **Whether it ratchets.**  A drift and a jitter have the same excursion
    percentile and want opposite fixes, so they are separated here by a least
    -squares fit of each waist joint against time:

    * ``slope`` (rad/s) -- the systematic component, the ratchet itself.  A
      torso creeping steadily one way shows up here and nowhere else, because
      it is perfectly consistent tick to tick: the temporal-consistency term
      prices the DIFFERENCE between successive plans and a slow march pays it
      almost nothing.
    * ``resid_rms`` (rad) -- what is left after removing the trend, i.e. the
      jitter that ``w_tc`` and ``w_waist_yaw_vel`` actually target.
    * ``net`` (rad) -- signed end-minus-start displacement, the plain statement
      of how far the torso ended up from where it began.

    A large ``slope`` with a small ``resid_rms`` is a ratchet and needs an
    ABSOLUTE anchor; the reverse is jitter and needs rate damping.
    """
    import amortized_opt as A
    from amortized_opt.constants import CONTROL_DT
    from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator
    from amortized_opt.probe import mpc_model_planner

    gen = MPCGenerator(
        kit=kit,
        config=MPCGenConfig(horizon=cfg["buckets"][-1], cmd_lead_s=cmd_lead,
                            sticky_goal_point=True),
    )
    gen0 = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=cfg["buckets"][-1]))
    sampler = A.StartSampler(kit=kit, library=gen0.library, fk=gen0.fk,
                             rnea=gen0.rnea, sdf=gen0.sdf)
    starts = sampler.sample(1, seed=int(seed))
    record: list = []
    plan, _st = mpc_model_planner(
        model, num_drums=cfg["num_drums"], num_future=cfg["num_future"],
        device=dev, amp_dtype=amp, iterations=1, record=record,
    )
    res = gen.run_episode(clip, starts.q[0].cpu().numpy(),
                          starts.qd[0].cpu().numpy(), seed=1000, planner=plan,
                          max_steps=int(steps))
    q = np.asarray(res.q_hist, float)
    cmd = np.asarray(res.cmd_hist, float)
    T = cmd.shape[0]
    n_dof = q.shape[1]
    wp1 = np.full((T, n_dof), np.nan)
    for t, r in enumerate(record[:T]):
        tr = np.asarray(r["traj"], float)
        if tr.ndim == 2 and tr.shape[0] >= 1:
            wp1[t] = tr[0]
    ok = ~np.isnan(wp1[:, 0])

    col = {n: A.G1_UPPER_BODY_JOINTS.index(n) for n in WAIST}
    out: dict = {"clip": clip.name, "ticks": int(T), "n_plans": int(ok.sum()),
                 "diverged": bool(res.diverged)}
    for name, c in col.items():
        tag = SHORT[name]
        a, b, m = wp1[ok, c], cmd[ok, c], q[1:T + 1][ok, c]
        out[f"{tag}_plan_abs_p50"] = float(np.percentile(np.abs(a), 50))
        out[f"{tag}_cmd_abs_p50"] = float(np.percentile(np.abs(b), 50))
        out[f"{tag}_meas_abs_p50"] = float(np.percentile(np.abs(m), 50))
        out[f"{tag}_lead_gap_p50"] = float(np.percentile(np.abs(b - a), 50))
        out[f"{tag}_track_gap_p50"] = float(np.percentile(np.abs(m - b), 50))
        # --- the ratchet test, on the MEASURED plant over the whole episode --
        series = q[:, c]
        t_s = np.arange(series.size, dtype=float) * float(CONTROL_DT)
        if series.size >= 3:
            slope, intercept = np.polyfit(t_s, series, 1)
            resid = series - (slope * t_s + intercept)
            out[f"{tag}_slope_rad_s"] = float(slope)
            out[f"{tag}_resid_rms"] = float(np.sqrt(np.mean(resid ** 2)))
            out[f"{tag}_net_rad"] = float(series[-1] - series[0])
            out[f"{tag}_range_rad"] = float(series.max() - series.min())
    return out


# ============================================================== reporting ====
def _f(v, prec=3, dash="--"):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(x) else f"{x:.{prec}f}"


def render(report: dict, notes: str = "") -> str:
    """The markdown table set.  Ordered so the ablated axis is read first.

    ``notes`` is prose written by hand and inlined verbatim after the header --
    the verdict, which is an interpretation of specific numbers and cannot be
    generated.  It is passed in (``--notes``) rather than appended to the output
    file afterwards so that re-rendering the tables never silently drops it.
    """
    L: list[str] = []
    A_ = L.append
    A_("# Waist-stabilisation ablation (section 9b)")
    A_("")
    A_(f"Generated by `scripts/ablation_waist.py` -- {report['generated']}.")
    A_("")
    A_("Five arms, all continuing from `runs/phase1_10m/best.pt @ 18500` with the")
    A_("same pool cache, batch (256), horizon (50) and buckets (25/50), so the")
    A_("comparison is paired.  The question is whether a waist-stabilisation")
    A_("mechanism can hold the torso still **without** costing the strike.")
    A_("")
    A_("| arm | mechanism | steps | flags |")
    A_("|---|---|---|---|")
    for label, run, mech in ARMS:
        d = report["arms"].get(label, {}).get("digest", {})
        if not d.get("present"):
            A_(f"| `{label}` | {mech} | *missing* | |")
            continue
        conf = report["arms"][label].get("config", {})
        frozen = (report["arms"][label].get("frozen_joints")
                  or conf.get("frozen_joints"))
        extra = "" if not frozen else " + frozen pitch/roll"
        A_(f"| `{label}` | {mech} | {d['last_step']}/{d.get('max_steps') or '?'}"
           f"{'' if d.get('done') else ' *(running)*'} | "
           f"`{flag_str(conf)}`{extra} |")
    A_("")
    seeds = {report["arms"][l].get("config", {}).get("seed")
             for l, _r, _m in ARMS if l in report["arms"]}
    ramps = {report["arms"][l].get("config", {}).get("waist_ramp")
             for l, _r, _m in ARMS if l in report["arms"]}
    seeds.discard(None)
    ramps.discard(None)
    if len(seeds) == 1 and len(ramps) == 1:
        r = ramps.pop()
        A_(f"All arms ran at seed {seeds.pop()} with `waist_ramp = {r}` -- the waist")
        A_(f"terms reach full weight by step {r}, i.e. they are on for essentially")
        A_("the whole run rather than fine-tuned in at the end.")
    else:
        A_(f"**Arms are NOT paired**: seeds {sorted(seeds)}, waist_ramp {sorted(ramps)}.")
    A_("")
    if notes:
        A_(notes.rstrip())
        A_("")

    # --- the axis being ablated -------------------------------------------
    A_("## Waist behaviour under the training distribution")
    A_("")
    A_("Per-batch maxima logged every 50 steps, summarised over the last quarter")
    A_("of each run.  `pr` is the pitch/roll excursion (rad), `yaw` the yaw")
    A_("excursion (rad), `yawd` the yaw RATE (rad/s) -- the one the torso uses to")
    A_("buy tip speed.  **Lower is the goal on all three.**")
    A_("")
    A_("| arm | pr p50 | pr p90 | pr max | yaw p50 | yaw p90 | yawd p50 | yawd p90 |")
    A_("|---|---|---|---|---|---|---|---|")
    for label, _run, _m in ARMS:
        d = report["arms"].get(label, {}).get("digest", {})
        if not d.get("present"):
            continue
        A_(f"| `{label}` | {_f(d.get('pr_p50'))} | {_f(d.get('pr_p90'))} | "
           f"{_f(d.get('pr_max'))} | {_f(d.get('yaw_p50'))} | {_f(d.get('yaw_p90'))} | "
           f"{_f(d.get('yawd_p50'))} | {_f(d.get('yawd_p90'))} |")
    A_("")

    # --- the cost of it ----------------------------------------------------
    A_("## Did it still learn to drum")
    A_("")
    A_("Rolling acceptance grid (81 cases) during training.")
    A_("")
    A_("| arm | best acc | @step | final acc | ok_hit | ok_torque | ok_coll | elbow |")
    A_("|---|---|---|---|---|---|---|---|")
    for label, _run, _m in ARMS:
        d = report["arms"].get(label, {}).get("digest", {})
        if not d.get("present") or "best_acceptance" not in d:
            continue
        A_(f"| `{label}` | **{_f(d['best_acceptance'])}** | {d['best_step']} | "
           f"{_f(d.get('final_acceptance'))} | {_f(d.get('final_ok_hit'))} | "
           f"{_f(d.get('final_ok_torque'))} | {_f(d.get('final_ok_collision'))} | "
           f"{_f(d.get('final_elbow_down'))} |")
    A_("")

    # --- open loop ---------------------------------------------------------
    ol = {k: v.get("open_loop") for k, v in report["arms"].items() if v.get("open_loop")}
    if ol:
        A_("## Open loop -- acceptance and posture off the same plans")
        A_("")
        for gridkey in sorted({g for v in ol.values() for g in v}):
            A_(f"### `{gridkey}`")
            A_("")
            A_("| arm | acc | ok_hit | ok_coll | ok_tau | pitch p90 | roll p90 | "
               "yaw p90 | lean p90 | yaw travel |")
            A_("|---|---|---|---|---|---|---|---|---|---|")
            for label, _run, _m in ARMS:
                b = ol.get(label, {}).get(gridkey)
                if not b:
                    continue
                w = b["waist"]
                A_(f"| `{label}` | **{_f(b['acceptance'])}** | {_f(b.get('ok_hit'))} | "
                   f"{_f(b.get('ok_collision'))} | {_f(b.get('ok_torque'))} | "
                   f"{_f(w['pitch_abs_p90'])} | {_f(w['roll_abs_p90'])} | "
                   f"{_f(w['yaw_abs_p90'])} | {_f(w['lean_norm_p90'])} | "
                   f"{_f(w['yaw_travel_p50'], 2)} |")
            A_("")

    # --- closed loop -------------------------------------------------------
    cl = {k: v.get("closed_loop") for k, v in report["arms"].items() if v.get("closed_loop")}
    if cl:
        A_("## Closed loop -- what the plant actually did")
        A_("")
        A_("Full `MPCGenerator` episodes.  Posture and strike timing come off the")
        A_("SAME rollouts, so the trade each mechanism makes is readable in one")
        A_("row: a torso held still that stops hitting the drums is not a win.")
        A_("")
        A_("`hit rate` is `1 - miss_rate`: the fraction of SCHEDULED strikes whose")
        A_("stick actually crossed the head plane inside the disk.  It is stated as")
        A_("a success rate so it reads in the same direction as open-loop")
        A_("acceptance -- but the two are NOT complements and should not be")
        A_("subtracted from each other.  Acceptance is a conjunction of six gates")
        A_("(hit + speed + collision + torque + velocity + box) over one plan from")
        A_("a clean start; hit rate asks only whether the stick arrived, over 250")
        A_("chained re-plans against a PD plant.  A plan can be rejected open loop")
        A_("for a torque excess and still land the hit, and vice versa.")
        A_("")
        A_("| arm | eps | div | pitch p90 | roll p90 | yaw p90 | lean p90 | "
           "lean>0.15 | yaw travel | hit rate | \\|lat\\| p50 ms |")
        A_("|---|---|---|---|---|---|---|---|---|---|---|")
        for label, _run, _m in ARMS:
            c = cl.get(label)
            if not c:
                continue
            # NOT ``L`` -- that name is the output line list this whole function
            # appends to, and rebinding it made ``"\n".join(L)`` emit the
            # latency dict's KEYS instead of the report.
            lt = c.get("latency", {})
            A_(f"| `{label}` | {c['n_traj']} | {c['n_diverged']} | "
               f"{_f(c['pitch_abs_p90'])} | {_f(c['roll_abs_p90'])} | "
               f"{_f(c['yaw_abs_p90'])} | {_f(c['lean_norm_p90'])} | "
               f"{_f(c['frac_lean_over_0.15'])} | {_f(c['yaw_travel_p50'], 2)} | "
               f"{_f(1.0 - lt['miss_rate']) if 'miss_rate' in lt else '--'} | "
               f"{_f(lt.get('lat_abs_p50_ms'), 1)} |")
        A_("")

    # --- where the excursion enters, and does it ratchet -------------------
    ds = {k: v.get("lean_sources") for k, v in report["arms"].items()
          if v.get("lean_sources")}
    if ds:
        A_("## Where the waist excursion enters, and does it ratchet")
        A_("")
        A_("One episode per arm.  `plan` is the planner's own waypoint 1 -- the")
        A_("only thing the cost ever prices; `cmd` is that after the reference")
        A_("lead, slew limit and box clamp; `meas` is what the plant reached.")
        A_("**If the lead and track gaps are both small, the excursion is the")
        A_("plan's own** and a cost weight can reach it; if they are not, no")
        A_("weight will.")
        A_("")
        for tag in sorted({t for v in ds.values() for t in v}):
            A_(f"### `{tag}`")
            A_("")
            A_("| arm | joint | plan p50 | cmd p50 | meas p50 | lead gap | track gap |")
            A_("|---|---|---|---|---|---|---|")
            for label, _run, _m in ARMS:
                d = ds.get(label, {}).get(tag)
                if not d:
                    continue
                for j in ("yaw", "pitch", "roll"):
                    A_(f"| `{label}` | {j} | {_f(d[j+'_plan_abs_p50'])} | "
                       f"{_f(d[j+'_cmd_abs_p50'])} | {_f(d[j+'_meas_abs_p50'])} | "
                       f"{_f(d[j+'_lead_gap_p50'], 4)} | "
                       f"{_f(d[j+'_track_gap_p50'], 4)} |")
            A_("")
            A_("Ratchet test -- least-squares fit of the MEASURED joint against")
            A_("time over the episode.  `slope` is the systematic creep, `resid`")
            A_("the jitter left after removing it, `net` the signed end-minus-start")
            A_("displacement.  **Large slope + small resid = a ratchet**, which")
            A_("needs an absolute anchor; the reverse is jitter, which needs rate")
            A_("damping.  A temporal-consistency price barely touches a ratchet:")
            A_("a steady march agrees with itself tick to tick.")
            A_("")
            A_("| arm | joint | slope rad/s | resid rms | net rad | range rad |")
            A_("|---|---|---|---|---|---|")
            for label, _run, _m in ARMS:
                d = ds.get(label, {}).get(tag)
                if not d:
                    continue
                for j in ("yaw", "pitch", "roll"):
                    if f"{j}_slope_rad_s" not in d:
                        continue
                    A_(f"| `{label}` | {j} | {_f(d[j+'_slope_rad_s'], 4)} | "
                       f"{_f(d[j+'_resid_rms'], 4)} | {_f(d[j+'_net_rad'], 3)} | "
                       f"{_f(d[j+'_range_rad'], 3)} |")
            A_("")
    return "\n".join(L) + "\n"


# ============================================================== driver =======
def main(argv=None) -> int:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="digest",
                    choices=["digest", "eval", "closed", "diagnose", "all"])
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--ckpt-name", default="best.pt",
                    help="which checkpoint per run dir (best.pt / last.pt)")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of arm labels; default all")
    ap.add_argument("--grids", nargs="+", default=["standard", "near_drum"])
    ap.add_argument("--grid-seeds", type=int, nargs="+", default=[0, 7])
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lead", default="fit")
    ap.add_argument("--tail-frac", type=float, default=0.25)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="ablation_results.md")
    ap.add_argument("--json-out", default="runs/final/waist_ablation.json")
    ap.add_argument("--notes", default="ablation_notes.md",
                    help="markdown file inlined verbatim after the header (the "
                         "hand-written verdict).  Missing file = no notes")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="render the report from existing JSON sidecars instead "
                         "of evaluating.  Lets the five arms be split across "
                         "GPUs in parallel and stitched back into one table; "
                         "later files win on a per-arm collision")
    args = ap.parse_args(argv)

    def _notes() -> str:
        q = Path(args.notes)
        if not q.is_absolute():
            q = ROOT / q
        return q.read_text() if q.exists() else ""

    # ---- merge mode: no model, no GPU, just re-render ---------------------
    if args.merge is not None:
        merged: dict = {"arms": {}}
        for spec in args.merge:
            p = Path(spec)
            if not p.is_absolute():
                p = ROOT / p
            part = json.loads(p.read_text())
            merged.setdefault("generated", part.get("generated", "?"))
            merged["generated"] = part.get("generated", merged["generated"])
            for label, blk in (part.get("arms") or {}).items():
                # Later shards win, but never let a shard that only ran the
                # digest blank out another shard's eval blocks for that arm.
                cur = merged["arms"].setdefault(label, {})
                for k, v in blk.items():
                    if v or k not in cur:
                        cur[k] = v
            print(f"[merge] {p.name}: {sorted((part.get('arms') or {}))}", flush=True)
        merged["stage"] = "merged"
        mout = Path(args.out)
        if not mout.is_absolute():
            mout = ROOT / mout
        mout.write_text(render(merged, _notes()))
        jout = Path(args.json_out)
        if not jout.is_absolute():
            jout = ROOT / jout
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps(merged, indent=1, default=str))
        print(f"\n[out] {mout}\n[out] {jout}")
        return 0

    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]
    want_eval = args.stage in ("eval", "all")
    want_closed = args.stage in ("closed", "all")
    want_diag = args.stage in ("diagnose", "all")

    report: dict = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "stage": args.stage, "ckpt": args.ckpt_name, "arms": {},
    }

    # ---- stage 1: always, and free -------------------------------------
    for label, run, _mech in arms:
        rd = ROOT / "runs" / run
        d = digest_run(rd, tail_frac=args.tail_frac)
        report["arms"][label] = {"run": run, "digest": d, "config": read_config(rd)}
        if d.get("present"):
            print(f"[digest] {label:8s} step {d['last_step']:6d}"
                  f"{'' if d.get('done') else ' (running)'}"
                  f"  best_acc {_f(d.get('best_acceptance'))}"
                  f"  pr_p90 {_f(d.get('pr_p90'))}"
                  f"  yaw_p90 {_f(d.get('yaw_p90'))}"
                  f"  yawd_p90 {_f(d.get('yawd_p90'))}", flush=True)
        else:
            print(f"[digest] {label:8s} MISSING ({rd})", flush=True)

    # ---- stages 2/3: need the checkpoints -------------------------------
    if want_eval or want_closed or want_diag:
        import torch

        import amortized_opt as A
        from amortized_opt.lead import LEAD_PRESETS

        dev = torch.device(args.device)
        if dev.type == "cuda":
            torch.cuda.set_device(dev.index or 0)
        torch.set_float32_matmul_precision("high")
        amp = None if args.no_amp else torch.bfloat16
        cmd_lead = (LEAD_PRESETS[args.lead] if args.lead in LEAD_PRESETS
                    else float(args.lead))

        kit = A.DrumKit.from_layout()
        library = A.GoalPoseLibrary()
        scorer = A.TrajectoryScorer(device="cpu")
        clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: args.clips]

        for label, run, _mech in arms:
            ck = ROOT / "runs" / run / args.ckpt_name
            if not ck.exists():
                print(f"[eval] {label}: no {ck}, skipping", flush=True)
                continue
            model, cfg, step = load_model(ck, dev)
            print(f"\n=== {label}  ({run}/{args.ckpt_name}, step {step}, "
                  f"frozen={cfg['frozen_joints'] or 'none'}) ===", flush=True)
            blk = report["arms"][label]
            blk["ckpt_step"] = step
            blk["frozen_joints"] = list(cfg["frozen_joints"])

            if want_eval:
                blk["open_loop"] = {}
                for gs in args.grid_seeds:
                    for grid in args.grids:
                        b = open_loop(model, cfg, kit, library, scorer, dev, amp,
                                      grid=grid, seed=gs)
                        blk["open_loop"][f"{grid}_seed{gs}"] = b
                        w = b["waist"]
                        print(f"  OL {grid} s{gs} ({b['n']}): acc {_f(b['acceptance'])}"
                              f"  pitch p90 {_f(w['pitch_abs_p90'])}"
                              f"  roll p90 {_f(w['roll_abs_p90'])}"
                              f"  yaw p90 {_f(w['yaw_abs_p90'])}", flush=True)
            if want_closed:
                print("  closed loop:", flush=True)
                blk["closed_loop"] = closed_loop(
                    model, cfg, kit, scorer, dev, amp, clips=clips,
                    probe_seeds=args.probe_seeds, steps=args.steps,
                    cmd_lead=cmd_lead)
                c = blk["closed_loop"]
                L = c.get("latency", {})
                print(f"  CL: {c['n_traj']} eps, {c['n_diverged']} diverged, "
                      f"lean p90 {_f(c['lean_norm_p90'])}, "
                      f"yaw p90 {_f(c['yaw_abs_p90'])}, "
                      f"miss {_f(L.get('miss_rate'))}, "
                      f"|lat| p50 {_f(L.get('lat_abs_p50_ms'), 1)} ms", flush=True)
            if want_diag:
                print("  drift diagnosis:", flush=True)
                blk["lean_sources"] = {}
                for tag, cl_ in (("lead_fit", cmd_lead), ("no_lead", 0.0)):
                    d = diagnose_lean(model, cfg, kit, dev, amp, clip=clips[0],
                                      seed=0, steps=args.steps, cmd_lead=cl_)
                    blk["lean_sources"][tag] = d
                    for j in ("yaw", "pitch", "roll"):
                        print(f"    [{tag:8s}] {j:5s} |q| p50 plan "
                              f"{_f(d[j+'_plan_abs_p50'])} -> cmd "
                              f"{_f(d[j+'_cmd_abs_p50'])} -> meas "
                              f"{_f(d[j+'_meas_abs_p50'])}"
                              f"  | gaps lead {_f(d[j+'_lead_gap_p50'], 4)}"
                              f" track {_f(d[j+'_track_gap_p50'], 4)}"
                              f"  | slope {_f(d[j+'_slope_rad_s'], 4)} rad/s"
                              f" resid {_f(d[j+'_resid_rms'], 4)}"
                              f" net {_f(d[j+'_net_rad'], 3)}", flush=True)

    # ---- write out --------------------------------------------------------
    jout = Path(args.json_out)
    if not jout.is_absolute():
        jout = ROOT / jout
    jout.parent.mkdir(parents=True, exist_ok=True)
    jout.write_text(json.dumps(report, indent=1, default=str))

    mout = Path(args.out)
    if not mout.is_absolute():
        mout = ROOT / mout
    mout.write_text(render(report, _notes()))
    print(f"\n[out] {mout}\n[out] {jout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
