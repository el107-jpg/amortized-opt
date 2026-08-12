"""Closed-loop hit latency, measured the SAME way `REPORT.md` §4 measured it.

The headline number of the RL-1 stage is the Isaac hit-latency histogram against
the analytic-plant 63 % miss rate.  For that comparison to mean anything the two
must be produced by the same measurement, not by two plausible ones, so this
module reuses ``amortized_opt.probe.hit_latency`` verbatim rather than
reimplementing it:

    for each scheduled strike, scan the plant over +/- 5 control ticks around
    the scheduled instant, find the FIRST crossing of the head plane that is
    inside the legal disk, and locate it to SUB-TICK resolution by interpolating
    the axial coordinate.  The in-plane test and the impact speed are evaluated
    at the interpolated pose.  A strike that never crosses in-disk in the window
    is a genuine MISS -- which the at-tick binary conflates with "late".

``probe.hit_latency`` wants a single-env ``EpisodeResult``-shaped object
(``q_hist``, ``hit_steps``).  The Isaac loop records ``(T, B, 17)``, so the only
work here is to slice it per env and hand each slice over.  That is deliberately
the *slow* path: it runs once at the end of an evaluation, on the host, and
being bit-identical to the offline measurement is worth far more than being
fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amortized_opt.constants import CONTROL_DT
from amortized_opt.eval_harness import TrajectoryScorer
from amortized_opt.probe import hit_latency, summarize_latency

from .rnea_cache import rnea_from_cache_if_needed

__all__ = ["EpisodeView", "latency_report", "per_env_latency"]


@dataclass
class EpisodeView:
    """The minimal duck-type ``probe.hit_latency`` reads off an episode."""

    q_hist: np.ndarray                     # (T + 1, 17)
    hit_steps: list                        # [(step, hand, drum)]


def per_env_latency(loop, kit, *, scorer: TrajectoryScorer | None = None,
                    window_ticks: int = 5, dt: float = CONTROL_DT):
    """One list of ``probe.hit_latency`` rows per env, tagged with the env id."""
    if not loop.history:
        return []
    scorer = scorer or TrajectoryScorer(device="cpu")
    Q = loop.q_hist()                                  # (T, B, 17)
    T, B, _n = Q.shape
    # ``probe`` indexes ``q_hist[step]`` with step 0 = the state BEFORE the
    # first command, so prepend the initial pose (the loop records post-step).
    events: dict[int, list] = {b: [] for b in range(B)}
    for (tick, b, hand, drum) in loop.fired_events():
        events[b].append((int(tick), hand, int(drum)))
    rows = []
    for b in range(B):
        if not events[b]:
            continue
        qh = np.concatenate([Q[0:1, b], Q[:, b]], axis=0)
        view = EpisodeView(q_hist=qh, hit_steps=events[b])
        for r in hit_latency(scorer, kit, view, window_ticks=window_ticks, dt=dt):
            r["env"] = b
            rows.append(r)
    return rows


def latency_report(loop, kit, *, device: str = "cpu", window_ticks: int = 5,
                   dt: float = CONTROL_DT) -> dict:
    """The 7c table, plus per-hand and per-drum breakdowns.

    The keys are ``probe.summarize_latency``'s, so the numbers drop straight
    into the same columns as README §7c's "the metric that actually ranks the
    knobs" table and `REPORT.md` §4.2.
    """
    # TrajectoryScorer builds its own RNEA and exposes no injection point; the
    # cache fallback is what lets the headline measurement run INSIDE Isaac.
    with rnea_from_cache_if_needed():
        scorer = TrajectoryScorer(device=device)
    rows = per_env_latency(loop, kit, scorer=scorer, window_ticks=window_ticks, dt=dt)
    out = {"overall": summarize_latency(rows), "n_strikes": len(rows)}
    for key in ("hand", "drum"):
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(str(r[key]), []).append(r)
        out[f"by_{key}"] = {k: summarize_latency(v) for k, v in sorted(groups.items())}
    return out
