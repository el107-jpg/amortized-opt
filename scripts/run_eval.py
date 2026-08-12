#!/usr/bin/env python3
"""Run any planner over the evaluation grid and persist cases / scores / paths.

Two phases, deliberately separated:

1. **Solve** -- the planner is called once per case, optionally fanned out over
   a process pool (the repo's SQP is single-threaded CPU work, so the pool is
   where the wall-clock win is).  Only trajectories come back.
2. **Score** -- every returned trajectory is scored in the parent process by one
   :class:`~amortized_opt.eval_harness.TrajectoryScorer`.  Scoring is a few tens
   of ms per case and is *not* what the pool is for; keeping it in the parent
   means one scorer, one device, and no CUDA-in-a-forked-child hazard.

Because the trajectories are persisted to NPZ alongside the cases, a later
change to the metrics can be re-run against stored baseline paths with
``--rescore`` instead of re-solving (which matters: the repo SQP is
nondeterministic -- roughly a third of cases move run to run -- so re-solving
would confound a metric change with solver noise).

Examples::

    # the reference baseline (repo-coupled)
    python scripts/run_eval.py \\
        --planner repo_adapters.trajopt_hpipm_planner:make_planner \\
        --planner-args '{"backend": "hpipm"}' \\
        --grid standard --jobs 16 \\
        --out-dir baselines --prefix trajopt_hpipm_2026-08-10

    # re-score stored trajectories after a metric change
    python scripts/run_eval.py --rescore baselines --prefix trajopt_hpipm_2026-08-10

    # sanity check with the built-in straight-line planner
    python scripts/run_eval.py --planner builtin:quintic --grid standard --jobs 8
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# One BLAS thread per worker: the fan-out is over processes.  MUST precede numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
for _p in (str(_PKG), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from amortized_opt.eval_harness import (  # noqa: E402
    AcceptanceThresholds,
    EvalCase,
    PlanResult,
    TrajectoryScorer,
    aggregate,
    format_summary,
    generate_grid,
    load_run,
    save_run,
)


# ------------------------------------------------------------- builtin plans --
def _builtin_planner(kind: str):
    """Trivial reference planners, used by the tests and for smoke runs.

    * ``quintic`` -- a joint-space quintic from the start pose to a terminal
      pose obtained by damped-least-squares IK onto the strike aim.  Not good,
      but a real trajectory: it moves, it usually misses, and it frequently
      drives straight through a drum -- which is exactly what the harness's
      violation metrics should light up on.
    * ``hold`` -- stay at the start pose.  Zero travel, guaranteed miss.
    * ``line_through_drum`` -- a straight joint-space line from the start pose to
      the *cached* terminal pose with no obstacle awareness at all.
    """
    import amortized_opt as A
    from amortized_opt.constants import N_DOF

    scorer = TrajectoryScorer()

    def hold(case: EvalCase) -> PlanResult:
        return PlanResult(
            Q=np.repeat(np.asarray(case.q_start)[None, :], case.n_waypoints, axis=0),
            solve_s=0.0,
        )

    def _ik(case: EvalCase, iters: int = 120) -> np.ndarray:
        """Damped least-squares IK of the contact point onto the strike aim."""
        q = np.asarray(case.q_start, float).copy()
        sigma = case.strike.resolved_contact_sigma()
        target = np.asarray(case.strike.position, float)
        lo, hi = scorer.q_lo_post, scorer.q_hi_post
        for _ in range(iters):
            p, _tip, _d = scorer.contact_world(q, case.hand, sigma)
            e = target - p
            if float(np.linalg.norm(e)) < 1e-4:
                break
            J = scorer.tip_jacobian(q, case.hand, sigma)
            dq = J.T @ np.linalg.solve(J @ J.T + (0.05**2) * np.eye(3), e)
            q = np.clip(q + 0.5 * dq, lo, hi)
        return q

    def quintic(case: EvalCase) -> PlanResult:
        t0 = time.perf_counter()
        q_end = _ik(case)
        T = case.n_waypoints
        s = np.linspace(0.0, 1.0, T)[:, None]
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        Q = np.asarray(case.q_start)[None, :] * (1 - blend) + q_end[None, :] * blend
        return PlanResult(Q=Q, solve_s=time.perf_counter() - t0)

    def line_through_drum(case: EvalCase) -> PlanResult:
        t0 = time.perf_counter()
        q_end = _ik(case)
        T = case.n_waypoints
        s = np.linspace(0.0, 1.0, T)[:, None]
        Q = np.asarray(case.q_start)[None, :] * (1 - s) + q_end[None, :] * s
        return PlanResult(Q=Q, solve_s=time.perf_counter() - t0)

    return {"hold": hold, "quintic": quintic, "line_through_drum": line_through_drum}[kind]


def load_planner(spec: str, kwargs: dict):
    """``"pkg.mod:factory"`` -> ``factory(**kwargs)``; ``"builtin:<kind>"`` too."""
    mod_name, _, attr = spec.partition(":")
    if mod_name == "builtin":
        return _builtin_planner(attr or "quintic")
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, attr or "make_planner")
    return factory(**kwargs)


# ------------------------------------------------------------------ workers --
_W: dict = {}


def _worker_init(spec: str, kwargs_json: str) -> None:
    _W["planner"] = load_planner(spec, json.loads(kwargs_json))


def _worker_solve(payload: dict) -> dict:
    case = EvalCase.from_dict(payload)
    t0 = time.perf_counter()
    try:
        res = _W["planner"](case)
        wall = time.perf_counter() - t0
        if isinstance(res, PlanResult):
            Q, solve_s, meta = res.Q, res.solve_s, dict(res.meta)
        else:
            Q, solve_s, meta = np.asarray(res, float), wall, {}
        if not np.isfinite(solve_s):
            solve_s = wall
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the batch
        wall = time.perf_counter() - t0
        Q = np.repeat(np.asarray(case.q_start)[None, :], case.n_waypoints, axis=0)
        solve_s = wall
        meta = {"failed": True, "message": f"{type(exc).__name__}: {exc}"}
        print(f"  !! case {case.case_id} raised {exc!r}", flush=True)
    return {
        "case_id": int(case.case_id),
        "Q": np.asarray(Q, float),
        "solve_s": float(solve_s),
        "meta": meta,
    }


# --------------------------------------------------------------------- main --
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--planner", default="builtin:quintic",
                    help="'pkg.mod:factory' or 'builtin:{hold,quintic,line_through_drum}'")
    ap.add_argument("--planner-args", default="{}", help="JSON kwargs for the factory")
    ap.add_argument("--grid", default="standard",
                    help="standard | near_drum | standard+near_drum")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", type=Path, default=None)
    ap.add_argument("--include-extra-drums", action="store_true",
                    help="add floor_tom (absent from the repo's 162-case grid)")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (debug)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="cpu",
                    help="scorer device; cpu is fastest at these batch sizes")
    ap.add_argument("--collision-substeps", type=int, default=4)
    ap.add_argument("--strike-exempt-waypoints", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix", default="run")
    ap.add_argument("--rescore", type=Path, default=None,
                    help="re-score stored trajectories in this directory")
    ap.add_argument("--tag", default="", help="free-text note stored in the metadata")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        import torch

        torch.cuda.set_device(int(args.device.split(":")[1]) if ":" in args.device else 0)

    scorer = TrajectoryScorer(
        device=args.device,
        layout_path=args.layout,
        collision_substeps=args.collision_substeps,
        strike_exempt_waypoints=args.strike_exempt_waypoints,
        thresholds=AcceptanceThresholds(),
    )

    # ---------------------------------------------------------- re-score mode --
    if args.rescore is not None:
        cases, trajs, meta = load_run(args.rescore, prefix=args.prefix)
        rows = []
        for c in cases:
            Q = trajs.get(int(c.case_id))
            if Q is None:
                continue
            pm = (meta.get("per_case", {}) or {}).get(str(c.case_id), {})
            rows.append(
                scorer.score(c, Q, solve_s=float(pm.get("solve_s", float("nan"))),
                             meta=pm.get("meta", {}))
            )
        title = f"{args.prefix} (re-scored from stored NPZ)"
        text = format_summary(rows, title)
        print(text)
        out = Path(args.rescore)
        from amortized_opt.eval_harness import write_csv

        write_csv(out / f"{args.prefix}_per_case.csv", rows)
        (out / f"{args.prefix}_summary.json").write_text(
            json.dumps(aggregate(rows), indent=1)
        )
        # keep the three artefacts in step -- a stale .txt next to a fresh .csv
        # is exactly how a re-scored baseline gets misread
        (out / f"{args.prefix}_summary.txt").write_text(text)
        return 0

    # ------------------------------------------------------------- solve mode --
    cases = generate_grid(
        seed=args.seed, grid=args.grid, scorer=scorer, layout_path=args.layout,
        include_extra_drums=args.include_extra_drums,
    )
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases  grid={args.grid}  seed={args.seed}  "
          f"planner={args.planner}  jobs={args.jobs}", flush=True)

    payloads = [c.to_dict() for c in cases]
    kwargs_json = args.planner_args
    t0 = time.perf_counter()
    if args.jobs > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs, initializer=_worker_init,
                      initargs=(args.planner, kwargs_json)) as pool:
            results = pool.map(_worker_solve, payloads, chunksize=1)
    else:
        _worker_init(args.planner, kwargs_json)
        results = []
        for i, p in enumerate(payloads):
            r = _worker_solve(p)
            results.append(r)
            if not args.quiet and i % 10 == 0:
                print(f"  {i + 1}/{len(payloads)}  {r['solve_s'] * 1e3:.0f} ms", flush=True)
    solve_wall = time.perf_counter() - t0
    print(f"solve phase done in {solve_wall:.1f}s "
          f"({solve_wall / max(len(cases), 1):.2f}s/case)", flush=True)

    by_id = {int(r["case_id"]): r for r in results}
    rows, trajs, per_case_meta = [], {}, {}
    t1 = time.perf_counter()
    for c in cases:
        r = by_id[int(c.case_id)]
        rows.append(scorer.score(c, r["Q"], solve_s=r["solve_s"], meta=r["meta"]))
        trajs[int(c.case_id)] = r["Q"]
        per_case_meta[str(int(c.case_id))] = {"solve_s": r["solve_s"], "meta": r["meta"]}
    print(f"score phase done in {time.perf_counter() - t1:.1f}s", flush=True)

    print(format_summary(rows, f"{args.prefix}  [{args.planner}  grid={args.grid}]"))

    # ------------------------------------------------------------- persistence --
    if args.out_dir is not None:
        planner_desc = {}
        try:
            planner = load_planner(args.planner, json.loads(kwargs_json))
            mod_name = args.planner.partition(":")[0]
            if mod_name != "builtin":
                m = importlib.import_module(mod_name)
                if hasattr(m, "describe"):
                    planner_desc = m.describe(planner)
        except Exception as exc:  # noqa: BLE001 - provenance is best-effort
            planner_desc = {"describe_failed": repr(exc)}

        meta = {
            "prefix": args.prefix,
            "tag": args.tag,
            "planner": args.planner,
            "planner_args": json.loads(kwargs_json),
            "planner_describe": planner_desc,
            "grid": args.grid,
            "seed": args.seed,
            "n_cases": len(cases),
            "collision_substeps": args.collision_substeps,
            "strike_exempt_waypoints": args.strike_exempt_waypoints,
            "scorer_device": args.device,
            "thresholds": vars(AcceptanceThresholds()),
            "solve_wall_s": solve_wall,
            "jobs": args.jobs,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_head": _git_head(),
            "per_case": per_case_meta,
        }
        paths = save_run(args.out_dir, cases, rows, trajs, prefix=args.prefix, meta=meta)
        (Path(args.out_dir) / f"{args.prefix}_summary.json").write_text(
            json.dumps(aggregate(rows), indent=1)
        )
        (Path(args.out_dir) / f"{args.prefix}_summary.txt").write_text(
            format_summary(rows, args.prefix)
        )
        for k, v in paths.items():
            print(f"wrote {v}")
    return 0


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PKG.parent), text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
