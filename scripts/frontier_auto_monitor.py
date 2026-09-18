#!/usr/bin/env python3
"""Poll phase-2 metrics and trigger frontier collocation when probe miss is low.

Writes ``<run_dir>/.frontier_activate`` for in-process pickup (trainer checks
every log line and after each probe).  Optionally restarts a run with
``FRONTIER=1 FULL_WEIGHTS=1`` when the live process lacks auto-frontier support.

Example (30-minute daemon, both active runs):

    nohup python scripts/frontier_auto_monitor.py --daemon \\
      >> runs/.frontier_monitor/monitor.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from amortized_opt.frontier_auto import evaluate_run

DEFAULT_RUNS = (
    {
        "tag": "rhythm_bellman_nf1_ls_rnea_from_grav",
        "run_dir": REPO / "runs/rhythm_bellman_nf1_ls_rnea_from_grav",
        "launcher": REPO / "scripts/_run_rhythm_bellman_rnea_from_grav.sh",
        "dev": "cuda:1",
        "log": REPO / "runs/rhythm_bellman_nf1_ls_rnea_from_grav_launch.log",
    },
    {
        "tag": "rhythm_bellman_nf1_adam_waist_fresh",
        "run_dir": REPO / "runs/rhythm_bellman_nf1_adam_waist_fresh",
        "launcher": REPO / "scripts/_run_rhythm_bellman_adam_armijo.sh",
        "dev": "cuda:2",
        "log": REPO / "runs/rhythm_bellman_nf1_adam_waist_fresh_launch.log",
    },
)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _pid_for_out(out: Path) -> int | None:
    try:
        out = out.resolve()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    "utf-8", errors="replace"
                )
            except OSError:
                continue
            if "train_phase2.py" not in cmd:
                continue
            if f"--out {out}" in cmd or f"--out {out}/" in cmd:
                return int(entry.name)
    except OSError:
        return None
    return None


def _write_activate(run_dir: Path) -> None:
    flag = run_dir / ".frontier_activate"
    flag.write_text(f"requested_at={_ts()}\n", encoding="utf-8")


def _restart_with_frontier(spec: dict) -> None:
    run_dir = Path(spec["run_dir"])
    tag = spec["tag"]
    dev = spec["dev"]
    launcher = Path(spec["launcher"])
    log_path = Path(spec.get("log", REPO / f"runs/{tag}_frontier_restart.log"))
    pid = _pid_for_out(run_dir)
    if pid is not None:
        print(f"[{_ts()}] {tag}: SIGTERM pid {pid} (graceful ckpt)", flush=True)
        os.kill(pid, signal.SIGTERM)
        last_pt = run_dir / "last.pt"
        t0 = last_pt.stat().st_mtime if last_pt.is_file() else 0.0
        for _ in range(90):
            if _pid_for_out(run_dir) is None:
                break
            if last_pt.is_file() and last_pt.stat().st_mtime > t0:
                break
            time.sleep(2)
        live = _pid_for_out(run_dir)
        if live is not None:
            print(f"[{_ts()}] {tag}: SIGKILL pid {live}", flush=True)
            os.kill(live, signal.SIGKILL)
    env = os.environ.copy()
    env.update({
        "DEV": dev,
        "TAG": tag,
        "RESUME": str(run_dir / "last.pt"),
        "FRONTIER": "1",
        "FULL_WEIGHTS": "1",
        "LINE_SEARCH_FRESH": "0",
        "FRONTIER_AUTO_MISS_MM": "0",
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_fh:
        log_fh.write(f"\n=== frontier restart {_ts()} ===\n")
        log_fh.flush()
        subprocess.Popen(
            ["bash", str(launcher)],
            cwd=str(REPO),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"[{_ts()}] {tag}: restarted with FRONTIER=1 (log {log_path})", flush=True)


def poll_once(
    runs: tuple[dict, ...],
    *,
    miss_mm: float,
    viol_cap: float,
    viol_ratio: float,
    viol_window: int,
    restart_fallback: bool,
) -> None:
    for spec in runs:
        run_dir = Path(spec["run_dir"])
        tag = spec["tag"]
        st = evaluate_run(
            run_dir,
            miss_mm=miss_mm,
            viol_cap=viol_cap,
            viol_ratio=viol_ratio,
            viol_window=viol_window,
        )
        miss_s = f"{st['miss_mm']:.1f}" if st["miss_mm"] is not None else "n/a"
        print(
            f"[{_ts()}] {tag}: step={st['last_step']} probe@{st['probe_step']} "
            f"miss={miss_s}mm frontier={st['frontier_on']} ready={st['ready']} "
            f"({st['reason']})",
            flush=True,
        )
        if st["frontier_on"] or st["done"]:
            continue
        if not st["ready"]:
            continue
        _write_activate(run_dir)
        print(f"[{_ts()}] {tag}: wrote {run_dir}/.frontier_activate", flush=True)
        if restart_fallback:
            has_auto = False
            pid = _pid_for_out(run_dir)
            if pid is not None:
                try:
                    cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes()
                    has_auto = b"--frontier-auto-miss-mm" in cmd
                except OSError:
                    has_auto = False
            if not has_auto:
                _restart_with_frontier(spec)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=int, default=1800,
                   help="seconds between polls in --daemon mode (default 1800)")
    p.add_argument("--miss-mm", type=float, default=30.0)
    p.add_argument("--viol-cap", type=float, default=8.0)
    p.add_argument("--viol-ratio", type=float, default=3.0)
    p.add_argument("--viol-window", type=int, default=10)
    p.add_argument("--daemon", action="store_true",
                   help="loop forever (for nohup)")
    p.add_argument("--restart-fallback", action="store_true",
                   help="restart with FRONTIER=1 if trainer lacks auto-frontier")
    p.add_argument("--run-dir", action="append", default=[],
                   help="extra run dir to watch (repeatable)")
    args = p.parse_args()

    runs = list(DEFAULT_RUNS)
    for extra in args.run_dir:
        path = Path(extra)
        runs.append({
            "tag": path.name,
            "run_dir": path,
            "launcher": REPO / "scripts/_run_rhythm_bellman.sh",
            "dev": os.environ.get("DEV", "cuda:0"),
            "log": path.parent / f"{path.name}_frontier_restart.log",
        })

    monitor_dir = REPO / "runs/.frontier_monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{_ts()}] frontier monitor: {len(runs)} run(s), interval={args.interval}s "
          f"miss<={args.miss_mm}mm", flush=True)

    while True:
        poll_once(
            tuple(runs),
            miss_mm=args.miss_mm,
            viol_cap=args.viol_cap,
            viol_ratio=args.viol_ratio,
            viol_window=args.viol_window,
            restart_fallback=bool(args.restart_fallback),
        )
        if not args.daemon:
            break
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    main()
