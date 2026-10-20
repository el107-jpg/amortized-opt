#!/usr/bin/env python3
"""Run prepared physical robustness profiles on the three spare GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve()
META = json.loads((OUT/"config.json").read_text())


def save(name, value):
    tmp = OUT/(name+".tmp")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(OUT/name)


jobs = META["jobs"]
priority = {name:i for i,name in enumerate(["nominal", "stick_angles_10deg", "dynamics_strong",
    "trained_range_positions", "positions_2x", "combined_strong"])}
pending = sorted(jobs, key=lambda j: (j["seed"], priority[j["name"]]))
active = {}
finished = []
begin = time.monotonic()
try:
    while pending or active:
        for gpu in (1, 2, 3):
            if gpu in active or not pending:
                continue
            j = pending.pop(0)
            cmd = [sys.executable, "-u", "scripts/play_isaac.py", "--headless",
                "--device", f"cuda:{gpu}", "--model_device", f"cuda:{gpu}",
                "--ckpt", META["checkpoint"], "--rhythm_path", META["source_rhythm"],
                "--steps", str(META["steps"]), "--seed", str(j["seed"]),
                "--direct_pid", "--ki", "5", "--ki_windup", "0.25", "--cmd_lead_s", "fit_kd4",
                "--contact_compliance", "--no_overlay", "--no_sticks",
                "--domain_probe_config", j["profile"], "--assignment_comparison_config", j["config"]]
            folder = Path(j["out"])
            with (folder/"isaac.log").open("w") as log:
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1"))
            active[gpu] = (p, j)
            print(f"Started {j['name']} seed{j['seed']} GPU{gpu} PID{p.pid}", flush=True)
        for gpu, (p, j) in list(active.items()):
            if p.poll() is None:
                continue
            if p.returncode:
                raise RuntimeError(f"{j['name']} seed{j['seed']} exited {p.returncode}; inspect {j['out']}/isaac.log")
            result = Path(j["out"])/"summary.json"
            if not result.exists():
                raise RuntimeError("Missing actual Isaac result")
            finished.append(j)
            del active[gpu]
            print(f"Finished {j['name']} seed{j['seed']}", flush=True)
        save("run.json", dict(status="running", completed=len(finished), total=len(jobs),
            elapsed_s=time.monotonic()-begin,
            active=[dict(name=j["name"], seed=j["seed"], pid=p.pid, gpu=g) for g,(p,j) in active.items()]))
        time.sleep(3)
    save("run.json", dict(status="complete", completed=len(finished), total=len(jobs), elapsed_s=time.monotonic()-begin))
except BaseException as exc:
    for p,j in active.values():
        if p.poll() is None:
            p.kill()
    save("run.json", dict(status="failed", error=str(exc), completed=len(finished)))
    raise
