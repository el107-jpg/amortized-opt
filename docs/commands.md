## AL monitor (latest train row)

```bash
python3 - <<'PY'
import json
from pathlib import Path

last = None
for m in map(json.loads, Path("runs/rhythm_bellman_nf1_ls_grav_static_tight/metrics.jsonl").open()):
    if m.get("event") == "train":
        last = m

print(f"step {last['step']}\n")
lam, pen = last["lam_i"], last.get("pen", {})
print(f"{'group':<28} {'lambda':>10} {'viol':>12} {'pen':>12}")
for k in sorted(lam):
    v = last.get("v_" + k)
    print(f"{k:<28} {lam[k]:10.4f} {(f'{v:.6g}' if v is not None else ''):>12} "
          f"{(f'{pen[k]:.6g}' if k in pen else ''):>12}")
PY
```

## TensorBoard

Training writes scalars to ``runs/<tag>/tb`` (on by default). Install once:

```bash
pip install -e '.[monitoring]'
```

Launch **only the active runs** (recommended):

```bash
bash scripts/tensorboard_phase2.sh
```

Sidebar: ``runs/*/tb`` dirs written in the last ``MAX_AGE_HOURS`` hours (default 48), capped at ``MAX_RUNS`` (default 8), named by run tag. Override with ``LOGDIR_SPEC='name:path,...'``. Default port **6008** (6006/6007 often taken). The launcher binds all interfaces and sets ``--load_fast=false`` so the Time Series dashboard actually populates.

Open in a **browser**: ``http://localhost:6008``  
SSH tunnel: ``ssh -L 6008:localhost:6008 eric@ubuntu-5090.tail57ea44.ts.net``

**Scalar sections** (grouped by ``/`` in the Scalars tab):

| prefix | what |
|--------|------|
| ``loss/`` | loss, cost, aux, tc, bptt |
| ``terms/`` | cost-term hinges (torque, collision, …) |
| ``violations/`` | AL raw ``v_*`` |
| ``al/lambda``, ``al/pen``, ``al/weight`` | λ, PHR pen, multipliers |
| ``diagnostics/hit``, ``collision``, ``torque``, ``waist`` | batch diagnostics |
| ``optimizer/`` | lr, grad_norm, steps/s |
| ``schedule/``, ``line_search/`` | curriculum α, SaLSa |
| ``grad/``, ``grad_balance/`` | per-term grad probes |
| ``eval/rates``, ``eval/errors``, ``eval/counts`` | rolling 81-grid |
| ``probe/`` | closed-loop probe |

New sections apply to **restarted** runs; old ``tb/`` event files keep the flat tags.

Disable per job: ``--no-tensorboard``.

## Frontier auto-monitor (30 min)

When closed-loop probe ``axial_abs_p50_mm`` ≤ **30 mm** and recent train
violations are not spiking, enable frontier collocation (``FRONTIER=1`` +
``FULL_WEIGHTS=1`` snap).  Active runs default ``FRONTIER_AUTO_MISS_MM=30`` on
the rhythm launchers.

```bash
# one status poll
bash scripts/frontier_auto_monitor.sh

# background loop every 30 minutes
bash scripts/frontier_auto_monitor.sh --daemon
# log: runs/.frontier_monitor/monitor.log
```

The trainer also auto-enables on probe when ``--frontier-auto-miss-mm`` is set;
the monitor writes ``runs/<tag>/.frontier_activate`` as a backup trigger.

## Checkpoints

``last.pt`` is written every **500** steps (``CKPT_LAST_EVERY``); a retained
``step_NNNNNN.pt`` is written on the same cadence (``CKPT_KEEP_EVERY``, default
500; never overwritten).  Full bundle (``last.pt`` + ``frontier.pt`` / plot)
every **2000** (``CKPT_EVERY``).  SIGTERM/SIGINT save ``last.pt`` at the next
step boundary (atomic write via ``.tmp`` + rename).  Restarts should use
``RESUME=runs/<tag>/last.pt``; plot a specific step with
``runs/<tag>/step_004000.pt``.
