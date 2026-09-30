#!/usr/bin/env python
"""Deterministic, resumable two-stage architecture search.

This is deliberately a small scheduler rather than an Optuna integration.  It
compares architectures at fixed parameter budgets and identical phase budgets,
then promotes only the best completed Phase-1 candidate in each parameter
bucket into Phase 2.  All decisions and subprocess commands are recorded in a
single JSON state file, so a later invocation can resume without guessing what
was run.

The default is a *dry run*.  Training requires ``--execute``.

Example (the requested 5 M / 10 M two-GPU race)::

    python scripts/architecture_search.py --gpus 0,1 --targets 5m,10m
    python scripts/architecture_search.py --gpus 0,1 --targets 5m,10m --execute

For a broader, budget-matched ablation, take the nearest three depth/width
profiles per target and keep the training recipe fixed::

    python scripts/architecture_search.py --gpus 0,1 --targets 5m,10m \\
        --trials-per-target 3 --execute

Additional trainer switches may be supplied with --p1-extra / --p2-extra.  The
same phase recipe and seed policy are used within every parameter budget; do
not put architecture switches in those strings.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STATE_VERSION = 1


@dataclass(frozen=True)
class Architecture:
    d_model: int
    dec_layers: int
    d_ctx: int
    ctx_layers: int
    params: int

    @property
    def name(self) -> str:
        return f"dm{self.d_model}_dl{self.dec_layers}_dc{self.d_ctx}_cl{self.ctx_layers}"

    def cli(self) -> list[str]:
        return [
            "--d-model", str(self.d_model), "--dec-layers", str(self.dec_layers),
            "--d-ctx", str(self.d_ctx), "--ctx-layers", str(self.ctx_layers),
        ]


def parse_budget(text: str) -> int:
    text = text.strip().lower().replace("_", "")
    scale = 1_000_000 if text.endswith("m") else 1
    number = text[:-1] if scale > 1 else text
    try:
        value = int(float(number) * scale)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid parameter target {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("parameter target must be positive")
    return value


def exact_params(d_model: int, dec_layers: int, d_ctx: int, ctx_layers: int) -> int:
    """Ask the current model implementation, rather than maintaining a formula."""
    # CPU construction is cheap, does not initialize CUDA, and keeps the state
    # correct if token/head shapes change in model.py.
    sys.path.insert(0, str(ROOT))
    import amortized_opt as A

    cfg = A.ModelConfig(
        # These are the direct-recipe Phase-1 shapes, not the historical
        # waist-SP experiment.  RoPE itself has no weights but plant features
        # add projections, so both must be represented in the count.
        max_horizon=50, num_future=3, idle_home_token=True,
        waist_sp_head=False, waist_pr_sp_head=False, dec_rope=True,
        # Build without plant features: construction initializes Pinocchio/RNEA
        # even though they are parameter-free, and that makes sizing depend on
        # host physics bindings.  Add its two known Linear projections below.
        dec_plant_feats=False, d_model=d_model, dec_layers=dec_layers,
        d_ctx=d_ctx, ctx_layers=ctx_layers,
    )
    base = int(A.param_counts(A.build_model(cfg, "cpu"))["total"])
    plant_projection_params = 2 * (cfg.plant_feat_dim * d_model + d_model)
    return base + plant_projection_params


def candidate_architectures() -> list[Architecture]:
    """A compact, interpretable width/depth grid; all widths divide 8 heads."""
    candidates: list[Architecture] = []
    for d_model in range(160, 385, 32):
        for d_ctx in (128, 192, 256):
            for ctx_layers in (1, 2):
                for dec_layers in range(2, 7):
                    candidates.append(Architecture(
                        d_model=d_model, dec_layers=dec_layers, d_ctx=d_ctx,
                        ctx_layers=ctx_layers,
                        params=exact_params(d_model, dec_layers, d_ctx, ctx_layers),
                    ))
    return candidates


def choose_architectures(targets: list[int], trials_per_target: int, tolerance: float) -> list[dict[str, Any]]:
    pool = candidate_architectures()
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for target in targets:
        eligible = [a for a in pool if abs(a.params - target) / target <= tolerance]
        if len(eligible) < trials_per_target:
            nearest = min(pool, key=lambda a: abs(a.params - target))
            raise ValueError(
                f"only {len(eligible)} profiles within {tolerance:.0%} of "
                f"{target:,} parameters; nearest is {nearest.params:,} ({nearest.name}). "
                "Increase --budget-tolerance or reduce --trials-per-target."
            )
        # First pick the closest size, then prefer a different depth for each
        # subsequent trial.  This is a transparent architecture ablation, not
        # a noisy black-box optimizer.
        eligible.sort(key=lambda a: (abs(a.params - target), a.dec_layers, a.d_model))
        selected: list[Architecture] = []
        seen_depths: set[int] = set()
        for arch in eligible:
            if arch.name in used:
                continue
            if arch.dec_layers not in seen_depths or len(selected) + 1 >= trials_per_target:
                selected.append(arch)
                seen_depths.add(arch.dec_layers)
            if len(selected) == trials_per_target:
                break
        for rank, arch in enumerate(selected):
            used.add(arch.name)
            chosen.append({
                "id": f"p{target // 1_000_000}m_{rank}_{arch.name}",
                "target_params": target,
                "architecture": asdict(arch),
            })
    return chosen


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_or_create_state(args: argparse.Namespace, state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("version") != STATE_VERSION:
            raise RuntimeError(f"unsupported state version in {state_path}")
        if state["config"]["targets"] != args.targets:
            raise RuntimeError("existing state targets differ; use another --out-root")
        return state
    trials = choose_architectures(args.targets, args.trials_per_target, args.budget_tolerance)
    return {
        "version": STATE_VERSION,
        "created_at": time.time(),
        "config": {
            "targets": args.targets,
            "gpus": args.gpus,
            "p1_steps": args.p1_steps,
            "p2_steps": args.p2_steps,
            "p1_max_best_val": args.p1_max_best_val,
            "p1_extra": args.p1_extra,
            "p2_extra": args.p2_extra,
        },
        "trials": [{
            **trial,
            "phase1": {"status": "pending"},
            "phase2": {"status": "pending"},
        } for trial in trials],
    }


def pid_is_alive(pid: int | None, expected_start_time: int | None = None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return expected_start_time is None or pid_start_time(pid) == expected_start_time


def phase_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def checkpoint_validation(run_dir: Path, phase: int, arch: dict[str, Any]) -> dict[str, Any]:
    """Verify that completion left loadable, architecture-matched checkpoints."""
    import torch

    result: dict[str, Any] = {"ok": False, "checked_at": time.time(), "files": {}}
    for name in ("best.pt", "last.pt"):
        path = run_dir / name
        if not path.is_file():
            result["files"][name] = "missing"
            return result
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("model_cfg")
            if not isinstance(ckpt.get("model"), dict) or not isinstance(cfg, dict):
                raise ValueError("missing model or model_cfg")
            mismatches = {
                k: (cfg.get(k), arch[k]) for k in ("d_model", "dec_layers", "d_ctx", "ctx_layers")
                if int(cfg.get(k, -1)) != int(arch[k])
            }
            # Both direct-recipe Phase-1 heads must remain off.  Phase 2 is
            # permitted to alter behaviour flags in the vetted wrapper, but it
            # must not silently change tensor architecture.
            if phase == 1:
                for k, expected in (("num_future", 3), ("waist_sp_head", False),
                                    ("waist_pr_sp_head", False), ("dec_rope", True),
                                    ("dec_plant_feats", True)):
                    if cfg.get(k) != expected:
                        mismatches[k] = (cfg.get(k), expected)
            if mismatches:
                raise ValueError(f"architecture mismatch: {mismatches}")
            result["files"][name] = {"step": ckpt.get("step"), "ok": True}
        except Exception as exc:  # checkpoint diagnostics must never crash recovery
            result["files"][name] = f"invalid: {exc}"
            return result
    result["ok"] = True
    return result


def reconcile_phase(rec: dict[str, Any], run_dir: Path, phase: int) -> None:
    """Recover completed jobs after the scheduler itself is restarted."""
    state = rec[f"phase{phase}"]
    summary = phase_summary(run_dir)
    if summary and summary.get("event") == "done":
        validation = checkpoint_validation(run_dir, phase, rec["architecture"])
        state.update({"summary": summary, "checkpoint_validation": validation, "finished_at": time.time()})
        state["status"] = "done" if validation["ok"] else "failed"
        if not validation["ok"]:
            state["note"] = "completion summary exists but checkpoints failed validation"
    elif state.get("status") == "running" and not pid_is_alive(
        state.get("pid"), state.get("pid_start_time")
    ):
        # Phase 2 can safely continue from last.pt; phase 1 has no resume CLI
        # and is therefore left explicit rather than silently overwriting it.
        state["status"] = "interrupted"
        state["note"] = "scheduler lost process; use --retry-interrupted after inspection"


def p1_command(args: argparse.Namespace, trial: dict[str, Any], gpu: int, run_dir: Path) -> list[str]:
    arch = Architecture(**trial["architecture"])
    return [
        sys.executable, "scripts/train_phase1_apex.py", "--device", f"cuda:{gpu}",
        "--out", str(run_dir), "--seed", str(args.seed + trial["rank"]),
        "--steps", str(args.p1_steps), "--batch", str(args.p1_batch),
        "--train-scenarios", "500000", "--val-scenarios", "40000",
        "--workers", str(args.workers), "--eval-every", "2000",
        "--horizon", "50", "--buckets", "25", "50", "--init-mode", "mix",
        "--init-mix", "0.5", "--waist-upright-ease-wp", "5",
        "--waist-aim-future-tau", "1.0", "--hit-time-jitter-s", "0.05",
        "--hit-loudness-jitter", "0.2", "--dec-rope", "--dec-plant-feats",
        *arch.cli(), *shlex.split(args.p1_extra),
    ]


def p2_command(args: argparse.Namespace, trial: dict[str, Any], gpu: int, p1_dir: Path, p2_dir: Path) -> list[str]:
    """Build only an explicitly vetted production command or wrapper.

    Phase 2's physics/cost recipe is intentionally not duplicated here.  A
    partial reimplementation would make a superficially comparable search run
    under a different objective, which is worse than refusing to launch.
    """
    arch = Architecture(**trial["architecture"])
    resume = p2_dir / "last.pt"
    values = {
        "python": shlex.quote(sys.executable),
        "device": shlex.quote(f"cuda:{gpu}"),
        "out": shlex.quote(str(p2_dir)),
        "init": shlex.quote(str(p1_dir / "best.pt")),
        "resume": shlex.quote(str(resume)),
        "seed": str(args.seed + trial["rank"]),
        "steps": str(args.p2_steps),
        "batch": str(args.p2_batch),
        "arch_flags": shlex.join(arch.cli()),
        "extra": args.p2_extra,
    }
    if args.phase2_template:
        try:
            return shlex.split(args.phase2_template.format(**values))
        except KeyError as exc:
            raise ValueError(f"unknown placeholder in --phase2-template: {exc}") from exc
    if args.phase2_wrapper:
        # A wrapper owns the full production recipe; these values are supplied
        # as ordinary arguments so a small wrapper can forward them safely.
        command = [args.phase2_wrapper, "--device", f"cuda:{gpu}", "--out", str(p2_dir),
                   "--init", str(p1_dir / "best.pt"), "--seed", str(args.seed + trial["rank"]),
                   "--steps", str(args.p2_steps), "--batch", str(args.p2_batch), *arch.cli()]
        if resume.exists():
            command.extend(["--resume", str(resume)])
        return [*command, *shlex.split(args.p2_extra)]
    raise ValueError("Phase 2 requires --phase2-template or --phase2-wrapper")


def launch(command: list[str], log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab", buffering=0)
    return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def pid_start_time(pid: int) -> int | None:
    """Linux process start tick, used to distinguish a recycled PID."""
    try:
        # Field 22 follows the final ')' in /proc/<pid>/stat; comm can contain
        # spaces, so splitting the full line would be incorrect.
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(tail[19])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def write_manifest(root: Path, trial: dict[str, Any]) -> None:
    atomic_json(root / trial["id"] / "manifest.json", {
        "trial_id": trial["id"], "target_params": trial["target_params"],
        "architecture": trial["architecture"], "phase1": trial["phase1"],
        "phase2": trial["phase2"], "updated_at": time.time(),
    })


def preflight(args: argparse.Namespace, out_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Fail closed before launch when disk or requested GPUs are unavailable."""
    probe = out_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk = shutil.disk_usage(probe)
    free_gib = disk.free / (1024 ** 3)
    if free_gib < args.min_free_disk_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free at {probe}; need --min-free-disk-gib "
            f"{args.min_free_disk_gib:.1f} before launching"
        )
    try:
        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT,
        ).splitlines()
        app_rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("GPU preflight requires a working nvidia-smi") from exc
    gpus: dict[int, dict[str, Any]] = {}
    for row in gpu_rows:
        index, uuid, free = [x.strip() for x in row.split(",", 2)]
        gpus[int(index)] = {"uuid": uuid, "free_mib": int(free), "pids": []}
    uuid_to_gpu = {item["uuid"]: index for index, item in gpus.items()}
    for row in app_rows:
        if not row or "No running processes" in row:
            continue
        uuid, pid = [x.strip() for x in row.split(",", 1)]
        if uuid in uuid_to_gpu:
            gpus[uuid_to_gpu[uuid]]["pids"].append(int(pid))
    managed_pids = {
        int(rec.get("pid"))
        for trial in state["trials"] for rec in (trial["phase1"], trial["phase2"])
        if rec.get("status") == "running" and rec.get("pid")
    }
    report = {"disk_free_gib": free_gib, "gpus": {str(k): v for k, v in gpus.items()}}
    for gpu in args.gpus:
        if gpu not in gpus:
            raise RuntimeError(f"requested cuda:{gpu} is not present")
        item = gpus[gpu]
        if item["free_mib"] < args.min_gpu_free_mib:
            raise RuntimeError(
                f"cuda:{gpu} has {item['free_mib']} MiB free; need "
                f"--min-gpu-free-mib {args.min_gpu_free_mib}"
            )
        foreign = [pid for pid in item["pids"] if pid not in managed_pids]
        if foreign and not args.allow_busy_gpus:
            raise RuntimeError(
                f"cuda:{gpu} is in use by PID(s) {foreign}; pass --allow-busy-gpus "
                "only after verifying co-location is safe"
            )
    return report


def promote(state: dict[str, Any], args: argparse.Namespace) -> None:
    """Mark the lowest Phase-1 validation loss per parameter budget for Phase 2."""
    for target in args.targets:
        family = [t for t in state["trials"] if t["target_params"] == target]
        completed = [t for t in family if t["phase1"]["status"] == "done"]
        if any(t["phase1"]["status"] in ("pending", "running") for t in family):
            continue
        viable = [t for t in completed if (
            args.p1_max_best_val is None or
            float(t["phase1"]["summary"].get("best_val", float("inf"))) <= args.p1_max_best_val
        )]
        if not viable:
            for t in family:
                if t["phase2"]["status"] == "pending":
                    t["phase2"].update({"status": "not_promoted", "reason": "no Phase-1 trial passed gate"})
            continue
        winner = min(viable, key=lambda t: float(t["phase1"]["summary"].get("best_val", float("inf"))))
        for t in family:
            if t is winner and t["phase2"]["status"] == "pending":
                t["phase2"].update({"status": "ready", "promotion": "best_phase1_val"})
            elif t is not winner and t["phase2"]["status"] == "pending":
                t["phase2"].update({"status": "not_promoted", "reason": "lost Phase-1 comparison"})


def status_line(trial: dict[str, Any]) -> str:
    arch = trial["architecture"]
    return (f"{trial['id']}: {arch['params'] / 1e6:.2f}M "
            f"(d={arch['d_model']}, L={arch['dec_layers']}, ctx={arch['d_ctx']}x{arch['ctx_layers']}) "
            f"p1={trial['phase1']['status']} p2={trial['phase2']['status']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=Path("runs/architecture_search"))
    p.add_argument("--gpus", default="0,1", help="comma-separated CUDA indices; one training process per listed GPU")
    p.add_argument("--targets", default="5m,10m", help="comma-separated parameter targets, e.g. 5m,10m")
    p.add_argument("--trials-per-target", type=int, default=1)
    p.add_argument("--budget-tolerance", type=float, default=0.15, help="per-trial parameter deviation allowed from target")
    p.add_argument("--p1-steps", type=int, default=20_000)
    p.add_argument("--p2-steps", type=int, default=60_000)
    p.add_argument("--p1-batch", type=int, default=1024)
    p.add_argument("--p2-batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=48, help="Phase-1 scenario workers per run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p1-max-best-val", type=float, default=None, help="optional Phase-1 loss gate before promotion")
    p.add_argument("--p1-extra", default="", help="extra quoted flags for train_phase1_apex.py")
    p.add_argument("--p2-extra", default="", help="extra quoted flags forwarded to the vetted Phase-2 command")
    p.add_argument(
        "--phase2-template", default=None,
        help="required for --execute; vetted shell-like command with {device}, {out}, {init}, "
             "{resume}, {seed}, {steps}, {batch}, {arch_flags}, {extra}, and {python} placeholders",
    )
    p.add_argument(
        "--phase2-wrapper", default=None,
        help="required alternative to --phase2-template; executable wrapper owning the production physics recipe",
    )
    p.add_argument("--min-free-disk-gib", type=float, default=50.0)
    p.add_argument("--min-gpu-free-mib", type=int, default=0)
    p.add_argument("--allow-busy-gpus", action="store_true")
    p.add_argument("--execute", action="store_true", help="actually launch training; default only prints the schedule")
    p.add_argument("--retry-interrupted", action="store_true", help="retry Phase-1 interrupted jobs; Phase-2 resumes last.pt")
    p.add_argument("--poll-seconds", type=float, default=10.0)
    args = p.parse_args(argv)
    args.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    args.targets = [parse_budget(x) for x in args.targets.split(",") if x.strip()]
    if not args.gpus or not args.targets or args.trials_per_target < 1 or not 0 <= args.budget_tolerance < 1:
        p.error("need GPUs, targets, trials-per-target >= 1, and 0 <= budget-tolerance < 1")
    if args.phase2_template and args.phase2_wrapper:
        p.error("choose only one of --phase2-template or --phase2-wrapper")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    state_path = out_root / "search_state.json"
    state = load_or_create_state(args, state_path)
    for rank, trial in enumerate(state["trials"]):
        trial["rank"] = rank
        p1_dir, p2_dir = out_root / trial["id"] / "phase1", out_root / trial["id"] / "phase2"
        reconcile_phase(trial, p1_dir, 1)
        reconcile_phase(trial, p2_dir, 2)
        if args.retry_interrupted:
            for phase in (1, 2):
                if trial[f"phase{phase}"]["status"] == "interrupted":
                    trial[f"phase{phase}"] = {"status": "pending" if phase == 1 else "ready"}
    promote(state, args)
    if not args.execute:
        print("DRY RUN: no files or subprocesses will be created.")
        for trial in state["trials"]:
            print(status_line(trial))
            p1_dir, p2_dir = out_root / trial["id"] / "phase1", out_root / trial["id"] / "phase2"
            print("  P1", shlex.join(p1_command(args, trial, args.gpus[0], p1_dir)))
            if args.phase2_template or args.phase2_wrapper:
                print("  P2", shlex.join(p2_command(args, trial, args.gpus[0], p1_dir, p2_dir)))
            else:
                print("  P2 REQUIRES --phase2-template or --phase2-wrapper for execution")
        return 0

    if not (args.phase2_template or args.phase2_wrapper):
        raise RuntimeError(
            "refusing --execute without --phase2-template or --phase2-wrapper; "
            "the scheduler will not invent a Phase-2 physics/cost recipe"
        )
    state["preflight"] = preflight(args, out_root, state)
    atomic_json(state_path, state)

    running: dict[int, tuple[subprocess.Popen[bytes], dict[str, Any], int, Path]] = {}
    while True:
        for gpu, (proc, trial, phase, run_dir) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[gpu]
            record = trial[f"phase{phase}"]
            record["returncode"] = code
            record["finished_at"] = time.time()
            summary = phase_summary(run_dir)
            if code == 0 and summary and summary.get("event") == "done":
                validation = checkpoint_validation(run_dir, phase, trial["architecture"])
                record.update({"status": "done" if validation["ok"] else "failed", "summary": summary,
                               "checkpoint_validation": validation})
                if not validation["ok"]:
                    record["note"] = "subprocess succeeded but checkpoints failed validation"
            else:
                record.update({"status": "failed", "note": "see log"})
            atomic_json(state_path, state)

        # A previous invocation may have launched a detached process.  Reserve
        # its recorded GPU and keep polling its run directory; this is what
        # makes stopping/restarting the scheduler safe without double-booking
        # a GPU or relaunching an in-flight trial.
        foreign_gpus: set[int] = set()
        for trial in state["trials"]:
            for phase in (1, 2):
                rec = trial[f"phase{phase}"]
                if rec.get("status") != "running":
                    continue
                if rec.get("gpu") in running:
                    continue
                run_dir = out_root / trial["id"] / f"phase{phase}"
                reconcile_phase(trial, run_dir, phase)
                rec = trial[f"phase{phase}"]
                if rec.get("status") == "running":
                    foreign_gpus.add(int(rec["gpu"]))
        atomic_json(state_path, state)

        promote(state, args)
        for gpu in args.gpus:
            if gpu in running or gpu in foreign_gpus:
                continue
            selected: tuple[dict[str, Any], int] | None = None
            # Finish the P1 screen before allowing P2, making promotion a real
            # staged decision rather than a race between unrelated GPUs.
            pending_p1 = [t for t in state["trials"] if t["phase1"]["status"] == "pending"]
            if pending_p1:
                selected = (pending_p1[0], 1)
            else:
                ready_p2 = [t for t in state["trials"] if t["phase2"]["status"] == "ready"]
                if ready_p2:
                    selected = (ready_p2[0], 2)
            if selected is None:
                continue
            trial, phase = selected
            p1_dir, p2_dir = out_root / trial["id"] / "phase1", out_root / trial["id"] / "phase2"
            command = p1_command(args, trial, gpu, p1_dir) if phase == 1 else p2_command(args, trial, gpu, p1_dir, p2_dir)
            run_dir = p1_dir if phase == 1 else p2_dir
            proc = launch(command, run_dir / "launcher.log")
            trial[f"phase{phase}"].update({
                "status": "running", "pid": proc.pid, "gpu": gpu,
                "started_at": time.time(), "pid_start_time": pid_start_time(proc.pid), "command": command,
                "log": str(run_dir / "launcher.log"),
            })
            write_manifest(out_root, trial)
            atomic_json(state_path, state)
            running[gpu] = (proc, trial, phase, run_dir)
            print(f"[launch gpu {gpu}] phase {phase} {trial['id']}", flush=True)

        active_or_pending = bool(running) or bool(foreign_gpus) or any(
            t["phase1"]["status"] == "pending" or t["phase2"]["status"] == "ready"
            for t in state["trials"]
        )
        if not active_or_pending:
            break
        time.sleep(args.poll_seconds)

    atomic_json(state_path, state)
    print(f"[done] state: {state_path}")
    for trial in state["trials"]:
        print(status_line(trial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
