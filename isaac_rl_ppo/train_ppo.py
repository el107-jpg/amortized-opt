#!/usr/bin/env python
"""Executed-prefix PPO fine-tune of the amortized optimizer, in Isaac.

    python isaac_rl_ppo/train_ppo.py --device cuda:1 --envs 64 --iterations 20 \
        --name smoke --headless

Isaac Lab's ``DirectRLEnv`` cannot be imported before ``AppLauncher`` has
started the simulation app, so every Isaac-touching import in this file is
deferred into :func:`main` -- the same discipline the sibling's ``smoke_rl.py``
follows.  Everything above ``main`` is importable headless, which is what lets
``--dry-run`` check a configuration without a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_rl_ppo.cfg import PpoRunConfig, resolve_checkpoint  # noqa: E402


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="ppo_v1")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--steps-per-env", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=None,
                   help="pretrained backbone; default resolves phase2d(accepted)"
                        " -> phase2b, never phase2c")
    p.add_argument("--resume", default=None, help="an isaac_rl_ppo checkpoint")
    p.add_argument("--critic-ckpt", default=None,
                   help="load only the value critic from an isaac_rl_ppo checkpoint")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.45)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--aux-terms", default="default", choices=("default", "full"),
                   help="full uses every FusedCost term, including terminal and waist")
    p.add_argument("--aux-cost-source", default="phase2",
                   choices=("phase2", "checkpoint"),
                   help="checkpoint restores the backbone run's exact CostConfig")
    p.add_argument("--aux-gate", default="auto", choices=("auto", "on", "off"),
                   help="in-Isaac gate for the analytic FusedCost aux loss; "
                        "'auto' enables it when the pre-launch RNEA build "
                        "succeeded (see the Boost.Python note in README)")
    p.add_argument("--no-orth", action="store_true",
                   help="disable always-orth combine; fall back to the scalar "
                        "sum surrogate + aux.weight * cost")
    p.add_argument("--orth-scale", type=float, default=1.0,
                   help="mix on the projected PPO leftover at waypoint 0, "
                        "at PPO's native scale (default 1)")
    p.add_argument("--cost-to-rl", type=float, default=1.0,
                   help="scale the cost traj-grad so ||g_cost|| = this times "
                        "||g_rl0|| per sample (default 1; unscaled cost "
                        "railed KL then NaNed)")
    p.add_argument("--traj-grad-scale", type=float, default=0.2,
                   help="multiply the combined traj-grad after the cost/RL "
                        "match (default 0.2; 1.0 spent the 17-dim KL budget "
                        "on the first Adam step)")
    p.add_argument("--orth-leak", type=float, default=0.0,
                   help="fraction of the parallel PPO component at waypoint 0 "
                        "that leaks through the cost nullspace (default 0 = "
                        "hard orth; 0.1 is a small leak on the executed knot)")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="1e-5 by default; 3e-5 still spent ~0.07 nats/step "
                        "on the Isaac orth runs (target 0.015)")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="adaptive-KL floor (default 1e-6; rsl_rl's 1e-5 "
                        "pinned the Isaac orth runs while KL stayed ~0.07)")
    p.add_argument("--micro-batch", type=int, default=128)
    p.add_argument("--clips", type=int, default=4)
    p.add_argument("--rhythm-split", default="train")
    p.add_argument("--clip-index", type=int, default=None,
                   help="train every env on one clip from the selected rhythm split")
    p.add_argument("--reward-mode", default="continuous",
                   choices=("continuous", "binary_hits"),
                   help="binary_hits uses only correct versus wrong physical contacts")
    p.add_argument("--correct-hit-weight", type=float, default=10.0)
    p.add_argument("--wrong-hit-weight", type=float, default=5.0)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--no-amp", action="store_true", default=True,
                   help="kept for symmetry with smoke_rl; PPO needs fp32")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve config + checkpoint and exit (no Isaac)")
    p.add_argument(
        "--no-gravity-ff", action="store_true",
        help="disable the gravity/Coriolis feedforward. NOT the standard "
             "config: ff ON is standard since RL-1b's rnea_cache landed, and "
             "ff OFF causes arm sag that was once misread as lead "
             "over-compensation. Diagnostic use only.")
    p.add_argument(
        "--safety", default="off", choices=("off", "on"),
        help="QP safety filter. Default OFF, and that is a CONTROL not an "
             "oversight: with ff ON the filter engages on 26-34%% of ticks, "
             "which confounds any reading of the PPO return (it rewrites the "
             "executed command, so the stored log-prob no longer describes the "
             "action that ran). Run the off-control before interpreting "
             "returns from a filtered run.")
    p.add_argument("--lead", default=None,
                   help="path to a lead JSON (default runs/rl1b/lead_isaac.json)")
    p.add_argument("--lead-key", default=None, help="default best_strike_s")
    p.add_argument("--no-lead", action="store_true", help="run with no lead at all")
    p.add_argument(
        "--freeze-wp", type=int, default=0,
        help="waypoints of no-replan before a hit (default 0 = lead alone; "
             "5 is the 7c probe that worked best). Copies the striking arm "
             "(and waist) from the previous plan through contact.")
    p.add_argument(
        "--isaac-ready", action="store_true", default=False,
        help="Keep the Unitree USD init_state (play_isaac --isaac_ready). "
             "Default overwrites to amortized ready_joint_pos, the start "
             "play_isaac and phase-2 use.")
    p.add_argument(
        "--first-hit-s", type=float, default=1.0,
        help="Pad every clip so the first HAND hit is at this time (seconds). "
             "1.0 matches play_isaac --first_hit_s 1.0 / idle-family training. "
             "0 = native RCC times.")
    p.add_argument(
        "--clip-gap-s", type=float, default=1.0,
        help="Idle after the last hit before chaining the next clip, holding "
             "leftover q (default 1.0). 0 = chain as soon as the last hit fires.")
    return p.parse_args(argv)


def build_cfg(args) -> PpoRunConfig:
    cfg = PpoRunConfig(name=args.name)
    cfg.env.num_envs = int(args.envs)
    cfg.env.device = args.device
    cfg.env.seed = int(args.seed)
    cfg.env.rhythm_split = args.rhythm_split
    cfg.actor.ckpt_path = args.ckpt
    cfg.actor.lora_rank = int(args.lora_rank)
    cfg.actor.noise_std_init = float(args.noise_std)
    cfg.ppo.learning_rate = float(args.lr)
    cfg.ppo.lr_min = float(args.lr_min)
    cfg.ppo.num_steps_per_env = int(args.steps_per_env)
    cfg.ppo.micro_batch_size = int(args.micro_batch)
    cfg.aux.weight = float(args.aux_weight)
    if args.aux_terms == "full":
        cfg.aux.terms = None
    cfg.aux.in_isaac = args.aux_gate
    cfg.orth.enabled = not bool(args.no_orth)
    cfg.orth.rl_to_cost_norm = float(args.orth_scale)
    cfg.orth.cost_to_rl_norm = float(args.cost_to_rl)
    cfg.orth.traj_grad_scale = float(args.traj_grad_scale)
    cfg.orth.wp0_leak = float(args.orth_leak)
    cfg.max_iterations = int(args.iterations)
    cfg.save_interval = int(args.save_interval)
    cfg.resume = args.resume
    if args.no_gravity_ff:
        cfg.env.gravity_feedforward = False
    cfg.env.safety_filter = (args.safety == "on")
    if args.lead:
        cfg.env.lead_json = args.lead
    if args.lead_key:
        cfg.env.lead_key = args.lead_key
    if args.no_lead:
        cfg.env.cmd_lead_s = 0.0
    cfg.env.freeze_wp = int(args.freeze_wp)
    cfg.actor.freeze_wp = int(args.freeze_wp)
    cfg.env.isaac_ready = bool(args.isaac_ready)
    cfg.env.first_hit_s = float(args.first_hit_s)
    cfg.env.clip_gap_s = float(args.clip_gap_s)
    cfg.validate()
    return cfg


class JsonlLogger:
    """Append-only metrics, one JSON object per iteration."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", buffering=1)

    def write(self, row: dict) -> None:
        self.fh.write(json.dumps(row, default=float) + "\n")

    def close(self) -> None:
        self.fh.close()


def main(argv=None) -> int:
    args = parse(argv)
    cfg = build_cfg(args)
    run_dir = cfg.run_dir()

    choice = resolve_checkpoint(cfg.actor.ckpt_path, file_name=cfg.actor.ckpt_file)
    print(choice.describe(), flush=True)
    if choice.path is None and not cfg.actor.allow_untrained_backbone:
        print("[fatal] no pretrained backbone resolved; pass --ckpt or set "
              "allow_untrained_backbone", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "cfg": cfg.to_dict()},
                         indent=2, default=str))
        return 0

    # ---- PINOCCHIO FIRST, before Isaac ---------------------------------------
    # ``BatchedRNEA.from_urdf`` reads ``model.parents`` (a std::vector<Index>)
    # through Boost.Python.  Once the Isaac app is up, that conversion raises
    #
    #   TypeError: No Python class registered for C++ class
    #              std::vector<unsigned long, std::allocator<unsigned long> >
    #
    # because Isaac's runtime brings its own Boost.Python/eigenpy and the
    # converter registration for that vector type is shadowed.  It is NOT a
    # broken Pinocchio build: the identical call succeeds in the same conda env
    # in a process that has not launched Isaac.  Building the RNEA here and
    # moving it to the GPU afterwards sidesteps it entirely -- the constructed
    # object is plain torch tensors plus Python ints, so nothing it does later
    # touches Boost.
    from amortized_opt.rnea import BatchedRNEA
    try:
        _rnea_cpu = BatchedRNEA.from_urdf(device="cpu")
        print("[prelaunch] BatchedRNEA built before Isaac (Boost.Python "
              "registry workaround) -- analytic aux cost available", flush=True)
    except Exception as exc:                       # pragma: no cover
        _rnea_cpu = None
        print(f"[prelaunch] BatchedRNEA FAILED ({exc!r}); the analytic aux "
              f"cost will be gated OFF", flush=True)

    # ---- Isaac next, before anything imports isaaclab.envs -------------------
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=bool(args.headless), device=args.device).app

    import numpy as np
    import torch

    import amortized_opt as A

    from isaac_rl_ppo import bridge
    from isaac_rl_ppo.actor import AmortizedPpoActor
    from isaac_rl_ppo.ppo_loop import PpoStorage, PpoUpdater, RolloutDriver, compute_gae

    surface = bridge.load_isaac_rl()
    print(f"[bridge] isaac_rl source = {surface.source}", flush=True)

    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    dtype = torch.float32

    # ---- env ----------------------------------------------------------------
    make_env = surface.make_env
    DrumRLEnvCfg = surface.DrumRLEnvCfg
    env_cfg = DrumRLEnvCfg(
        num_envs=cfg.env.num_envs, device=args.device, seed=args.seed,
        rhythm_split=cfg.env.rhythm_split, contact_sensors=cfg.env.contact_sensors,
        gravity_feedforward=cfg.env.gravity_feedforward, foh_ramp=cfg.env.foh_ramp,
        pin_pelvis=cfg.env.pin_pelvis, freeze_legs=cfg.env.freeze_legs,
        use_ideal_pd=cfg.env.use_ideal_pd, pd_damping_scale=cfg.env.pd_damping_scale,
        hit_mode=cfg.env.hit_mode, cmd_lead_s=cfg.env.cmd_lead_s,
        freeze_wp=cfg.env.freeze_wp, hit_lead_s=cfg.env.hit_lead_s,
    )
    # ---- build the Isaac cfg, PIN THE PHYSX DEVICE, then instantiate ---------
    # ``SimulationCfg.device`` defaults to ``cuda:0``.  If the app is running on
    # a different GPU, PhysX blocks FOREVER inside ``get_dof_velocities`` --
    # without raising, without a timeout, and without a log line.  On this box
    # cuda:0 is held by another session, so the default is exactly the trap.
    # ``isaac_rl.env.make_env`` composes build+instantiate in one call and does
    # not expose the sim cfg in between, so the two halves are called directly.
    envmod = surface.modules["env"]
    t0 = time.perf_counter()
    isaac_cfg = envmod.build_isaac_cfg(env_cfg)
    isaac_cfg.sim.device = args.device
    if hasattr(envmod, "_make"):
        env = envmod._make(isaac_cfg, env_cfg)
    else:                                          # contract changed; be loud
        print("[warn] isaac_rl.env._make is gone; falling back to make_env, "
              "which leaves sim.device at its default -- see the PhysX device "
              "trap note in README", flush=True)
        env = make_env(env_cfg)
    print(f"[env] sim.device pinned to {isaac_cfg.sim.device}", flush=True)
    env.reset()
    print(f"[env] {cfg.env.num_envs} envs built in {time.perf_counter() - t0:.1f}s "
          f"step_dt={float(env.step_dt):.5f}", flush=True)

    from isaac_rl_ppo.start import (
        describe_clip_pad, ensure_clip_gap, install_duration_done,
        install_train_ready, make_leftover_on_done, pad_clip_pool,
    )
    if not cfg.env.isaac_ready:
        install_train_ready(env)
    else:
        print("[ready] --isaac-ready: keeping env composed pose", flush=True)

    # ---- the pretrained backbone -------------------------------------------
    ck = torch.load(str(choice.path), map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    H = int((ck.get("scenario_cfg") or {}).get("horizon", cfg.actor.horizon))
    cfg.actor.horizon = H

    actor = AmortizedPpoActor(model, cfg.actor, cfg.critic, num_future=mcfg.num_future,
                              device=dev)
    if args.critic_ckpt:
        critic_blob = torch.load(args.critic_ckpt, map_location=dev,
                                 weights_only=False)
        critic_state = critic_blob.get("actor", critic_blob)
        critic_state = {
            k.removeprefix("critic."): v for k, v in critic_state.items()
            if k.startswith("critic.")
        }
        if not critic_state:
            raise ValueError(f"{args.critic_ckpt} contains no critic parameters")
        actor.critic.load_state_dict(critic_state, strict=True)
        print(f"[critic] initialized from {args.critic_ckpt} "
              f"({len(critic_state)} tensors)", flush=True)
    counts = actor.param_counts()
    print(f"[actor] trainable {counts['trainable']:,} "
          f"(lora {counts['lora']:,} + head {counts['gaussian_head']:,} + "
          f"critic {counts['critic']:,}) | frozen {counts['frozen']:,}", flush=True)
    print(f"[actor] AR(1) rho: " + ", ".join(
        f"{n.replace('_joint', '')}={r:.3f}" for n, _t, r in actor.ar1_table[:4]),
        flush=True)

    # ---- schedule / mpc / reward -------------------------------------------
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir(cfg.env.rhythm_split), kit)
    if args.clip_index is not None:
        if not 0 <= int(args.clip_index) < len(clips):
            raise IndexError(f"clip index {args.clip_index} outside [0, {len(clips)})")
        clips = [clips[int(args.clip_index)]]
        print(f"[clip] fixed {cfg.env.rhythm_split} clip index {args.clip_index}: "
              f"{clips[0].name}", flush=True)
    else:
        clips = clips[: max(int(args.clips), 1)]
    clips, pad_stats = pad_clip_pool(clips, cfg.env.first_hit_s)
    clips = ensure_clip_gap(clips, cfg.env.clip_gap_s)
    print(describe_clip_pad(pad_stats), flush=True)
    print(
        f"[clip] gap after last hit {cfg.env.clip_gap_s:.2f}s "
        f"({'done waits for duration' if cfg.env.clip_gap_s > 0 else 'chain on last hit'})",
        flush=True,
    )
    ScheduleConfig, BatchedSchedule = surface.ScheduleConfig, surface.BatchedSchedule
    scfg = ScheduleConfig(horizon=H, num_drums=mcfg.num_drums,
                          num_future=mcfg.num_future, hit_lead_s=cfg.env.hit_lead_s)
    sched = BatchedSchedule(cfg.env.num_envs, scfg, device=dev, dtype=dtype)
    install_duration_done(sched, enabled=cfg.env.clip_gap_s > 0)
    rng = np.random.default_rng(args.seed)

    def _reset_sched(ids: torch.Tensor) -> None:
        picks = [clips[int(i) % len(clips)] for i in range(ids.numel())]
        sched.reset(ids, picks, [kit] * ids.numel(), rng=rng)

    all_ids = torch.arange(cfg.env.num_envs, device=dev)
    _reset_sched(all_ids)

    lead_value, lead_source = cfg.resolve_lead()
    print(f"[lead] {lead_source}", flush=True)
    if isinstance(lead_value, (tuple, list)):
        ms = [1000.0 * v for v in lead_value]
        print(f"[lead] ms: waist {ms[0]:.1f}/{ms[1]:.1f}/{ms[2]:.1f} | "
              f"L wrist {ms[7]:.1f}/{ms[8]:.1f}/{ms[9]:.1f} | "
              f"R wrist {ms[14]:.1f}/{ms[15]:.1f}/{ms[16]:.1f}", flush=True)

    IsaacMPC, MPCConfig = surface.IsaacMPC, surface.MPCConfig
    mpc = IsaacMPC(
        model, sched,
        MPCConfig(horizon=H, iterations=cfg.actor.iterations,
                  cmd_lead_s=lead_value, freeze_wp=cfg.env.freeze_wp,
                  freeze_waist=cfg.actor.freeze_waist,
                  amp_dtype=None),          # fp32: the update must reproduce the mean
        device=dev, dtype=dtype, kit_names=tuple(kit.names))
    print(f"[freeze] wp={cfg.env.freeze_wp} waist={cfg.actor.freeze_waist} "
          f"({'ON: copy striking arm from previous plan when tth<=freeze_wp' if cfg.env.freeze_wp else 'OFF'})",
          flush=True)
    print(f"[orth] leak={cfg.orth.wp0_leak:g} traj_grad_scale={cfg.orth.traj_grad_scale:g} "
          f"({'hard nullspace' if cfg.orth.wp0_leak == 0.0 else 'wp0 parallel leak ON'})",
          flush=True)

    from amortized_opt.train_cost import phase2_cost_config

    # A general-purpose ChainFK for the strike scorer and the critic's tip
    # block.  It is deliberately NOT handed to FusedCost: the cost needs the
    # SHOULDER frames for its posture term (FusedCost.FRAMES) and a default
    # ChainFK does not carry them, so sharing the instance raises
    # KeyError('left_shoulder') on the first reward call.  Letting FusedCost
    # build its own FK costs one extra frame set and is what the tests exercise.
    fk = A.ChainFK(device=dev, dtype=dtype)

    # ---- the analytic aux cost, GATED ---------------------------------------
    gate = cfg.aux.in_isaac
    want_cost = cfg.aux.enabled and gate != "off" and (
        _rnea_cpu is not None or gate == "on")
    cost = None
    if want_cost:
        # ``FusedCost`` takes no ``sdf`` kwarg (the sibling's smoke_rl.py passes
        # one, which would raise -- flagged in README, not fixed here:
        # isaac_rl is read-only to us).  phase2_cost_config() is the
        # TRAINING-calibrated weighting, not CostConfig()'s validation weights
        # whose group means differ by ~1e3.
        rnea = (_rnea_cpu.to(dev, dtype) if _rnea_cpu is not None
                else A.BatchedRNEA.from_urdf(device=dev).to(dev, dtype))
        if args.aux_cost_source == "checkpoint":
            saved_cost = (ck.get("train_cfg") or {}).get("cost")
            if not isinstance(saved_cost, dict):
                raise ValueError("--aux-cost-source checkpoint requested but "
                                 "checkpoint has no train_cfg.cost")
            cost_cfg = A.CostConfig(**saved_cost)
            print("[aux] restored exact CostConfig from backbone checkpoint",
                  flush=True)
        else:
            cost_cfg = phase2_cost_config()
        cost = A.FusedCost(cost_cfg, rnea=rnea,
                           max_horizon=H, device=dev, dtype=dtype)
    print(f"[aux] analytic cost gate={gate!r} -> "
          f"{'ON' if cost is not None else 'OFF'} "
          f"(weight {cfg.aux.weight}, terms {cfg.aux.terms})", flush=True)
    if cost is None:
        print("[aux] the plan tail is UNANCHORED this run: nothing prices "
              "waypoints 1..H-1, which are the next tick's warm start. Expect "
              "the plan to degrade where the reward cannot see it.", flush=True)

    RewardConfig = surface.RewardConfig
    rcfg = RewardConfig()
    rcfg.reward_mode = args.reward_mode
    rcfg.w_correct_hit = float(args.correct_hit_weight)
    rcfg.w_wrong_hit = float(args.wrong_hit_weight)
    rcfg.w_plan_cost = cfg.reward_plan_cost_weight
    print(f"[reward] mode={rcfg.reward_mode} correct={rcfg.w_correct_hit:g} "
          f"wrong={rcfg.w_wrong_hit:g}", flush=True)
    scorer = surface.ExecutedStrikeScorer(fk=fk, dt=float(env.step_dt), device=dev,
                                          dtype=dtype)
    reward = surface.ExecutedReward(rcfg, scorer=scorer, cost=cost, device=dev,
                                    dtype=dtype)

    safety = None
    if cfg.env.safety_filter:
        Filt, SConf = bridge.load_safety()
        if Filt is not None:
            safety = Filt(SConf(mode="qp"), device=dev, dtype=dtype)
    print(f"[safety] QP filter {'ON' if safety is not None else 'OFF (control)'}",
          flush=True)
    if safety is not None:
        print("[safety] the filter rewrites the executed command, so the stored "
              "log-prob no longer describes the action that ran; it engaged on "
              "26-34% of ticks in the sibling's ff-ON baseline. Interpret "
              "returns against a --safety off control.", flush=True)

    # ---- reset handling -----------------------------------------------------
    q, qd = env.read_state()
    mpc.reset(all_ids, q)
    on_done = make_leftover_on_done(env, _reset_sched, mpc)

    driver = RolloutDriver(actor=actor, mpc=mpc, schedule=sched, reward=reward,
                           plant=env.plant, cfg=cfg, fk=fk, on_done=on_done,
                           safety=safety)
    updater = PpoUpdater(actor, cfg, device=dev, cost=cost)

    # ---- resume -------------------------------------------------------------
    start_iter = 0
    if cfg.resume:
        blob = torch.load(cfg.resume, map_location=dev, weights_only=False)
        missing, unexpected = actor.load_state_dict(blob["actor"], strict=False)
        updater.optimizer.load_state_dict(blob["optimizer"])
        updater.learning_rate = float(blob.get("learning_rate", cfg.ppo.learning_rate))
        start_iter = int(blob.get("iteration", 0)) + 1
        print(f"[resume] {cfg.resume} @ iter {start_iter} "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    config_record = cfg.to_dict()
    config_record["launch_args"] = vars(args)
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2,
                                                    default=str))
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    from torch.utils.tensorboard import SummaryWriter
    tb_dir = run_dir / "tb"
    tb = SummaryWriter(str(tb_dir))
    registry = ROOT / "runs" / "tensorboard_active"
    registry.mkdir(parents=True, exist_ok=True)
    link = registry / run_dir.name
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == tb_dir.resolve():
            pass
        else:
            raise FileExistsError(f"TensorBoard registry entry already exists: {link}")
    else:
        link.symlink_to(tb_dir.resolve())
    print(f"[tensorboard] logdir={tb_dir} registered={link}", flush=True)

    def save(tag: str, it: int) -> None:
        torch.save({"actor": actor.policy_state_dict(),
                    "optimizer": updater.optimizer.state_dict(),
                    "iteration": it, "learning_rate": updater.learning_rate,
                    "cfg": cfg.to_dict(), "backbone_ckpt": str(choice.path)},
                   run_dir / f"{tag}.pt")

    # ---- the loop -----------------------------------------------------------
    best = -float("inf")
    for it in range(start_iter, cfg.max_iterations):
        t_start = time.perf_counter()
        storage = PpoStorage(device=dev, horizon=H, n_dof=actor.n)
        q, qd, roll_log = driver.collect(q, qd, cfg.ppo.num_steps_per_env, storage)
        t_collect = time.perf_counter() - t_start

        with torch.no_grad():
            last_obs = driver.privileged_obs(q, qd, mpc.last_ref0, sched.context(),
                                             sched.hit_index())
            last_values = actor.critic(last_obs)
        storage.returns, storage.advantages = compute_gae(
            torch.stack(storage.rewards, 0), torch.stack(storage.values, 0),
            torch.stack(storage.dones, 0), last_values,
            gamma=cfg.ppo.gamma, lam=cfg.ppo.lam)
        storage.finalize()

        t_up = time.perf_counter()
        stats = updater.update(storage)
        t_update = time.perf_counter() - t_up

        env_steps = cfg.ppo.num_steps_per_env * cfg.env.num_envs
        row = {"iteration": it, "time_collect_s": t_collect,
               "time_update_s": t_update, "env_steps": env_steps,
               "env_fps": env_steps / max(t_collect, 1e-9),
               **{f"train/{k}": v for k, v in stats.items()},
               **roll_log}
        logger.write(row)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                tb.add_scalar(key, value, it)
        tb.flush()
        if it % cfg.log_interval == 0:
            print(f"[{it:5d}] R {roll_log.get('reward/total', float('nan')):+8.3f} "
                  f"| kl {stats['kl']:.5f} lr {stats['learning_rate']:.2e} "
                  f"| surr {stats['surrogate']:+.4f} val {stats['value']:.4f} "
                  f"aux {stats['aux']:.3f} | sig {stats['noise_std_mean']:.3f} "
                  f"| orth {stats.get('orth_cos', 0):+.2f} "
                  f"| {env_steps / max(t_collect, 1e-9):6.0f} env/s "
                  f"(col {t_collect:.1f}s upd {t_update:.1f}s)", flush=True)
        if stats["lr_floor_hits"] > 0:
            print(f"[warn] learning rate railed at the {cfg.ppo.lr_min:g} floor "
                  f"{int(stats['lr_floor_hits'])}x -- the documented failure mode; "
                  f"check desired_kl / epochs", flush=True)

        r = roll_log.get("reward/total", -float("inf"))
        if r > best:
            best = r
            save("best", it)
        if cfg.save_interval and it % cfg.save_interval == 0:
            save("last", it)

    save("last", cfg.max_iterations - 1)
    logger.close()
    tb.close()
    # Isaac's ``app.close()`` HANGS in the carb tasking mutex (it serialised the
    # sibling's queued runs behind a 28-minute timeout), so teardown is a hard
    # exit after the artifacts are flushed -- the same thing their scripts do.
    # Everything worth keeping is already on disk at this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
