#!/usr/bin/env python3
"""LoRA PPO in the analytic closed-loop probe environment.

The actor executes only waypoint 0.  PPO's trajectory gradient is projected
onto the nullspace of the checkpoint's full augmented-Lagrangian trajectory
cost before it reaches decoder LoRA; the unexecuted tail receives cost only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig, mass_matrix  # noqa: E402
from amortized_opt.lead import resolve_lead  # noqa: E402
from amortized_opt.probe import closed_loop_probe  # noqa: E402
from amortized_opt.phase2_data import Phase2Pool  # noqa: E402
from amortized_opt.rhythm import DrumKit, Hit, RhythmClip  # noqa: E402
from amortized_opt.train_cost import _TrajectorySuffixMPCParentRing  # noqa: E402
from isaac_rl_ppo.actor import AmortizedPpoActor  # noqa: E402
from isaac_rl_ppo.cfg import PpoRunConfig  # noqa: E402
from isaac_rl_ppo.mpc import IsaacMPC, MPCConfig  # noqa: E402
from isaac_rl_ppo.ppo_loop import (  # noqa: E402
    PpoStorage, PpoUpdater, RolloutDriver, compute_gae,
)
from isaac_rl_ppo.reward import (  # noqa: E402
    ExecutedReward, ExecutedStrikeScorer, RewardConfig,
)
from isaac_rl_ppo.schedule import BatchedSchedule, ScheduleConfig  # noqa: E402
from isaac_rl_ppo.start import (  # noqa: E402
    ensure_clip_gap, install_duration_done, pad_clip_pool,
)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--steps-per-env", type=int, default=32)
    p.add_argument("--clips", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument(
        "--yaw-head-only", action="store_true",
        help="place LoRA only on head_traj and mask its delta to waist yaw",
    )
    p.add_argument("--noise-std", type=float, default=0.10)
    p.add_argument(
        "--white-noise", action="store_true",
        help="disable AR(1) action correlation and sample independent noise per tick",
    )
    p.add_argument(
        "--perfect-tracking", action="store_true",
        help="replace the analytic PD plant with exact command tracking",
    )
    p.add_argument(
        "--perfect-tracking-noise-std", type=float, default=0.0,
        help="independent executed-position noise in radians on each perfect-tracking tick",
    )
    p.add_argument("--freeze-wp", type=int, default=5)
    p.add_argument("--first-hit-s", type=float, default=1.0)
    p.add_argument("--clip-gap-s", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1.0e-5)
    p.add_argument("--lr-min", type=float, default=1.0e-7)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument(
        "--waist-consistency-weight", type=float, default=0.0,
        help="Huber weight tying replanned waist yaw to the first plan",
    )
    p.add_argument(
        "--waist-consistency-beta", type=float, default=0.02,
        help="Huber transition in radians for first-plan waist consistency",
    )
    p.add_argument(
        "--actor-loss-weight", type=float, default=1.0,
        help="scale PPO policy/entropy gradients; 0 makes LoRA auxiliary-only",
    )
    p.add_argument("--traj-grad-scale", type=float, default=0.2)
    p.add_argument("--orth-leak", type=float, default=0.0)
    p.add_argument(
        "--no-pcgrad", action="store_true",
        help="disable gradient projection and optimize PPO + weighted aux directly",
    )
    p.add_argument("--trust-rel-tol", type=float, default=0.01)
    p.add_argument("--trust-backtracks", type=int, default=4)
    p.add_argument("--reward-hit-weight", type=float, default=10.0)
    p.add_argument("--reward-hit-scale-m", type=float, default=0.05)
    # Accepted for replaying the discarded five-tick shaping ablation; the
    # production reward evaluates geometry only when the scheduled hit fires.
    p.add_argument("--reward-hit-window", type=int, default=1,
                   help=argparse.SUPPRESS)
    p.add_argument("--reward-hit-time-scale-s", type=float, default=0.02,
                   help=argparse.SUPPRESS)
    p.add_argument("--reward-speed-weight", type=float, default=0.5)
    p.add_argument("--reward-speed-scale-mps", type=float, default=0.5)
    p.add_argument("--reward-collision-weight", type=float, default=1.0e-3)
    p.add_argument(
        "--waist-terminal-reward-weight", type=float, default=0.0,
        help="sparse Cauchy reward at the first plan's original endpoint",
    )
    p.add_argument(
        "--waist-terminal-reward-scale-rad", type=float, default=0.10,
        help="Cauchy scale for executed yaw error at the cached endpoint",
    )
    p.add_argument("--critic-only", action="store_true")
    p.add_argument(
        "--reset-optimizer", action="store_true",
        help="load actor/critic weights from --resume but start a fresh optimizer",
    )
    p.add_argument(
        "--reset-critic", action="store_true",
        help="load policy weights from --resume but reinitialize the value critic",
    )
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument(
        "--best-metric",
        choices=("probe_score", "waist_plan_anchor_mae", "waist_terminal_error"),
        default="probe_score",
        help="checkpoint-selection metric (waist consistency is minimized)",
    )
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--pool-cache",
        default="runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    if args.perfect_tracking_noise_std < 0.0:
        raise ValueError("--perfect-tracking-noise-std must be non-negative")
    if args.perfect_tracking_noise_std and not args.perfect_tracking:
        raise ValueError(
            "--perfect-tracking-noise-std requires --perfect-tracking"
        )
    if args.waist_consistency_weight < 0.0 or args.waist_consistency_beta <= 0.0:
        raise ValueError("waist consistency weight must be >= 0 and beta > 0")
    if args.actor_loss_weight < 0.0:
        raise ValueError("--actor-loss-weight must be non-negative")
    if args.waist_terminal_reward_weight < 0.0:
        raise ValueError("--waist-terminal-reward-weight must be non-negative")
    if args.waist_terminal_reward_scale_rad <= 0.0:
        raise ValueError("--waist-terminal-reward-scale-rad must be positive")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.float32

    resume_blob = (
        torch.load(args.resume, map_location=dev, weights_only=False)
        if args.resume else None
    )
    resume_start = (
        int(resume_blob.get("iteration", -1)) + 1
        if resume_blob is not None else 0
    )
    if resume_start:
        torch.manual_seed(args.seed + resume_start * 1_000_003)
        np.random.seed(args.seed + resume_start * 1_000_003)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(model_cfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    scenario_cfg = ck.get("scenario_cfg") or {}
    train_cfg = ck.get("train_cfg") or {}
    H = int(scenario_cfg.get("horizon", 50))

    cfg = PpoRunConfig(name=Path(args.out).name)
    cfg.env.plant_regime = "analytic"
    cfg.env.device = str(dev)
    cfg.env.num_envs = int(args.envs)
    cfg.env.seed = int(args.seed)
    cfg.env.rhythm_split = "eval"
    cfg.env.freeze_wp = int(args.freeze_wp)
    cfg.env.first_hit_s = float(args.first_hit_s)
    cfg.env.clip_gap_s = float(args.clip_gap_s)
    cfg.actor.ckpt_path = str(args.ckpt)
    cfg.actor.horizon = H
    cfg.actor.lora_rank = int(args.lora_rank)
    if args.yaw_head_only:
        yaw_i = A.G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
        cfg.actor.lora_targets = ("head_traj",)
        cfg.actor.lora_output_rows = (yaw_i,)
    cfg.actor.noise_std_init = float(args.noise_std)
    cfg.actor.ar1_enabled = not bool(args.white_noise)
    cfg.actor.mean_correction = False
    cfg.actor.freeze_wp = int(args.freeze_wp)
    cfg.ppo.learning_rate = float(args.lr)
    cfg.ppo.lr_min = float(args.lr_min)
    cfg.ppo.schedule = "fixed"
    cfg.ppo.num_steps_per_env = int(args.steps_per_env)
    cfg.ppo.micro_batch_size = min(128, args.envs * args.steps_per_env)
    if float(args.actor_loss_weight) == 0.0:
        # LoRA moves only under the differentiable consistency/physics loss.
        # Its changing Gaussian mean can still make the diagnostic PPO KL
        # large; that must not prematurely stop auxiliary minibatches.
        cfg.ppo.kl_early_stop = False
    cfg.aux.enabled = True
    cfg.aux.in_isaac = "on"
    cfg.aux.weight = float(args.aux_weight)
    cfg.orth.enabled = not bool(args.no_pcgrad)
    cfg.orth.traj_grad_scale = float(args.traj_grad_scale)
    cfg.orth.wp0_leak = float(args.orth_leak)
    cfg.orth.trust_region = True
    cfg.orth.trust_rel_tol = float(args.trust_rel_tol)
    cfg.orth.trust_backtracks = int(args.trust_backtracks)
    cfg.max_iterations = int(args.iterations)
    cfg.save_interval = int(args.save_every)
    cfg.validate()

    actor = AmortizedPpoActor(
        model, cfg.actor, cfg.critic, num_future=model_cfg.num_future, device=dev,
    )
    print(f"[actor] {actor.param_counts()} LoRA paths={len(actor.lora_paths)}", flush=True)

    kit = A.DrumKit.from_layout()
    probe_clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    schedule = BatchedSchedule(
        args.envs,
        ScheduleConfig(
            horizon=H, num_drums=model_cfg.num_drums,
            num_future=model_cfg.num_future, hit_lead_s=0.0,
        ),
        device=dev, dtype=dtype,
    )
    install_duration_done(schedule, enabled=True)
    rng = np.random.default_rng(args.seed)
    all_ids = torch.arange(args.envs, device=dev)
    checkpoint_lead = train_cfg.get("cmd_lead_s")
    if args.perfect_tracking:
        lead_value = 0.0
        lead_source = "perfect tracking (lead disabled)"
    elif checkpoint_lead is not None:
        lead_value = resolve_lead(checkpoint_lead)
        lead_source = "checkpoint train_cfg.cmd_lead_s"
    else:
        lead_value, lead_source = cfg.resolve_lead()
        lead_value = resolve_lead(lead_value)
    print(f"[lead] {lead_source}", flush=True)
    mpc = IsaacMPC(
        model, schedule,
        MPCConfig(
            horizon=H, iterations=1, cmd_lead_s=lead_value,
            freeze_wp=args.freeze_wp, freeze_waist=True, amp_dtype=None,
        ),
        device=dev, dtype=dtype, kit_names=tuple(kit.names),
    )

    rnea = A.BatchedRNEA.from_urdf(device=dev).to(dev, dtype)
    sim = BatchedPDSim(
        PDSimConfig(
            dt=A.CONTROL_DT,
            # RL exploration can issue commands outside the pretrained
            # neighborhood.  The unconstrained analytic servo can then demand
            # impossible torque and numerically explode; the deployed plant
            # saturates at the actuator rating, so the RL environment must too.
            saturate_effort=True,
        ),
        device=dev, dtype=dtype, rnea=rnea,
    )
    gain_gen = torch.Generator(device=dev); gain_gen.manual_seed(args.seed + 17)
    kp, kd, ki = sim.sample_gains(args.envs, generator=gain_gen)
    tracking_gen = torch.Generator(device=dev)
    tracking_gen.manual_seed(args.seed + 31_337)

    if args.perfect_tracking:
        def plant(prev_cmd, cmd, q, qd, integral=None):
            del prev_cmd, qd
            noise = float(args.perfect_tracking_noise_std) * torch.randn(
                cmd.shape, device=cmd.device, dtype=cmd.dtype,
                generator=tracking_gen,
            )
            q_next = cmd + noise
            return q_next, (q_next - q) / float(A.CONTROL_DT), integral
    else:
        def plant(prev_cmd, cmd, q, qd, integral=None):
            minv = torch.linalg.inv(mass_matrix(rnea, q)).detach()
            return sim.step(
                prev_cmd, cmd, q, qd, kp=kp, kd=kd, ki=ki, Minv=minv,
                integral=integral,
            )

    cost_cfg = A.CostConfig(**train_cfg["cost"])
    cost = A.FusedCost(
        cost_cfg, rnea=rnea, max_horizon=H, device=dev, dtype=dtype,
    )
    al = A.AugmentedLagrangian(A.ALConfig(**train_cfg["al"]))
    al.load_state_dict(ck["al"])
    cost_weights = al.cost_weights()
    scorer = ExecutedStrikeScorer(
        fk=A.ChainFK(device=dev, dtype=dtype), dt=A.CONTROL_DT,
        device=dev, dtype=dtype,
    )
    reward_cfg = RewardConfig()
    reward_cfg.w_hit_error = float(args.reward_hit_weight)
    reward_cfg.hit_error_scale_m = float(args.reward_hit_scale_m)
    reward_cfg.w_impact_speed = float(args.reward_speed_weight)
    reward_cfg.impact_speed_scale_mps = float(args.reward_speed_scale_mps)
    reward_cfg.w_collision = float(args.reward_collision_weight)
    reward_cfg.w_plan_cost = 0.0
    reward = ExecutedReward(
        reward_cfg, scorer=scorer, cost=cost, device=dev, dtype=dtype,
    )
    updater = PpoUpdater(
        actor, cfg, device=dev, cost=cost, cost_weights=cost_weights,
        critic_only=bool(args.critic_only),
        waist_consistency_weight=float(args.waist_consistency_weight),
        waist_consistency_beta=float(args.waist_consistency_beta),
        actor_loss_weight=float(args.actor_loss_weight),
    )

    # ---- critic-certified fresh collocations ------------------------------
    z = np.load(args.pool_cache, allow_pickle=True)
    buckets = {}
    for key in z.files:
        if key.startswith("_"):
            continue
        bucket_name, name = key.split("/", 1)
        buckets.setdefault(int(bucket_name), {})[name] = z[key]
    pool_cfg = SimpleNamespace(
        dt=float(scenario_cfg.get("dt", A.CONTROL_DT)),
        joint_names=tuple(scenario_cfg.get("joint_names", A.G1_UPPER_BODY_JOINTS)),
    )
    phase2_pool = Phase2Pool(buckets, cfg=pool_cfg, device=dev, dtype=dtype)
    phase2_pool.n_shaft = cost.n_points - 2
    ring = _TrajectorySuffixMPCParentRing(
        device=dev, pool=phase2_pool, costs={H: cost},
        q_noise=float(train_cfg.get("timing_mpc_q_noise", 0.01)),
        qd_noise=float(train_cfg.get("timing_mpc_qd_noise", 0.05)),
        capacity=int(train_cfg.get("timing_mpc_parent_capacity", 2048)),
        terminal_peak_frac=float(train_cfg.get("terminal_suffix_peak_frac", 0.55)),
        terminal_long_frac=float(train_cfg.get("terminal_suffix_long_frac", 0.42)),
        terminal_low_frac=float(train_cfg.get("terminal_suffix_low_frac", 0.03)),
        terminal_peak_wp=int(train_cfg.get("terminal_suffix_peak_wp", 5)),
        terminal_tail_tau=float(train_cfg.get("terminal_suffix_tail_tau", 8.0)),
    )
    ring_state = (ck.get("extra") or {}).get("timing_critic", {}).get(
        "primary_mpc_ring"
    )
    if not ring_state:
        raise RuntimeError("checkpoint has no critic primary_mpc_ring")
    ring.load_state_dict(ring_state)
    if H not in ring.buckets:
        raise RuntimeError(f"critic parent ring has buckets {ring.buckets}, needs {H}")
    colloc_gen = torch.Generator(device=dev)
    colloc_gen.manual_seed(args.seed + 2901 + resume_start * 1_000_003)
    nominal_kit = kit
    reset_counter = 0

    def draw_collocations(n):
        idx = ring.sample_idx(H, int(n), generator=colloc_gen)
        return ring.batch(H, idx, generator=colloc_gen)

    def install_collocations(ids, q_now=None, qd_now=None):
        nonlocal reset_counter
        ids = ids.reshape(-1)
        batch = draw_collocations(ids.numel())
        context = batch["context"]
        hit_index = batch["hit"].long()
        active = batch["active"].bool()
        clips, kits, slot_rows = [], [], []
        for local in range(ids.numel()):
            valid = context.drum_valid[local].bool().cpu().numpy()
            drum_count = int(valid.sum())
            row_kit = DrumKit(
                names=list(nominal_kit.names[:drum_count]),
                positions=context.drum_pos[local, :drum_count].cpu().numpy(),
                axes=context.drum_axis[local, :drum_count].cpu().numpy(),
                radii=context.drum_radius[local, :drum_count].cpu().numpy(),
                heights=context.drum_height[local, :drum_count].cpu().numpy(),
                shaft_hit=nominal_kit.shaft_hit[:drum_count].copy(),
                edge_hit=nominal_kit.edge_hit[:drum_count].copy(),
            )
            hits, slots = [], {0: [], 1: []}
            for hand_i, hand in enumerate(("left", "right")):
                if bool(active[local, hand_i]):
                    drum = int(batch["strike"].target_drum[local, hand_i])
                    when = float(hit_index[local, hand_i]) * A.CONTROL_DT
                    hits.append(Hit(when, when, drum, row_kit.names[drum], hand, 0.5))
                    slots[hand_i].append((
                        context.goal_pos[local, hand_i].detach(),
                        context.goal_vel[local, hand_i].detach(),
                    ))
                for future_i in range(context.future_valid.shape[2]):
                    if not bool(context.future_valid[local, hand_i, future_i]):
                        continue
                    drum = int(context.future_drum[local, hand_i, future_i])
                    base_t = (
                        float(hit_index[local, hand_i]) * A.CONTROL_DT
                        if bool(active[local, hand_i]) else 0.0
                    )
                    when = base_t + float(context.future_gap[local, hand_i, future_i])
                    hits.append(Hit(when, when, drum, row_kit.names[drum], hand, 0.5))
                    slots[hand_i].append((
                        context.future_pos[local, hand_i, future_i].detach(),
                        context.future_vel[local, hand_i, future_i].detach(),
                    ))
            hits.sort(key=lambda value: value.time)
            duration = max([value.time for value in hits], default=0.1) + cfg.env.clip_gap_s
            clips.append(RhythmClip(f"critic_collocation_{reset_counter + local}", hits, duration))
            kits.append(row_kit)
            slot_rows.append(slots)
        reset_counter += int(ids.numel())
        schedule.reset(ids, clips, kits, rng=None)
        # Restore the exact certified/jittered aim points and velocities; the
        # synthetic RhythmClip carries only drum id/time/loudness.
        for local, env_id in enumerate(ids.tolist()):
            for hand_i in (0, 1):
                for slot_i, (position, velocity) in enumerate(slot_rows[local][hand_i]):
                    schedule.hit_pos[env_id, hand_i, slot_i].copy_(position)
                    schedule.hit_vel[env_id, hand_i, slot_i].copy_(velocity)
        q_seed = batch["q0"].to(dev, dtype)
        qd_seed = batch["v0"].to(dev, dtype)
        if q_now is None:
            q_new, qd_new = q_seed.clone(), qd_seed.clone()
        else:
            q_new, qd_new = q_now.clone(), qd_now.clone()
            q_new.index_copy_(0, ids, q_seed)
            qd_new.index_copy_(0, ids, qd_seed)
        mpc.reset(ids, q_new)
        mpc.init.index_copy_(0, ids, batch["init"].to(dev, dtype))
        return q_new, qd_new, batch

    q, qd, initial_collocation = install_collocations(all_ids)
    driver_ref = {}

    def on_done(ids, q_now, qd_now):
        ids = ids.reshape(-1)
        q_new, qd_new, _batch = install_collocations(ids, q_now, qd_now)
        driver = driver_ref.get("driver")
        if driver is not None and driver._integral is not None:
            driver._integral[ids] = 0.0
        return q_new, qd_new

    driver = RolloutDriver(
        actor=actor, mpc=mpc, schedule=schedule, reward=reward,
        plant=plant, cfg=cfg, fk=scorer.fk, on_done=on_done,
        waist_terminal_reward_weight=float(args.waist_terminal_reward_weight),
        waist_terminal_reward_scale_rad=float(args.waist_terminal_reward_scale_rad),
    )
    driver_ref["driver"] = driver

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tb").mkdir(exist_ok=True)
    saved_cfg = cfg.to_dict()
    saved_cfg["reward"] = asdict(reward_cfg)
    saved_cfg["perfect_tracking"] = bool(args.perfect_tracking)
    saved_cfg["perfect_tracking_noise_std"] = float(
        args.perfect_tracking_noise_std
    )
    saved_cfg["waist_consistency_weight"] = float(args.waist_consistency_weight)
    saved_cfg["waist_consistency_beta"] = float(args.waist_consistency_beta)
    saved_cfg["actor_loss_weight"] = float(args.actor_loss_weight)
    saved_cfg["yaw_head_only"] = bool(args.yaw_head_only)
    saved_cfg["waist_terminal_reward_weight"] = float(
        args.waist_terminal_reward_weight
    )
    saved_cfg["waist_terminal_reward_scale_rad"] = float(
        args.waist_terminal_reward_scale_rad
    )
    (out / "config.json").write_text(json.dumps(saved_cfg, indent=2, default=str))
    metrics = (out / "metrics.jsonl").open("a", buffering=1)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(out / "tb"))

    start_it = 0
    if resume_blob is not None:
        resume_state = resume_blob["actor"]
        if args.reset_critic:
            resume_state = {
                key: value for key, value in resume_state.items()
                if not key.startswith("critic.")
            }
        actor.load_state_dict(resume_state, strict=False)
        if not args.critic_only and not args.reset_optimizer and not args.reset_critic:
            updater.optimizer.load_state_dict(resume_blob["optimizer"])
            updater.learning_rate = float(resume_blob["learning_rate"])
        start_it = 0 if args.critic_only else resume_start

    def save(name, iteration):
        torch.save(
            {
                # Save the complete frozen backbone as well as LoRA.  The
                # current 4.7M architecture carries terminal-head buffers and
                # decoder structure that older PPO checkpoints assumed could
                # be reconstructed from a different champion checkpoint.
                "actor": actor.state_dict(),
                "optimizer": updater.optimizer.state_dict(),
                "iteration": int(iteration),
                "learning_rate": updater.learning_rate,
                "critic_only": bool(args.critic_only),
                "reward": asdict(reward_cfg),
                "perfect_tracking": bool(args.perfect_tracking),
                "perfect_tracking_noise_std": float(
                    args.perfect_tracking_noise_std
                ),
                "waist_consistency_weight": float(args.waist_consistency_weight),
                "waist_consistency_beta": float(args.waist_consistency_beta),
                "actor_loss_weight": float(args.actor_loss_weight),
                "best_metric": str(args.best_metric),
                "yaw_head_only": bool(args.yaw_head_only),
                "waist_terminal_reward_weight": float(
                    args.waist_terminal_reward_weight
                ),
                "waist_terminal_reward_scale_rad": float(
                    args.waist_terminal_reward_scale_rad
                ),
                "cfg": cfg.to_dict(),
                "backbone_ckpt": str(Path(args.ckpt).resolve()),
            },
            out / name,
        )

    probe_scorer = A.TrajectoryScorer(
        device="cpu", track_torque=False,
        thresholds=A.AcceptanceThresholds(hit_margin_m=0.0),
    )

    def evaluate_probe():
        probe = closed_loop_probe(
            actor.model, kit=kit, scorer=probe_scorer, clips=probe_clips,
            device=dev, num_drums=model_cfg.num_drums,
            num_future=model_cfg.num_future, horizon=H, steps=250,
            seed=0, amp_dtype=torch.bfloat16, lead_s=0.0,
            cmd_lead_s=lead_value, freeze_wp=args.freeze_wp,
            bounce=True, record_drift=True,
            waist_upright_ease_wp=scenario_cfg.get("waist_upright_ease_wp"),
            perfect_tracking=bool(args.perfect_tracking),
            record_waist_trace=True,
        )
        trace = probe.get("waist_trace", [[]])[0]
        yaw_anchor_mae = float("nan")
        yaw_cmd_mae = float("nan")
        yaw_terminal_error = float("nan")
        if trace:
            first = trace[0]["waist_yaw"]
            h0 = max(trace[0]["hit"].values())
            anchor = np.asarray(first["generated_rad"][:h0], dtype=float)
            plan_err, cmd_err = [], []
            for age, row in enumerate(trace[: len(anchor)]):
                h = min(max(row["hit"].values()), len(anchor) - age)
                if h <= 0:
                    continue
                generated = np.asarray(
                    row["waist_yaw"]["generated_rad"][:h], dtype=float,
                )
                plan_err.extend(
                    np.abs(generated - anchor[age : age + h]).tolist()
                )
                cmd_err.append(abs(float(generated[0]) - float(anchor[age])))
            if plan_err:
                yaw_anchor_mae = float(np.mean(plan_err))
            if cmd_err:
                yaw_cmd_mae = float(np.mean(cmd_err))
            if len(trace) > len(anchor) and len(anchor):
                yaw_terminal_error = abs(
                    float(trace[len(anchor)]["waist_yaw"]["measured_rad"])
                    - float(anchor[-1])
                )
        probe.pop("_hits", None)
        return {
            "probe/score": probe["score"],
            "probe/in_disk": probe["in_disk"],
            "probe/crossed": probe["latency"]["crossed_in_disk"],
            "probe/axial_abs_p50_mm": probe["axial_abs_p50_mm"],
            "probe/waist_plan_anchor_mae_rad": yaw_anchor_mae,
            "probe/waist_cmd_anchor_mae_rad": yaw_cmd_mae,
            "probe/waist_terminal_error_rad": yaw_terminal_error,
        }

    metric_key = {
        "probe_score": "probe/score",
        "waist_plan_anchor_mae": "probe/waist_plan_anchor_mae_rad",
        "waist_terminal_error": "probe/waist_terminal_error_rad",
    }[args.best_metric]
    minimize_best = args.best_metric != "probe_score"
    best_probe = float("inf") if minimize_best else -float("inf")
    if start_it > 0 and (out / "metrics.jsonl").exists():
        for line in (out / "metrics.jsonl").read_text().splitlines():
            try:
                previous = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = previous.get(metric_key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                best_probe = (
                    min(best_probe, float(value))
                    if minimize_best else max(best_probe, float(value))
                )
    metrics_empty = (out / "metrics.jsonl").stat().st_size == 0
    if metrics_empty:
        initial = {"iteration": start_it - 1, **evaluate_probe()}
        metrics.write(json.dumps(initial, default=float) + "\n")
        for key, value in initial.items():
            if isinstance(value, (int, float)):
                tb.add_scalar(key, value, 0)
        best_probe = float(initial[metric_key])
        save("best.pt", -1)
        print(
            f"[   0] frozen probe={initial['probe/score']:.3f} "
            f"waist_mae={initial['probe/waist_plan_anchor_mae_rad']:.4f}",
            flush=True,
        )
    for it in range(start_it, args.iterations):
        t0 = time.perf_counter()
        storage = PpoStorage(device=dev, horizon=H, n_dof=actor.n)
        collect_start = time.perf_counter()
        q, qd, rollout = driver.collect(q, qd, args.steps_per_env, storage)
        collect_s = time.perf_counter() - collect_start
        with torch.no_grad():
            obs = driver.privileged_obs(
                q, qd, mpc.last_ref0, schedule.context(), schedule.hit_index()
            )
            last_values = actor.critic(obs)
        storage.returns, storage.advantages = compute_gae(
            torch.stack(storage.rewards), torch.stack(storage.values),
            torch.stack(storage.dones), last_values,
            gamma=cfg.ppo.gamma, lam=cfg.ppo.lam,
        )
        storage.finalize()
        update_start = time.perf_counter()
        stats = updater.update(storage)
        update_s = time.perf_counter() - update_start
        row = {
            "iteration": it, "wall_s": time.perf_counter() - t0,
            "speed/collect_s": collect_s,
            "speed/update_s": update_s,
            "speed/env_steps_per_s": (
                args.envs * args.steps_per_env / max(collect_s, 1.0e-9)
            ),
            "collocation/advance_wp": ring.last_advance_offset_mean,
            "collocation/advance_s": ring.last_advance_seconds_mean,
            "collocation/unique_parent_frac": ring.last_unique_parent_frac,
            **{f"train/{k}": v for k, v in stats.items()}, **rollout,
        }
        do_eval = (it + 1) % max(args.eval_every, 1) == 0 or it + 1 == args.iterations
        if do_eval:
            probe_start = time.perf_counter()
            row.update(evaluate_probe())
            row["speed/probe_s"] = time.perf_counter() - probe_start
            candidate = float(row[metric_key])
            improved = candidate < best_probe if minimize_best else candidate > best_probe
            if improved:
                best_probe = candidate
                save("best.pt", it)
        metrics.write(json.dumps(row, default=float) + "\n")
        for key, value in row.items():
            if isinstance(value, (int, float)):
                tb.add_scalar(key, value, it + 1)
        print(
            f"[{it + 1:4d}] R={row.get('reward/total', float('nan')):+.3f} "
            f"KL={stats['kl']:.4g} lr={stats['learning_rate']:.2e} "
            f"orth={stats['orth_cos']:+.3f}/{stats['orth_remain']:.3f} "
            f"value={stats['value']:.3f} "
            f"EV={stats.get('critic_explained_variance', float('nan')):+.3f} "
            f"probe={row.get('probe/score', float('nan')):.3f} "
            f"speed={row['speed/env_steps_per_s']:.0f} env-step/s "
            f"(collect {collect_s:.2f}s update {update_s:.2f}s)",
            flush=True,
        )
        if (it + 1) % max(args.save_every, 1) == 0:
            save("last.pt", it)

    save("last.pt", args.iterations - 1)
    tb.flush(); tb.close(); metrics.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
