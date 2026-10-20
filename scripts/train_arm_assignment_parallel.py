#!/usr/bin/env python
"""Resume learned arm assignment with parallel Isaac and Double DQN.

The frozen clean-line generator and committed-plan PID execute each choice.
Default observation: eight beats of lookahead and 32 upcoming strikes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/arm_assignment_mlp_20261002/config.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--commit-lead-s", type=float, default=None,
                        help="Lock an arm choice this many seconds before its hit; independent of lookahead.")
    parser.add_argument("--pending-per-arm", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--window-sampling", choices=("uniform", "balanced"), default=None)
    parser.add_argument("--refresh-windows", action="store_true")
    parser.add_argument("--deduplicate-windows", action="store_true")
    parser.add_argument("--max-window-probability", type=float, default=None)
    parser.add_argument("--start-pose-pool", default=None,
                        help="Executed training pose anchors for randomized arm/waist starts.")
    parser.add_argument("--reset-replay", action="store_true",
                        help="Discard old replay while retaining learned Q, target and optimizer.")
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None,
                        help="Bellman updates per collection batch; default 1024 with late commitment, otherwise 8 per environment.")
    parser.add_argument("--lookahead-beats", type=float, default=8)
    parser.add_argument("--future-hits", type=int, default=32)
    parser.add_argument("--graph-check", action="store_true", help="Evaluate the graph in all envs before training.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-seconds", type=float, default=None)
    parser.add_argument("--check-output", default=None)
    parser.add_argument("--legacy-lookahead", action="store_true", help="Graph parity check with the previous two-second queue.")
    parser.add_argument("--benchmark-output", default=None,
                        help="Collect varied short training episodes and write feasibility metrics without updating weights.")
    parser.add_argument("--benchmark-window-s", type=float, default=1.0)
    parser.add_argument("--close-finetune", action="store_true",
                        help="Paired Isaac graph/Q evaluation and a short anchored Double DQN fine-tune.")
    parser.add_argument("--close-updates", type=int, default=5)
    parser.add_argument("--close-lr", type=float, default=1e-5)
    parser.add_argument("--close-anchor-weight", type=float, default=5.)
    parser.add_argument("--close-max-flips", type=float, default=.05)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.envs < 1 or (args.gradient_steps is not None and args.gradient_steps < 1) or args.lookahead_beats <= 0 or args.future_hits < 1:
        raise ValueError("env count, gradient steps, beats and future strikes must be positive")
    cfg = json.loads((ROOT/args.config).read_text())
    run_dir = Path(cfg["run_dir"])
    resume = Path(args.resume) if args.resume else next((run_dir/n for n in
                  ("dqn_last.pt", "assignment_initial.pt", "ppo_before_parallel_dqn.pt") if (run_dir/n).exists()),
                  run_dir/"ppo_before_parallel_dqn.pt")
    if args.run_name:
        run_dir = ROOT/"runs"/args.run_name
    if not resume.is_file():
        raise FileNotFoundError(resume)
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import isaaclab_tasks  # noqa: F401
        import g1_drummer.tasks  # noqa: F401
        from isaac_rl.decoder_scene import (bind_registered_task_geometry,
            configure_tip_geometry,configure_drum_kit_geometry)
        bind_registered_task_geometry()
        from g1_drummer.viz import enable_hit_telemetry
        from isaaclab_tasks.utils import parse_env_cfg
        import amortized_opt as A
        from amortized_opt.constants import ready_joint_pos, G1_UPPER_BODY_JOINTS
        from amortized_opt.probe import head_frame_error
        from isaac_rl.arm_assignment import AssignmentConfig
        from isaac_rl.assignment_dqn import AssignmentQPolicy
        from isaac_rl.assignment_parallel import ParallelCommittedMPC, ParallelPID
        from isaac_rl.assignment_parallel_train import collect, load_with_tempo, train
        from isaac_rl.env import try_build_rnea
        from isaac_rl.mpc import MPCConfig
        from isaac_rl.rnea_cache import rnea_from_cache_if_needed
        from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

        dev = torch.device(args.device)
        if dev.type != "cuda":
            raise ValueError("Isaac assignment collection requires CUDA")
        torch.cuda.set_device(dev)
        torch.manual_seed(cfg["seed"])
        torch.set_float32_matmul_precision("high")
        blob = torch.load(resume, map_location="cpu", weights_only=False)
        if blob.get("format") in ("arm_assignment_dqn_v1", "arm_assignment_dqn_init_v1", "arm_assignment_search_q_v1"):
            policy = AssignmentQPolicy(AssignmentConfig(**blob["policy_config"]))
            policy.load_state_dict(blob["policy"])
            if (policy.cfg.future_hits != args.future_hits or policy.cfg.lookahead_beats != args.lookahead_beats):
                raise ValueError("resume observation differs from requested lookahead")
        else:
            policy = AssignmentQPolicy.from_ppo(blob, future_hits=args.future_hits,
                                               lookahead_beats=args.lookahead_beats)
        learning_checkpoint = None if blob.get("format") == "arm_assignment_search_q_v1" else blob
        if args.commit_lead_s is not None:
            import math
            if args.commit_lead_s <= 0:
                raise ValueError("commitment lead must be positive")
            pending = args.pending_per_arm or int(math.ceil(args.commit_lead_s/A.CONTROL_DT))+1
            if pending < int(math.ceil(args.commit_lead_s/A.CONTROL_DT))+1:
                raise ValueError("pending capacity must cover every 20 ms hit in the commitment window")
            if (policy.cfg.commit_lead_s != args.commit_lead_s or policy.cfg.pending_per_arm != pending):
                policy = policy.with_queue_config(commit_lead_s=args.commit_lead_s, pending_per_arm=pending)
                learning_checkpoint = None
        if args.run_name and run_dir.exists() and resume.parent.resolve() != run_dir.resolve():
            raise FileExistsError("refusing to overwrite an existing comparison run")
        if args.gradient_steps is None:
            args.gradient_steps = 1024 if policy.cfg.commit_lead_s is not None else 8*args.envs
        policy.to(dev)
        cfg.update(algorithm="Double DQN", parallel_envs=args.envs, episodes_per_update=args.envs,
                   gradient_steps=args.gradient_steps,
                   run_dir=str(run_dir), name=run_dir.name,
                   replay_capacity=args.replay_capacity or (500000 if learning_checkpoint is None else
                                    blob.get("config", {}).get("replay_capacity", 50000)),
                   exploration_decay_decisions=250000 if policy.cfg.commit_lead_s is not None else 20000,
                   policy_config=asdict(policy.cfg), resumed_from=str(resume.resolve()),
                   device=str(dev), critic=True, bellman=True, drum_action_masks=False)
        if args.iterations is not None:
            cfg["iterations"] = args.iterations
        if args.window_sampling is not None:
            cfg["window_sampling"] = args.window_sampling
        if args.refresh_windows:
            cfg["refresh_balanced_windows"] = True
        if args.deduplicate_windows:
            cfg["deduplicate_windows"] = True
        if args.max_window_probability is not None:
            if not (0 < args.max_window_probability <= 1):
                raise ValueError("window probability cap must be in (0, 1]")
            cfg["max_window_probability"] = args.max_window_probability
        if args.start_pose_pool:
            cfg["start_pose_pool"] = str(Path(args.start_pose_pool).resolve())
        if args.reset_replay:
            if not learning_checkpoint or learning_checkpoint.get("format") != "arm_assignment_dqn_v1":
                raise ValueError("replay reset requires a compatible DQN checkpoint")
            cfg["exploration_decisions_offset"] = (blob.get("config", {}).get("exploration_decisions_offset", 0)
                                                  +blob["replay"]["size"])
            cfg["replay_reset_source"] = str(resume.resolve())
        if args.eval_seconds is not None:
            cfg["eval_seconds"] = args.eval_seconds
        if not args.eval_only and not args.benchmark_output:
            run_dir.mkdir(parents=True, exist_ok=True)
            if args.run_name and not (run_dir/"config.json").exists():
                (run_dir/"config.json").write_text(json.dumps(cfg, indent=2))
            if not (run_dir/"assignment_initial.pt").exists():
                torch.save(dict(format="arm_assignment_dqn_init_v1", policy=policy.state_dict(),
                                policy_config=asdict(policy.cfg), config=cfg,
                                source_checkpoint=str(resume.resolve())), run_dir/"assignment_initial.pt")
            (run_dir/"dqn_config.json").write_text(json.dumps(cfg, indent=2))
            if os.environ.get("TB_AUTO_REGISTER", "1") != "0":
                registry = ROOT/"runs/tensorboard_active"
                registry.mkdir(parents=True, exist_ok=True)
                link = registry/run_dir.name
                if not link.exists():
                    link.symlink_to((run_dir/"tb").resolve())
            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter(str(run_dir/"tb")) as setup_writer:
                setup_writer.add_scalar("setup/configured_environments", args.envs, 0)
                setup_writer.add_scalar("setup/lookahead_beats", policy.cfg.lookahead_beats, 0)
                setup_writer.add_scalar("setup/commit_lead_s", policy.cfg.commit_lead_s or policy.cfg.lookahead_s, 0)
                setup_writer.add_scalar("setup/pending_per_arm", policy.cfg.pending_per_arm, 0)
                setup_writer.add_scalar("setup/gradient_steps_per_collection", args.gradient_steps, 0)
                setup_writer.flush()

        kit = A.DrumKit.from_layout()
        seed_path = max(cfg["training_clips"], key=lambda path: len(A.load_rcc(path, kit, on_ineligible="keep").hits))
        task = "Isaac-G1-Drum-Direct-v0"
        ec = parse_env_cfg(task, device=str(dev), num_envs=args.envs)
        ec.seed = cfg["seed"]
        configure_drum_kit_geometry(ec)
        configure_tip_geometry(ec)
        ec.pd_damping_scale = 4.0
        ec.foh_vel_target_ema = 1.0
        ec.dynamics_ff_inertial_enable = False
        ec.start_joint_jitter_rad = 0.0
        ec.perturb_enable = False
        ec.dr_enable = False
        enable_hit_telemetry(ec)
        ec.hit_detection_mode = "contact"
        ec.contact_force_threshold = 1.0
        ec.rhythm_file = str(seed_path)
        ec.rhythm_pool = None
        ec.rhythm_curriculum = None
        ec.__post_init__()
        ec.scene.env_spacing = 4.0
        ec.sim.dt = A.PD_DT
        ec.decimation = int(round(A.CONTROL_DT/A.PD_DT))
        ec.sim.render_interval = ec.decimation
        ec.episode_length_s = max(cfg["eval_seconds"]+cfg["tail_s"], cfg["window_s"]+1+cfg["tail_s"])+2
        ec.pin_waist = False
        ec.pin_waist_lean = False
        env = gym.make(task, cfg=ec)
        u = env.unwrapped
        u.assignment_kit = kit
        u.ready_joint_pos[:, list(u.upper_body_joint_ids)] = torch.as_tensor(
            ready_joint_pos(G1_UPPER_BODY_JOINTS), device=dev, dtype=torch.float32)
        source = torch.load(cfg["frozen_planner"], map_location="cpu", weights_only=False)
        mc = A.model_config_from_checkpoint(source)
        with rnea_from_cache_if_needed():
            model = A.build_model(mc, dev)
        model.load_state_dict(source["model"], strict=True)
        model.eval().requires_grad_(False)
        sc = source.get("scenario_cfg") or {}
        H = int(sc.get("buckets", [25, 50, 100])[-1])
        rnea = try_build_rnea(dev)
        if rnea is None:
            raise RuntimeError("RNEA cache is required for unchanged physics cost and waist feedforward")
        cost_cfg = replace(A.CostConfig(**source["train_cfg"]["cost"]), n_sub=4, w_velocity=1.0)
        cost = A.FusedCost(cost_cfg, rnea=rnea, max_horizon=H, device=dev, dtype=torch.float32, dt=A.CONTROL_DT)
        schedule = BatchedSchedule(args.envs, ScheduleConfig(horizon=H, num_drums=mc.num_drums,
                                   num_future=mc.num_future), device=dev)
        mpc = ParallelCommittedMPC(model, schedule, MPCConfig(horizon=H, cmd_lead_s="fit_kd4",
                                  fresh_accel_init=True, fresh_terminal_every_tick=True,
                                  waist_upright_ease_wp=sc.get("waist_upright_ease_wp")),
                                  device=dev, kit_names=kit.names, descent_cost=cost)
        pid = ParallelPID(u, mpc, rnea)

        class GeometryScorer:
            def _t(self, x):
                return torch.as_tensor(np.asarray(x), device=dev, dtype=torch.float32)

            def contact_world(self, q, hand, sigma):
                Q = torch.as_tensor(np.asarray(q).reshape(1, -1), device=dev, dtype=torch.float32)
                R, p = mpc._chain_fk().frame(Q, hand+"_wrist")
                vals = [p+(R@torch.as_tensor(offset, device=dev, dtype=Q.dtype).view(1, 3, 1)).squeeze(-1)
                        for offset in (A.stick_contact_offset(hand, sigma), A.stick_tip_offset(hand))]
                direction = (R@torch.as_tensor(A.stick_axis_local(hand), device=dev,
                                              dtype=Q.dtype).view(1, 3, 1)).squeeze(-1)
                return tuple(x[0].detach().cpu().numpy() for x in (*vals, direction))

        scorer = GeometryScorer()
        def geometry(q, hand, drum):
            axial, lateral, inside = head_frame_error(scorer, q.detach().cpu().numpy(), hand, kit, drum,
                                                      A.default_contact_sigma(kit.names[drum]))
            return dict(axial_mm=float(axial*1000), lateral_beyond_mm=float(lateral*1000), in_hit_region=inside)
        cfg["geometry_scorer"] = geometry
        print(f"[assignment setup] frozen planner step={source.get('step')} envs={args.envs} "
              f"controller=committed one-pass/PID/contact credit resume={resume}", flush=True)
        if args.close_finetune:
            from isaac_rl.assignment_isaac_close import run_close_experiment
            run_close_experiment(env, mpc, pid, policy, cfg, args)
            return
        if args.benchmark_output:
            import resource
            import time
            rng = np.random.default_rng(cfg["seed"])
            if "rng" in blob:
                rng.bit_generator.state = blob["rng"]
            from isaac_rl.assignment_train import window_clip
            choices = rng.integers(len(cfg["training_clips"]), size=args.envs)
            picks = [load_with_tempo(cfg["training_clips"][int(i)], kit, policy.cfg) for i in choices]
            windows = [window_clip(c, rng, span_s=args.benchmark_window_s, tail_s=cfg["tail_s"])
                       for c, _ in picks]
            torch.cuda.reset_peak_memory_stats(dev)
            start = time.perf_counter()
            def progress(t, n):
                finite = bool(torch.isfinite(u.robot.data.joint_pos).all())
                if not finite:
                    raise FloatingPointError("nonfinite Isaac joint state in scale benchmark")
                print(f"[assignment benchmark] tick={t}/{n} envs={args.envs} "
                      f"elapsed_s={time.perf_counter()-start:.1f} "
                      f"torch_reserved_gib={torch.cuda.memory_reserved(dev)/2**30:.2f}", flush=True)
            from isaac_rl.assignment_parallel_train import make_start_sampler
            start_sampler = make_start_sampler(mpc, pid, cfg)
            start_poses, start_stats = start_sampler.sample(rng, args.envs) if start_sampler else (None, {})
            _, reports, times = collect(env, mpc, pid, policy, windows, [pc for _, pc in picks], cfg,
                                        epsilon=0.1, progress=progress, start_poses=start_poses)
            torch.cuda.synchronize(dev)
            seconds = time.perf_counter()-start
            steps = sum(r["steps"] for r in reports)
            result = dict(environments=args.envs, seconds=seconds, environment_steps=steps,
                          environment_steps_per_second=steps/seconds, timings=times,
                          requested=sum(r["requested"] for r in reports), hits=sum(r["hits"] for r in reports),
                          peak_torch_allocated_gib=torch.cuda.max_memory_allocated(dev)/2**30,
                          peak_torch_reserved_gib=torch.cuda.max_memory_reserved(dev)/2**30,
                          peak_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                          checkpoint_iteration=blob.get("iteration"), window_s=args.benchmark_window_s)
            result["starting_pose"] = start_stats
            Path(args.benchmark_output).write_text(json.dumps(result, indent=2))
            print(f"[assignment benchmark result] {result}", flush=True)
            return
        if args.graph_check or args.eval_only:
            clip, pc = load_with_tempo(cfg["eval_clip"], kit, policy.cfg)
            clip, _ = A.pad_first_hit_to(clip, 1.0)
            clip = replace(clip, hits=[h for h in clip.hits if h.time <= cfg["eval_seconds"]],
                           duration=cfg["eval_seconds"]+cfg["tail_s"])
            if args.legacy_lookahead:
                pc = replace(pc, lookahead_s=2, lookahead_beats=0)
            _, reports, times = collect(env, mpc, pid, policy, [clip]*args.envs, [pc]*args.envs,
                                        cfg, graph=args.graph_check,
                                        progress=lambda t, n: print(f"[assignment check] tick={t}/{n}", flush=True))
            path = Path(args.check_output) if args.check_output else run_dir/(
                "dqn_graph_check.json" if args.graph_check else "dqn_policy_check.json")
            path.write_text(json.dumps(dict(reports=reports, timings=times), indent=2))
            print(f"[assignment check] hits={[r['hits'] for r in reports]} / {reports[0]['requested']} saved={path}", flush=True)
        if not args.eval_only:
            train(env, mpc, pid, cfg, policy, checkpoint=learning_checkpoint, reset_replay=args.reset_replay)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
