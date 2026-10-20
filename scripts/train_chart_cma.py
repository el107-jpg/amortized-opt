#!/usr/bin/env python3
"""CMA-ES conditional-chart search with a frozen batched Isaac controller."""
from __future__ import annotations

import argparse
import faulthandler
from dataclasses import replace
import hashlib
from itertools import groupby
import json
import os
import signal
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from isaac_rl.chart_cma import (ChartCMA, plan_population, aggregate_population,
    plan_independent_population)
from isaac_rl.chart_cma_sampling import (IndependentRhythmSampler,
    ProceduralChartRhythmSampler, planned_transition_coverage)
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.min_time_assignment import MinimumTimeChart


def write_json(path, value):
    temp = Path(str(path)+".tmp")
    temp.write_text(json.dumps(value, indent=2, default=float, allow_nan=False)+"\n")
    temp.replace(path)


def export_overlay(optimizer, residual, path, selection):
    chart = optimizer.candidate_chart(residual)
    chart.metadata["parameter_search"] = dict(algorithm="CMA-ES",
        generation=optimizer.generation, selection=selection,
        run_directory=str(path.parent.resolve()))
    temporary = path.with_name(path.name+".tmp")
    chart.save(temporary)
    temporary.replace(path)


def load_configured_overlay(args, base):
    chart = ConditionalMinimumTimeChart.load(args.overlay, base)
    if getattr(args, 'initialize_unknown', False):
        chart = chart.with_initialized_unknown(center_s=args.unknown_center_s,
            half_width_s=args.unknown_half_width_s, seed=args.unknown_seed)
    return chart


def physics_timing(physics_hz, control_dt=.02):
    """Keep decoder/control time fixed and require whole physics substeps."""
    steps = float(physics_hz)*control_dt
    if (not np.isfinite(steps) or steps < 1 or abs(steps-round(steps)) > 1e-8):
        raise ValueError('Physics frequency must be a positive integer multiple of the 50 Hz control rate')
    return 1./float(physics_hz), int(round(steps))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=str(ROOT/"runs/transition_decoder_torque_integral_20261005_2240/collision_5mm/transition_table.npz"))
    parser.add_argument("--overlay", default=str(ROOT/"runs/conditional_min_time_chart_20261006/conditional_chart.npz"))
    parser.add_argument("--initialize-unknown", action=argparse.BooleanOptionalAction, default=True,
        help="include unknown eligible cells with randomized planning guesses (default: enabled for new searches)")
    parser.add_argument("--unknown-center-s", type=float, default=.3)
    parser.add_argument("--unknown-half-width-s", type=float, default=.05)
    parser.add_argument("--unknown-seed", type=int, default=20261006)
    parser.add_argument("--ready-initial", default=str(ROOT/"runs/min_time_chart_vs_graph_4_6_groups_20261006/ready_initial_minimum_s.npy"))
    parser.add_argument("--decoder-checkpoint", help="explicit decoder override while retaining existing timing/ready matrices")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", help="local last.pkl checkpoint; saved settings restored except generation budget/device/env capacity")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--sampling", choices=("independent", "matched"), default="independent")
    parser.add_argument("--rhythm-source", choices=("procedural", "recorded"), default=None)
    parser.add_argument("--fitness-ranking", choices=("speed", "global"), default=None)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--scenarios", type=int, default=1)
    parser.add_argument("--validation-scenarios", type=int, default=0)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--horizon", type=int, choices=(4, 6), default=4)
    parser.add_argument("--sigma", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--decode-chunk", type=int, default=None)
    parser.add_argument("--collision-chunk", type=int, default=None)
    parser.add_argument("--plan-diagnostics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--planning-workers", type=int, default=None)
    parser.add_argument("--rollout-devices", nargs='+', default=None,
        help="persistent Isaac workers on distinct GPUs; --envs is their total capacity")
    parser.add_argument("--physics-hz", type=float, default=None,
        help="physics frequency (legacy/default 250); control remains 50 Hz")
    parser.add_argument("--clone-kit-from-source", action="store_true",
        help="spawn the nominal kit once and replicate it with the scene")
    parser.add_argument('--freeze-hit-scoring',action=argparse.BooleanOptionalAction,default=None,help='Keep fresh-scene hit-credit timing/speed kernels fixed')
    parser.add_argument('--decoder-fp32',action=argparse.BooleanOptionalAction,default=None,help='Disable bfloat16/TF32 decoder inference for repeatability probes')
    parser.add_argument('--rollout-audit',action=argparse.BooleanOptionalAction,default=None,help='Save command/state/credit traces; for small diagnostic panels')
    parser.add_argument('--clear-reset-contacts',action=argparse.BooleanOptionalAction,default=None,help='Clear pre-teleport contact reports before the first physics substep')
    parser.add_argument('--match-decoder-speed',action=argparse.BooleanOptionalAction,default=None,help='Grade impact speed against the decoder requested m/s target')
    parser.add_argument("--torque-weight", type=float, default=.01)
    parser.add_argument("--joint-weight", type=float, default=.1)
    parser.add_argument("--collision-weight", type=float, default=.05)
    parser.add_argument("--collision-stride", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=.001)
    parser.add_argument("--near-miss-weight", type=float, default=None,
        help="maximum proximity credit for a missed hit (new runs default 0.25)")
    parser.add_argument("--near-miss-scale-m", type=float, default=None)
    parser.add_argument("--near-miss-window-s", type=float, default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--validation-dir", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dry-run", action="store_true")
    # AppLauncher arguments are registered only by the actual Isaac entrypoint.
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args(argv)
    proximity_keys = ('near_miss_weight', 'near_miss_scale_m', 'near_miss_window_s')
    explicit_proximity = {key: getattr(args, key) for key in proximity_keys}
    explicit_source, explicit_ranking = args.rhythm_source, args.fitness_ranking
    explicit_execution = {k: getattr(args, k) for k in ('decode_chunk', 'collision_chunk', 'plan_diagnostics', 'planning_workers', 'rollout_devices', 'physics_hz','freeze_hit_scoring','decoder_fp32','rollout_audit','clear_reset_contacts','match_decoder_speed')}
    saved = {}
    if args.resume:
        saved = json.loads((Path(args.resume).resolve().parent/"config.json").read_text())["args"]
        # Old checkpoints keep their original dimension and fingerprint.
        args.initialize_unknown = saved.get("initialize_unknown", False)
        args.sampling = saved.get("sampling", "matched")
        if args.sampling == 'matched':
            args.validation_scenarios = saved.get('validation_scenarios', 8)
        for key, value in saved.items():
            if key not in ("out", "resume", "generations", "device", "envs", "dry_run", "headless",
                    "rhythm_source", "fitness_ranking", *proximity_keys):
                setattr(args, key, value)
    for key, default in [('decode_chunk', 128), ('collision_chunk', None), ('plan_diagnostics', True), ('planning_workers', 1), ('rollout_devices', []), ('physics_hz', 250.),('freeze_hit_scoring',not bool(args.resume)),('decoder_fp32',False),('rollout_audit',False),('clear_reset_contacts',not bool(args.resume)),('match_decoder_speed',not bool(args.resume))]:
        setattr(args, key, explicit_execution[key] if explicit_execution[key] is not None else saved.get(key, default))
    try:
        physics_timing(args.physics_hz)
    except ValueError as exc:
        parser.error(str(exc))
    if args.rollout_devices and (len(set(args.rollout_devices)) != len(args.rollout_devices)
            or any(not d.startswith('cuda:') or not d[5:].isdigit() for d in args.rollout_devices)):
        parser.error('Rollout devices must be distinct explicit CUDA devices, e.g. cuda:0 cuda:1')
    if args.collision_chunk is None:
        args.collision_chunk = args.decode_chunk
    old_source = saved.get('rhythm_source', 'recorded')
    args.rhythm_source = (explicit_source if explicit_source is not None else old_source
        if args.resume else 'procedural' if args.sampling == 'independent' else 'recorded')
    source_changed = bool(args.resume and old_source != args.rhythm_source)
    args.fitness_ranking = (explicit_ranking if explicit_ranking is not None else
        'speed' if args.rhythm_source == 'procedural' and (source_changed or not args.resume) else
        saved.get('fitness_ranking', 'global'))
    if args.rhythm_source == 'procedural' and args.sampling != 'independent':
        parser.error('Procedural CMA currently requires independent sampling')
    if args.fitness_ranking == 'speed' and args.rhythm_source != 'procedural':
        parser.error('Speed-tier fitness ranking requires procedural rhythms')
    defaults = dict(near_miss_weight=0. if args.resume else .25,
        near_miss_scale_m=.05, near_miss_window_s=.08)
    for key in proximity_keys:
        setattr(args, key, explicit_proximity[key] if explicit_proximity[key] is not None else saved.get(key, defaults[key]))
    if (not np.isfinite([getattr(args, k) for k in proximity_keys]).all()
            or not 0 <= args.near_miss_weight < 1
            or min(args.near_miss_scale_m, args.near_miss_window_s) <= 0):
        parser.error('Near-miss weight must be in [0,1); scale/window must be positive')
    if (args.population < 4 or min(args.generations, args.scenarios,
            args.envs, args.decode_chunk, args.collision_chunk, args.collision_stride, args.planning_workers) < 1
            or args.groups < 2):
        parser.error("Positive budgets/capacities, population>=4 and groups>=2 required")
    if args.sampling == 'independent' and (args.scenarios != 1 or args.validation_scenarios != 0):
        parser.error('Independent mode requires --scenarios 1 and --validation-scenarios 0')
    if args.sampling == 'matched' and args.validation_scenarios < 1:
        parser.error('Matched mode requires a positive --validation-scenarios count')
    for key in ("sigma", "regularization", "torque_weight", "joint_weight", "collision_weight"):
        value = getattr(args, key)
        if not np.isfinite(value) or value < 0 or (key == "sigma" and value == 0):
            parser.error(f"Invalid {key}")
    return args


def sample_panel(clips, count, groups, rng, chart):
    from isaac_rl.assignment_search import hit_groups
    result = []
    for _ in range(count*200):
        source = clips[int(rng.integers(len(clips)))]
        all_groups = [list(g) for _, g in groupby(sorted(source.hits, key=lambda h: (h.time, h.drum)), key=lambda h: h.time)]
        if len(all_groups) < groups:
            continue
        start = int(rng.integers(len(all_groups)-groups+1))
        selected = all_groups[start:start+groups]
        if any(len(g) > 2 or not chart.choices(g) for g in selected):
            continue
        gaps = np.diff([g[0].time for g in selected])
        if (gaps > 1.+1e-8).any() or (gaps < .02-1e-8).any():
            continue
        offset = 1.-selected[0][0].time
        hits = [replace(h, time=round(h.time+offset, 8), raw_time=h.raw_time+offset) for g in selected for h in g]
        result.append(replace(source, name=f"{source.name}@cma_group{start}", hits=hits, duration=hits[-1].time+.3))
        if len(result) == count:
            return result
    raise ValueError("Could not fill an eligible matched rhythm panel")


def setup_isaac(args, checkpoint, source, seed_clip):
    # Pin the same fingerprinted plant used by the prior Isaac probes and avoid
    # live Pinocchio builds inside the simulator's shared-library environment.
    os.environ.setdefault("AMORTIZED_RNEA_CACHE_ONLY", "1")
    import torch
    from isaaclab.app import AppLauncher
    launcher_kwargs=dict(headless=True,device=args.device)
    if getattr(args,'capture_video',False):launcher_kwargs['enable_cameras']=True
    launcher = AppLauncher(**launcher_kwargs)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    env = None
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import g1_drummer.tasks  # noqa: F401
        from isaac_rl.decoder_scene import (bind_registered_task_geometry,
            configure_tip_geometry,configure_drum_kit_geometry)
        bind_registered_task_geometry()
        from g1_drummer.viz import enable_hit_telemetry
        from isaaclab_tasks.utils import parse_env_cfg
        import amortized_opt as A
        from amortized_opt.constants import ready_joint_pos, G1_UPPER_BODY_JOINTS
        from isaac_rl.env import try_build_rnea
        from isaac_rl.rnea_cache import rnea_from_cache_if_needed
        from isaac_rl.schedule import BatchedSchedule, ScheduleConfig
        from isaac_rl.mpc import MPCConfig
        from isaac_rl.assignment_parallel import ParallelPID
        from isaac_rl.chart_cma_isaac import ChunkedChartMPC, IsaacChartEvaluator
        torch.cuda.set_device(args.device)
        torch.set_float32_matmul_precision('highest' if getattr(args,'decoder_fp32',False) else 'high')
        if getattr(args,'decoder_fp32',False):
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        kit = A.DrumKit.from_layout()
        cfg = parse_env_cfg("Isaac-G1-Drum-Direct-v0", device=args.device, num_envs=args.envs)
        cfg.seed = args.seed; cfg.pd_damping_scale = 4.
        cfg.foh_vel_target_ema = 1.; cfg.dynamics_ff_inertial_enable = False
        cfg.start_joint_jitter_rad = 0.; cfg.perturb_enable = False; cfg.dr_enable = False
        enable_hit_telemetry(cfg)
        cfg.hit_detection_mode = "contact"; cfg.contact_force_threshold = 1.
        cfg.rhythm_file = str(seed_clip); cfg.rhythm_pool = None; cfg.rhythm_curriculum = None
        configure_drum_kit_geometry(cfg)
        cfg.__post_init__(); cfg.scene.env_spacing = 4.
        configure_tip_geometry(cfg)
        cfg.sim.dt, cfg.decimation = physics_timing(getattr(args, 'physics_hz', 250.), A.CONTROL_DT)
        cfg.sim.render_interval = cfg.decimation
        print(f'[cma-setup] physics={1/cfg.sim.dt:g}Hz control={1/A.CONTROL_DT:g}Hz substeps={cfg.decimation}', flush=True)
        cfg.episode_length_s = args.groups+4.
        cfg.pin_waist = False; cfg.pin_waist_lean = False
        if args.clone_kit_from_source:
            # The scene already replicates env_0, including its static kit.
            # Avoid repeated USD lookups/spawns for thousands of identical kits.
            original_spawn = cfg.drum_kit_cfg.spawn.func
            def spawn_source_kit(prim_path, *positional, **keywords):
                if prim_path == '/World/envs/env_0/DrumKit':
                    return original_spawn(prim_path, *positional, **keywords)
                return None
            cfg.drum_kit_cfg.spawn.func = spawn_source_kit
        env = (gym.make("Isaac-G1-Drum-Direct-v0",cfg=cfg,render_mode='rgb_array')
            if getattr(args,'capture_video',False) else gym.make("Isaac-G1-Drum-Direct-v0",cfg=cfg))
        print("[cma-setup] Isaac scene ready", flush=True)
        u = env.unwrapped
        u.assignment_kit = kit
        if getattr(args,'match_decoder_speed',False) and not hasattr(u,'_beat_target_speeds'):
            raise RuntimeError('Installed Isaac drum task lacks explicit physical impact-speed scoring support')
        if getattr(args,'freeze_hit_scoring',False):
            from isaac_rl.rollout_reproducibility import freeze_initial_hit_scoring
            print('[scoring] '+json.dumps(freeze_initial_hit_scoring(u)),flush=True)
        u.ready_joint_pos[:, list(u.upper_body_joint_ids)] = torch.tensor(
            ready_joint_pos(G1_UPPER_BODY_JOINTS), device=args.device, dtype=u.ready_joint_pos.dtype)
        mc = A.model_config_from_checkpoint(source)
        with rnea_from_cache_if_needed():
            model = A.build_model(mc, args.device)
        model.load_state_dict(source["model"], strict=True); model.eval().requires_grad_(False)
        print("[cma-setup] frozen decoder loaded", flush=True)
        H = int((source.get("scenario_cfg") or {}).get("buckets", [25, 50])[-1])
        if H < 50:
            raise ValueError("Ready calibration/lead requires at least a one-second decoder horizon")
        rnea = try_build_rnea(args.device)
        if rnea is None:
            raise RuntimeError("Frozen RNEA cache required")
        cost_cfg = replace(A.CostConfig(**source["train_cfg"]["cost"]), n_sub=4, w_velocity=1.)
        cost = A.FusedCost(cost_cfg, rnea=rnea, max_horizon=H, device=args.device, dt=A.CONTROL_DT)
        schedule = BatchedSchedule(args.envs, ScheduleConfig(horizon=H, num_drums=mc.num_drums,
            num_future=mc.num_future), device=args.device)
        mpc = ChunkedChartMPC(model, schedule, MPCConfig(horizon=H, cmd_lead_s="fit_kd4",
            fresh_accel_init=True, fresh_terminal_every_tick=True,
            amp_dtype=None if getattr(args,'decoder_fp32',False) else torch.bfloat16,
            waist_upright_ease_wp=(source.get("scenario_cfg") or {}).get("waist_upright_ease_wp")),
            device=args.device, kit_names=kit.names, descent_cost=cost, decode_chunk=args.decode_chunk,
            plan_diagnostics=args.plan_diagnostics)
        pid = ParallelPID(u, mpc, rnea)
        print("[cma-setup] chunked decoder/PID ready", flush=True)
        evaluator = IsaacChartEvaluator(env, mpc, pid, seed=args.seed,
            torque_weight=args.torque_weight, joint_weight=args.joint_weight,
            collision_weight=args.collision_weight, collision_stride=args.collision_stride,
            near_miss_weight=args.near_miss_weight, near_miss_scale_m=args.near_miss_scale_m,
            near_miss_window_s=args.near_miss_window_s, collision_chunk=args.collision_chunk,
            rollout_audit=getattr(args,'rollout_audit',False),clear_reset_contacts=getattr(args,'clear_reset_contacts',False),
            match_decoder_speed=getattr(args,'match_decoder_speed',False),
            video_path=getattr(args,'video_path',None),
            video_camera=getattr(args,'video_camera','3quarter'))
        if getattr(args,'rollout_audit',False) and getattr(u.contact_tracker,'_capsule_tracker',False):
            u.contact_tracker.debug_contact_geometry=True
        return launcher, env, evaluator
    except BaseException:
        import traceback
        traceback.print_exc()
        if env is not None:
            env.close()
        launcher.app.close()
        raise


def main(argv=None):
    args = parse_args(argv)
    import torch
    import amortized_opt as A
    base = MinimumTimeChart.load(args.table)
    chart = load_configured_overlay(args, base)
    metadata = json.loads((Path(args.table).parent/"config.json").read_text())
    matrix_checkpoint = Path(metadata['checkpoint'])
    checkpoint = Path(args.decoder_checkpoint) if args.decoder_checkpoint else matrix_checkpoint
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not args.decoder_checkpoint and digest != metadata["checkpoint_sha256"]:
        raise ValueError("The chart's frozen decoder checkpoint changed")
    if args.decoder_checkpoint:
        override = torch.load(checkpoint, map_location='cpu', weights_only=False)
        mc = A.model_config_from_checkpoint(override)
        scenario = override.get('scenario_cfg') or {}
        from amortized_opt.constants import G1_UPPER_BODY_JOINTS
        if (mc.n_dof != len(G1_UPPER_BODY_JOINTS) or mc.max_horizon < 50
                or mc.num_drums < len(base.names)
                or abs(mc.dt-A.CONTROL_DT) > 1e-9
                or (scenario.get('joint_names') is not None and
                    list(scenario['joint_names']) != list(G1_UPPER_BODY_JOINTS))):
            raise ValueError('Decoder override has incompatible joint, drum, horizon or time-grid configuration')
        del override
    initial = np.load(args.ready_initial)
    calibration = json.loads((Path(args.ready_initial).parent/"config.json").read_text())
    if (Path(calibration["table"]).resolve() != Path(args.table).resolve()
            or Path(calibration["checkpoint"]).resolve() != matrix_checkpoint.resolve()
            or initial.shape != (2, len(base.names)) or np.isnan(initial).any() or (initial < 0).any()):
        raise ValueError("Ready calibration must match the base chart and frozen checkpoint")
    train_dir = Path(args.train_dir) if args.train_dir else A.default_rcc_dir("train")
    validation_dir = Path(args.validation_dir) if args.validation_dir else A.default_rcc_dir("eval")
    if args.sampling == 'matched' and train_dir.resolve() == validation_dir.resolve():
        raise ValueError("Training and validation rhythms must use separate splits")
    kit = A.DrumKit.from_layout()
    training = A.load_rcc_pool(train_dir, kit) if args.rhythm_source == 'recorded' else []
    validation = A.load_rcc_pool(validation_dir, kit) if args.sampling == 'matched' else []
    if (args.rhythm_source == 'recorded' and not training) or (args.sampling == 'matched' and not validation):
        raise ValueError("Required rhythm splits must be nonempty")
    sampler = (ProceduralChartRhythmSampler(kit, args.groups, chart) if args.rhythm_source == 'procedural' else
        IndependentRhythmSampler(training, args.groups, chart) if args.sampling == 'independent' else None)
    optimizer = ChartCMA(chart, population=args.population, sigma=args.sigma,
        seed=args.seed, regularization=args.regularization)
    rng = np.random.default_rng(args.seed)
    pending_panel = None
    pending_strata = None
    validated = optimizer.incumbent.copy()
    extra = {}
    if args.resume:
        optimizer, extra = ChartCMA.load(args.resume, chart)
        if extra.get('sampling', 'matched') != args.sampling:
            raise ValueError('Resume cannot change independent/matched sampling mode')
        if extra["checkpoint_sha256"] != digest:
            raise ValueError("Frozen decoder differs from resumed search")
        rng.bit_generator.state = extra["panel_rng"]
        validation_panel = extra["validation_panel"]
        pending_panel = extra.get("pending_panel")
        pending_strata = extra.get('pending_strata')
        if extra.get('rhythm_source', 'recorded') != args.rhythm_source:
            # Keep the sampled parameter offspring and CMA state, replace the
            # obsolete recorded-rhythm panel with a new procedural generation.
            pending_panel = None
            pending_strata = None
        validated = extra["validated"]
    else:
        validation_panel = (sample_panel(validation, args.validation_scenarios, args.groups, rng, chart)
            if args.sampling == 'matched' else [])
    if args.dry_run:
        print(json.dumps(dict(parameters=len(optimizer.initial), population=args.population,
            initialized_unknown_cells=int(chart.initialized_unknown_mask.sum()),
            scenarios=args.scenarios, logical_population_environments=args.population*args.scenarios,
            sampling=args.sampling, rhythm_source=args.rhythm_source, fitness_ranking=args.fitness_ranking,
            independent_training_windows=len(sampler.records) if isinstance(sampler, IndependentRhythmSampler) else None,
            physical_env_capacity=args.envs, decoder_chunk=args.decode_chunk,
            source_checkpoint=str(checkpoint), restored_generation=optimizer.generation,
            matrix_checkpoint=str(matrix_checkpoint), decoder_overridden=bool(args.decoder_checkpoint),
            optimizer="CMA-ES", phase2_decoder_training=False), indent=2))
        return
    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError("Use a new run directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    chart.save(out/"initial_chart.npz")
    if not args.resume:
        export_overlay(optimizer, optimizer.incumbent, out/'last_chart.npz',
            'cma_population_mean' if args.sampling == 'independent' else 'current_matched_training_panel')
    config = dict(args=vars(args), checkpoint=str(checkpoint), checkpoint_sha256=digest,
        matrix_checkpoint=str(matrix_checkpoint), matrix_checkpoint_sha256=metadata['checkpoint_sha256'],
        decoder_overridden=bool(args.decoder_checkpoint),
        algorithm="CMA-ES conditional chart search", sampling=args.sampling,
        rhythm_source=args.rhythm_source, fitness_ranking=args.fitness_ranking,
        rhythm_generator=sampler.summary if isinstance(sampler, ProceduralChartRhythmSampler) else None,
        rollouts_per_offspring=args.scenarios, phase2_decoder_training=False,
        frozen_generator=True, controller="Committed one-pass decoder / PID5 / fit_kd4 / contact compliance",
        fitness="Credited-hit fraction plus bounded missed-hit proximity minus demand-torque, geometry and joint penalties",
        reward_version='contact_plus_proximity_v1' if args.near_miss_weight else 'contact_only_v1',
        overlay=chart.description())
    write_json(out/"config.json", config)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out/"tb"), flush_secs=5)
    writer.add_text("setup/config", json.dumps(config), 0); writer.flush()
    if os.environ.get("TB_AUTO_REGISTER", "1") != "0":
        registry = ROOT/"runs/tensorboard_active"; registry.mkdir(exist_ok=True)
        link = registry/out.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(out/"tb")
        if not link.is_symlink() or link.resolve() != out/"tb":
            raise ValueError("TensorBoard registration already belongs to another directory")
        for _ in range(6):
            names = json.load(urllib.request.urlopen("http://localhost:6006/data/runs", timeout=10))
            if out.name in names:
                break
            time.sleep(5)
        else:
            raise RuntimeError("Shared TensorBoard server did not list this training run")
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sampler, ProceduralChartRhythmSampler):
        seed_clip = out/'procedural_scene_seed.json'
        write_json(seed_clip, sampler.seed_payload())
    else:
        seed_clip = max(train_dir.glob("*.json"), key=lambda p: p.stat().st_size)
    launcher = env = evaluator = None
    planner = None
    if args.planning_workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from isaac_rl.chart_cma import planning_worker_ready
        planner = ProcessPoolExecutor(max_workers=args.planning_workers,
            mp_context=multiprocessing.get_context('spawn'))
        for job in [planner.submit(planning_worker_ready) for _ in range(args.planning_workers)]:
            job.result()  # Start CPU workers before Isaac alters the process environment.
    started = time.monotonic()
    def save_state(panel=None, strata=None):
        optimizer.save(out/"last.pkl", panel_rng=rng.bit_generator.state,
            validation_panel=validation_panel, pending_panel=panel, validated=validated,
            checkpoint_sha256=digest, sampling=args.sampling, rhythm_source=args.rhythm_source,
            pending_strata=strata)
    try:
        if args.rollout_devices:
            from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
            evaluator = DistributedIsaacEvaluator(args, checkpoint, seed_clip)
        else:
            launcher, env, evaluator = setup_isaac(args, checkpoint, source, seed_clip)
        write_json(out/"run.json", dict(status="running", generation=optimizer.generation))
        while optimizer.generation < args.generations and not optimizer.es.stop():
            generation_started = time.monotonic()
            reusing_panel = pending_panel is not None
            panel = pending_panel if pending_panel is not None else (
                sampler.sample(args.population, rng) if sampler else
                sample_panel(training, args.scenarios, args.groups, rng, chart))
            candidates = optimizer.ask()
            strata = (pending_strata if reusing_panel else sampler.last_strata
                if isinstance(sampler, ProceduralChartRhythmSampler) else None)
            if args.fitness_ranking == 'speed' and (strata is None or len(strata) != len(candidates)):
                raise ValueError('Speed-tier ranking requires saved tier labels for every offspring')
            save_state(panel, strata)
            if args.sampling == 'independent':
                def planning(done, total):
                    write_json(out/'progress.json', dict(generation=optimizer.generation+1,
                        phase='assignment_planning', offspring_done=done, offspring=total))
                    print(f'[cma-plan] offspring={done}/{total}', flush=True)
                plan = plan_independent_population(optimizer, candidates, panel, initial, args.horizon, planning,
                    executor=planner)
                planning_s = time.monotonic()-generation_started
                def rollout(offset, total, tick, ticks):
                    write_json(out/'progress.json', dict(generation=optimizer.generation+1,
                        phase='independent_rollouts', episode_offset=offset, episodes=total,
                        tick=tick, ticks=ticks))
                    print(f'[cma] independent offspring={offset}/{total} tick={tick}/{ticks}', flush=True)
                rollout_started = time.monotonic()
                reports = evaluator.evaluate(plan['episodes'], rollout)
                rollout_s = time.monotonic()-rollout_started
                values = aggregate_population(plan, [r['reward'] for r in reports])
                hit_rates = aggregate_population(plan, [r['hit_rate'] for r in reports])
                stats = optimizer.tell_independent(values,
                    strata=strata if args.fitness_ranking == 'speed' else None)
                pending_panel = None
                pending_strata = None
                export_overlay(optimizer, optimizer.incumbent, out/'last_chart.npz', 'cma_population_mean')
                save_state()
                stats.update(sampling='independent', rollouts_per_offspring=1,
                    physical_env_capacity=args.envs, logical_rollouts=args.population,
                    unique_rollouts=plan['unique_rollouts'],
                    planning_seconds=planning_s, rollout_seconds=rollout_s,
                    generation_seconds=time.monotonic()-generation_started,
                    mean_offspring_hit_rate=float(np.nanmean(hit_rates)) if np.isfinite(hit_rates).any() else None,
                    mean_offspring_near_miss_bonus=float(np.mean([r['near_miss_bonus'] for r in reports])) if reports else None,
                    mean_offspring_hit_quality=float(np.mean([r['hit_quality'] for r in reports])) if reports else None,
                    reward_version='contact_plus_proximity_v1' if args.near_miss_weight else 'contact_only_v1',
                    rhythm_source=args.rhythm_source,
                    speed_tier_offspring_counts=np.bincount(strata, minlength=3).tolist() if strata is not None else None,
                    offspring_hits=sum(r['hits'] for r in reports),
                    offspring_requested=sum(r['requested'] for r in reports),
                    elapsed_s=time.monotonic()-started)
                stats.update(planned_transition_coverage(plan['episodes'], len(chart.names)))
                write_json(out/f'generation_{optimizer.generation:05d}.json', dict(metrics=stats,
                    training_reports=reports, training_gather=plan['gather'].tolist(),
                    speed_tiers=strata.tolist() if strata is not None else None,
                    training_rhythms=[c.name for c in panel]))
                with (out/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(stats)+'\n')
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar('cma/'+key, value, optimizer.generation)
                writer.flush()
                write_json(out/'run.json', dict(status='running', generation=optimizer.generation,
                    sampling='independent', parameters=len(optimizer.initial)))
                print('[cma-result] '+json.dumps(stats), flush=True)
                continue
            plan = plan_population(optimizer, candidates, panel, initial, args.horizon)
            def progress(offset, total, tick, ticks):
                write_json(out/"progress.json", dict(generation=optimizer.generation+1,
                    episode_offset=offset, episodes=total, tick=tick, ticks=ticks))
                print(f"[cma] generation={optimizer.generation+1} episodes={offset}/{total} tick={tick}/{ticks}", flush=True)
            reports = evaluator.evaluate(plan["episodes"], progress)
            values = aggregate_population(plan, [r["reward"] for r in reports])
            train_hits = aggregate_population(plan, [r["hit_rate"] for r in reports])
            stats = optimizer.tell(values[:-2], values[-2], distinct_schedules=plan["distinct_schedules"])
            pending_panel = None
            # Re-evaluate both current training winner and previous validated
            # incumbent on the same fixed held-out panel. Never compare stale scores.
            validation_candidates = np.stack((optimizer.incumbent, validated))
            vplan = plan_population(optimizer, validation_candidates, validation_panel, initial, args.horizon)
            vreport = evaluator.evaluate(vplan["episodes"])
            vvalues = aggregate_population(vplan, [r["reward"] for r in vreport])
            validation_hits = aggregate_population(vplan, [r["hit_rate"] for r in vreport])
            selected_candidate = vvalues[0]-optimizer.penalty(optimizer.incumbent) > vvalues[1]-optimizer.penalty(validated)
            if selected_candidate:
                validated = optimizer.incumbent.copy()
            export_overlay(optimizer, validated, out/"best_chart.npz", "paired_fixed_validation")
            export_overlay(optimizer, optimizer.incumbent, out/"last_chart.npz", "current_matched_training_panel")
            save_state()
            stats.update(train_previous_incumbent_reward=float(values[-2]), train_graph_reward=float(values[-1]),
                validation_candidate_reward=float(vvalues[0]), validation_previous_reward=float(vvalues[1]),
                validation_graph_reward=float(vvalues[-1]), unique_rollouts=plan["unique_rollouts"],
                train_previous_hit_rate=float(train_hits[-2]), train_graph_hit_rate=float(train_hits[-1]),
                validation_candidate_hit_rate=float(validation_hits[0]),
                validation_previous_hit_rate=float(validation_hits[1]),
                validation_best_hit_rate=float(validation_hits[0 if selected_candidate else 1]),
                validation_graph_hit_rate=float(validation_hits[-1]),
                validation_best_reward=float(vvalues[0 if selected_candidate else 1]),
                validated_changed=bool(selected_candidate),
                logical_rollouts=plan["logical_rollouts"], distinct_schedules=plan["distinct_schedules"],
                elapsed_s=time.monotonic()-started)
            write_json(out/f"generation_{optimizer.generation:05d}.json", dict(metrics=stats,
                training_reports=reports, validation_reports=vreport,
                training_gather=plan["gather"].tolist(), validation_gather=vplan["gather"].tolist()))
            with (out/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(stats)+"\n")
            for key, value in stats.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar("cma/"+key, value, optimizer.generation)
            writer.flush()
            print("[cma-result] "+json.dumps(stats), flush=True)
        write_json(out/"run.json", dict(status="complete", generations=optimizer.generation,
            stop_reasons={k: str(v) for k,v in optimizer.es.stop().items()}, elapsed_s=time.monotonic()-started))
    except BaseException as exc:
        write_json(out/"run.json", dict(status="failed", error=str(exc), generation=optimizer.generation))
        raise
    finally:
        if args.rollout_devices and evaluator is not None:
            evaluator.close()
        if planner is not None:
            planner.shutdown(wait=True)
        writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
