"""Paired hand-assignment playback using actual Isaac contacts and hit credit.

Imported after AppLauncher by the standard player. No training or simulated
rebound is performed. Physics, PID, contact geometry and credit rules are kept
at the player's configured settings.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from amortized_opt.rhythm import Hit, RhythmClip
from isaac_rl.arm_assignment import AssignmentConfig, AssignmentQueue
from isaac_rl.assignment_train import install_beats


def save(path, value):
    temp = Path(str(path)+".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(path)


def assigned_case(case):
    clip = RhythmClip(case["name"], [Hit(**h) for h in case["hits"]], case["duration"])
    return clip


def run_comparison(env, agent, config_path, simulation_app):
    cfg = json.loads(Path(config_path).read_text())
    out = Path(cfg["out"])
    u = env.unwrapped
    steps = int(cfg["steps"])
    cases = cfg["cases"]
    reports = []
    started = time.monotonic()
    save(out/"run.json", dict(status="running", stage="isaac_rollouts", completed=0))
    try:
        for seed_index, seed in enumerate(cfg["seeds"]):
            # Rotate order across seeds while retaining identical initial state.
            order = cases[seed_index % len(cases):]+cases[:seed_index % len(cases)]
            reference_q = reference_v = None
            for case in order:
                clip = assigned_case(case)
                name = case["method"]
                torch.manual_seed(seed)
                np.random.seed(seed)
                u.seed(seed)
                agent.args.hit_diagnostics = str(out/f"{name}_seed{seed}_hits.json")
                agent._clip = clip
                with torch.inference_mode():
                    env.reset()
                    domain_draws = agent._domain_reset() if hasattr(agent, "_domain_reset") else None
                    queue = AssignmentQueue(clip, agent.kit, AssignmentConfig())
                    # Carry the saved hands directly; never run graph reassignment
                    # over a chart-selected rhythm when installing Isaac credit.
                    queue.hands = {i: hit.hand for i, hit in enumerate(queue.hits)}
                    queue.next_group = len(queue.groups)
                    install_beats(u, queue)
                    u.env_max_steps.fill_(steps+2)
                    agent.reset()
                    q0 = u.robot.data.joint_pos[0, agent._joint_ids].clone().cpu()
                    v0 = u.robot.data.joint_vel[0, agent._joint_ids].clone().cpu()
                if reference_q is None:
                    reference_q, reference_v = q0, v0
                torch.testing.assert_close(q0, reference_q, atol=1e-6, rtol=0)
                torch.testing.assert_close(v0, reference_v, atol=1e-6, rtol=0)
                requested = len(clip.hits)
                assert int(u.beat_valid[0].sum()) == requested
                expected_hands = torch.tensor([h.hand == "right" for h in queue.hits], device=u.device)
                assert torch.equal(u.beat_hand[0, :requested].bool(), expected_hands)
                ended = False
                ticks = 0
                label = f"{name}_seed{seed}"
                print(f"[comparison] {label}: {requested} requested hits, {steps} control steps", flush=True)
                for tick in range(steps):
                    if not simulation_app.is_running():
                        raise RuntimeError("Isaac closed before completing the paired comparison")
                    now = int(u.episode_length_buf[0].item())*u.step_dt
                    with torch.inference_mode():
                        action = agent.act(now)
                        _, reward, terminated, truncated, _ = env.step(action)
                        agent.post_step()
                    ticks = tick+1
                    if tick % 100 == 0:
                        save(out/"progress.json", dict(stage="isaac_rollouts", episode=label,
                            tick=tick, total=steps, completed=len(reports), elapsed_s=time.monotonic()-started))
                        print(f"[comparison] {label} {tick}/{steps} {agent.status_line(now)}", flush=True)
                    if bool(terminated[0]) or bool(truncated[0]):
                        ended = True
                        break
                statistics = agent.violation_summary()
                agent.write_hit_diagnostics()
                beat_done = u.beat_done[0, :requested].detach().cpu().numpy()
                per_drum = {}
                for drum in agent.kit.names:
                    ids = [i for i, h in enumerate(queue.hits) if h.drum_name == drum]
                    per_drum[drum] = dict(requested=len(ids), hits=int(beat_done[ids].sum()),
                                         assigned_left=sum(queue.hits[i].hand == "left" for i in ids))
                report = dict(method=name, seed=seed, requested=requested,
                    credited_hits=statistics["credited_hits"],
                    hit_rate=statistics["credited_hits"]/requested, ticks=ticks,
                    ended_early=ended, statistics=statistics, by_drum=per_drum,
                    initial_q=q0.tolist(), initial_v=v0.tolist(),
                    domain_draws=domain_draws,
                    collision_reporting="Player collision counter is not instrumented; do not interpret zero as collision-free")
                reports.append(report)
                save(out/f"{label}.json", report)
                save(out/"episodes.json", reports)
                print("[comparison-result] "+json.dumps(report), flush=True)
        methods = {}
        for case in cases:
            rows = [r for r in reports if r["method"] == case["method"]]
            requested = sum(r["requested"] for r in rows)
            hits = sum(r["credited_hits"] for r in rows)
            methods[case["method"]] = dict(credited_hits=hits, requested=requested,
                hit_rate=hits/requested, hits_per_seed=[r["credited_hits"] for r in rows],
                physical_touches_per_seed=[r["statistics"]["physical_touches"] for r in rows],
                joint_limit_step_fraction_mean=float(np.mean([r["statistics"]["joint_limit_frac"] for r in rows])),
                torque_step_fraction_mean=float(np.mean([r["statistics"]["torque_frac"] for r in rows])),
                max_torque_over=float(max(r["statistics"]["max_torque_over"] for r in rows)),
                early_terminations=sum(r["ended_early"] for r in rows))
        summary = dict(status="complete", rhythm=cfg["rhythm"], checkpoint=cfg["checkpoint"],
            seeds=cfg["seeds"], elapsed_s=time.monotonic()-started, methods=methods,
            training=False, actual_isaac_contacts=True, simulated_rebound=False,
            matched_initial_state_verified=True,
            controller=cfg["controller"], assignment_scope=cfg["assignment_scope"],
            collision_reporting="Uninstrumented; no collision-free claim")
        save(out/"summary.json", summary)
        save(out/"run.json", summary)
        print("[comparison-complete] "+json.dumps(summary), flush=True)
    except BaseException as exc:
        save(out/"run.json", dict(status="failed", error=str(exc), completed=len(reports)))
        raise
