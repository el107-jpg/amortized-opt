#!/usr/bin/env python3
"""Numerical/autograd audit of the full-RNEA phase-2 torque objective.

This is a read-only diagnostic: it does not change a checkpoint, replay pool,
or training defaults.  It can audit deterministic pool rows or reconstruct a
short MPC replay with the checkpoint policy.

Examples
--------

Healthy checkpoint rows::

    python scripts/audit_full_rnea_numerics.py \
      --checkpoint runs/rhythm_bellman_nf1_ls_full_rnea_fresh/best.pt \
      --pool-cache runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz \
      --device cuda:2 --batch-size 8 --output analysis/rnea_numerics/healthy.json

Reconstructed replay rows::

    python scripts/audit_full_rnea_numerics.py \
      --checkpoint runs/rhythm_bellman_nf1_ls_full_rnea_replay_dw/last.pt \
      --pool-cache runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz \
      --device cuda:2 --batch-size 8 --source replay --prime-roller-ticks 20 \
      --output analysis/rnea_numerics/replay.json

The report separates:

* spline construction error from RNEA arithmetic error (fp32 versus fp64);
* actual training autocast from eager fp32 physics;
* smooth RNEA, warning, p-norm peak, and complete AL torque derivatives;
* finite-difference cancellation from real ReLU/abs/hard-max active-set changes;
* horizon rebuilding from count normalization on fixed states;
* gravity, velocity bias, diagonal inertia, and inertial coupling.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import amortized_opt as A  # noqa: E402
from train_phase2 import _load_npz_buckets  # noqa: E402
from amortized_opt.constants import (  # noqa: E402
    G1_UPPER_BODY_JOINTS,
    torque_limit_eta_scale,
)
from amortized_opt.cost import smooth_max  # noqa: E402
from amortized_opt.fk import StickRandomization  # noqa: E402
from amortized_opt.idle_cylinder import derive_cylinder  # noqa: E402
from amortized_opt.lagrangian import (  # noqa: E402
    ACTUATOR_SEGMENTS,
    AugmentedLagrangian,
)
from amortized_opt.limits import (  # noqa: E402
    available_torque_torch,
    isaac_effort_limits,
    isaac_velocity_limits,
    scale_soft_wrist_effort,
)
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig, mass_matrix  # noqa: E402
from amortized_opt.phase2_data import (  # noqa: E402
    Phase2Noise,
    Phase2Pool,
    Phase2ScenarioConfig,
)
from amortized_opt.rhythm import DrumKit, KitRandomization  # noqa: E402
from amortized_opt.train_cost import (  # noqa: E402
    MPCRoller,
    ReplayPool,
    _dataclass_from_mapping,
    _wtensors,
    build_costs,
    effective_hits,
    group_lengths,
    migrate_al_state,
    phase2_config_from_checkpoint,
    prime_replay_from_roller,
    resolve_idle_cylinder_library,
)
from amortized_opt.tspline import TrajectorySplineBuilder  # noqa: E402
from amortized_opt.waist_aim import WaistAimConfig  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pool-cache", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bucket", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", choices=("pool", "replay"), default="pool")
    p.add_argument(
        "--selection",
        choices=("random", "lowest", "highest"),
        default="random",
        help="pool-row selection by checkpoint torque peak (default: random)",
    )
    p.add_argument(
        "--candidate-size",
        type=int,
        default=128,
        help="pool candidates scored for lowest/highest selection",
    )
    p.add_argument("--prime-roller-ticks", type=int, default=20)
    p.add_argument(
        "--fd-steps",
        type=float,
        nargs="+",
        default=(1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5),
    )
    p.add_argument("--compile", action="store_true", help="also compile isolated RNEA")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    if args.candidate_size < args.batch_size:
        p.error("--candidate-size must be at least --batch-size")
    if any(h <= 0.0 for h in args.fd_steps):
        p.error("--fd-steps must all be positive")
    if args.source == "replay" and args.prime_roller_ticks < 1:
        p.error("replay source needs --prime-roller-ticks >= 1")
    return args


def _number(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return float(x) if x.numel() == 1 else x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _stats(x: torch.Tensor) -> dict[str, float | int | None]:
    y = x.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    finite = y[torch.isfinite(y)]
    if not finite.numel():
        return {
            "count": int(y.numel()),
            "nonfinite": int(y.numel()),
            "min": None,
            "p50": None,
            "rms": None,
            "max": None,
        }
    return {
        "count": int(y.numel()),
        "nonfinite": int(y.numel() - finite.numel()),
        "min": float(finite.min()),
        "p50": float(finite.quantile(0.5)),
        "rms": float(finite.square().mean().sqrt()),
        "max": float(finite.max()),
    }


def _error_stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    a = got.detach().to(device="cpu", dtype=torch.float64)
    b = ref.detach().to(device="cpu", dtype=torch.float64)
    d = a - b
    den = b.square().mean().sqrt().clamp_min(1e-30)
    return {
        "max_abs": float(d.abs().max()),
        "rms_abs": float(d.square().mean().sqrt()),
        "relative_rms": float(d.square().mean().sqrt() / den),
        "max_relative_element": float((d.abs() / b.abs().clamp_min(1e-8)).max()),
    }


def _finite_stage_map(values: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, value in values.items():
        x = value.detach()
        out[name] = {
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "nonfinite": int((~torch.isfinite(x)).sum().item()),
            "max_abs": (
                float(x[torch.isfinite(x)].abs().max().item())
                if bool(torch.isfinite(x).any())
                else None
            ),
        }
    return out


def _digest(*values: torch.Tensor) -> str:
    h = hashlib.sha256()
    for value in values:
        a = value.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode())
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def _scenario_from_checkpoint(
    checkpoint: dict,
    bucket_keys: tuple[int, ...],
    arrays: Mapping[str, np.ndarray],
) -> Phase2ScenarioConfig:
    raw = dict(checkpoint.get("scenario_cfg") or {})
    for key, cls in (
        ("kit_randomization", KitRandomization),
        ("waist_cfg", WaistAimConfig),
        ("stick_rand", StickRandomization),
    ):
        if isinstance(raw.get(key), dict):
            raw[key] = _dataclass_from_mapping(cls, raw[key])
    valid = {f.name for f in fields(Phase2ScenarioConfig)}
    raw = {k: v for k, v in raw.items() if k in valid}
    raw["buckets"] = bucket_keys
    raw["horizon"] = max(bucket_keys)
    if "joint_names" in raw:
        raw["joint_names"] = tuple(raw["joint_names"])
    if arrays.get("drum_pos") is not None:
        raw["num_drums"] = int(arrays["drum_pos"].shape[1])
    if arrays.get("future_pos") is not None:
        raw["num_future"] = int(arrays["future_pos"].shape[2])
    return Phase2ScenarioConfig(**raw)


def _load_al(cfg, checkpoint: dict) -> AugmentedLagrangian:
    al = AugmentedLagrangian(cfg.al)
    if checkpoint.get("al"):
        al.load_state_dict(migrate_al_state(checkpoint["al"]))
    return al


def _install_idle(costs, cfg, pool, kit):
    idle = resolve_idle_cylinder_library(
        cfg.resolved_idle_cylinder_path(), joint_names=pool.cfg.joint_names
    )
    if idle is None:
        return None
    try:
        cylinder = derive_cylinder(kit, costs[max(costs)].fk)
    except Exception:
        cylinder = idle.cylinder
    for cost in costs.values():
        cost.set_idle_cylinder(cylinder)
    return idle


def _replay_batch(
    *,
    pool,
    model,
    costs,
    cfg,
    al,
    bucket: int,
    batch_size: int,
    ticks: int,
    seed: int,
    step: int,
    selection: str,
    candidate_size: int,
    weights,
    amp_dtype,
    kit,
    idle,
):
    base = costs[max(costs)]
    replay = ReplayPool(
        pool,
        cfg.replay_capacity,
        buckets=[b for b in pool.buckets if b <= cfg.roller_horizon],
        recency_bias=cfg.recency_bias,
        viol_downweight=float(cfg.replay_viol_downweight),
        viol_uniform_mix=float(cfg.replay_viol_uniform_mix),
        proficiency=bool(cfg.replay_proficiency),
        ema_alpha=float(cfg.frontier_ema_alpha),
    )
    sim = BatchedPDSim(
        PDSimConfig(
            dt=pool.cfg.dt,
            saturate_effort=bool(cfg.plant_saturate_effort),
        ),
        device=pool.device,
        dtype=torch.float32,
        rnea=base.rnea,
    )
    library = A.GoalPoseLibrary()
    clips = None
    source0 = next(iter(pool.data.values()))
    if "clip_i" in source0:
        from amortized_opt.rhythm_chain_data import roller_clip_list

        clips = roller_clip_list(kit, library=library, cfg=pool.cfg)
    roller = MPCRoller(
        pool,
        model=model,
        sim=sim,
        rnea=base.rnea,
        fk=base.fk,
        horizon=cfg.roller_horizon,
        slots=cfg.roller_slots,
        chain_frac=cfg.roller_chain_frac,
        bounce=cfg.bounce,
        device=pool.device,
        dtype=torch.float32,
        seed=seed + 1301,
        amp_dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            cfg.amp_dtype
        ),
        regoal=cfg.roller_regoal,
        cmd_lead_s=cfg.cmd_lead_s,
        library=library,
        kit=kit,
        clips=clips,
        idle_library=idle,
        idle_settle_wp=cfg.idle_settle_wp,
        rest_ticks=cfg.roller_rest_ticks if idle is not None else 0,
        n_shaft=costs[min(costs)].n_points - 2,
        waist_upright_ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
    )
    stats = prime_replay_from_roller(
        roller, replay, pool, costs, cfg, al, step, ticks
    )
    available = [b for b in replay.data if replay.size(b) >= batch_size]
    if bucket not in available:
        if not available:
            raise RuntimeError(
                f"{ticks} roller ticks produced no replay bucket with "
                f"{batch_size} rows"
            )
        bucket = max(available)
    gen = torch.Generator(device=pool.device)
    gen.manual_seed(seed + 7919)
    draws = batch_size if selection == "random" else candidate_size
    candidate_indices = replay.sample_idx(
        bucket,
        draws,
        generator=gen,
        short_wp=cfg.short_wp,
        short_boost=cfg.short_boost,
        viol_downweight=cfg.replay_viol_downweight,
        viol_uniform_mix=cfg.replay_viol_uniform_mix,
        stratify=bool(cfg.replay_proficiency),
        frac_prof=float(cfg.replay_frac_prof),
        frac_new=float(cfg.replay_frac_new),
        ema_nmin=int(cfg.frontier_ema_nmin),
        ema_prof=float(cfg.frontier_ema_prof),
    )
    if selection == "random":
        indices = candidate_indices[:batch_size]
        selection_scores = None
    else:
        candidate_batch = replay.batch(
            bucket, candidate_indices, generator=gen, noise=False
        )
        candidate_hit = effective_hits(
            candidate_batch["hit"], candidate_batch["active"], bucket
        )
        candidate_lengths = group_lengths(candidate_hit)
        candidate_traj = _run_model(
            model, candidate_batch, amp_dtype, enabled=True
        ).float()
        selector = TorquePlant(
            costs[bucket], cfg.cost, weights, dtype=torch.float32
        )
        with torch.inference_mode():
            candidate_out = selector.evaluate(
                candidate_batch["q0"],
                candidate_traj,
                candidate_batch["v0"],
                candidate_lengths,
            )
        scores = candidate_out["hard_peak"].detach().cpu()
        descending = selection == "highest"
        chosen = torch.argsort(scores, descending=descending)[:batch_size]
        indices = candidate_indices.index_select(
            0, chosen.to(candidate_indices.device)
        )
        selection_scores = scores.index_select(0, chosen).tolist()
    batch = replay.batch(bucket, indices, generator=gen, noise=False)
    return batch, bucket, {
        "prime": stats,
        "sizes": {str(b): replay.size(b) for b in replay.data},
        "selection": selection,
        "candidate_size": draws,
        "selection_hard_peak_nm": selection_scores,
        "indices": indices.detach().cpu().tolist(),
    }


class TorquePlant:
    """Torque-only public-component reconstruction of ``FusedCost._torque``."""

    def __init__(self, cost, cfg, weights, *, dtype: torch.dtype):
        self.cfg = cfg
        self.device = cost.device
        self.dtype = dtype
        self.rnea = cost.rnea.to(self.device, dtype)
        self.spline = TrajectorySplineBuilder(
            max_segments=cost.max_horizon,
            dt=cost.dt,
            joint_names=cost.joint_names,
            device=self.device,
            dtype=dtype,
        )
        self.names = tuple(cost.joint_names)
        effort = scale_soft_wrist_effort(
            isaac_effort_limits(self.names),
            self.names,
            scale=cfg.soft_wrist_effort_scale,
        )
        self.tau_hard = torch.as_tensor(effort, device=self.device, dtype=dtype)
        self.v_hard = torch.as_tensor(
            isaac_velocity_limits(self.names), device=self.device, dtype=dtype
        )
        self.eta = torch.as_tensor(
            torque_limit_eta_scale(self.names, eta_torque=cfg.eta_torque),
            device=self.device,
            dtype=dtype,
        )
        self.weights = {
            key: (
                value.to(device=self.device, dtype=dtype)
                if torch.is_tensor(value)
                else torch.tensor(value, device=self.device, dtype=dtype)
            )
            for key, value in weights.items()
        }
        self.segments = [
            (
                name,
                torch.tensor(
                    [self.names.index(joint) for joint in joints],
                    device=self.device,
                    dtype=torch.long,
                ),
            )
            for name, joints in ACTUATOR_SEGMENTS
        ]

    def evaluate(
        self,
        q0: torch.Tensor,
        waypoints: torch.Tensor,
        v0: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q0 = q0.to(dtype=self.dtype)
        waypoints = waypoints.to(dtype=self.dtype)
        v0 = v0.to(dtype=self.dtype)
        sp = self.spline.build(q0, waypoints, v0, lengths)
        if self.cfg.physics_mode == "knot_fd":
            q, qd, qdd = sp.knot_finite_differences(v0)
            idx = torch.arange(
                sp.max_segments + 1, device=q.device, dtype=torch.long
            ).view(1, -1, 1)
            mask = (idx <= sp.lengths.unsqueeze(1)).to(q.dtype)
        else:
            samples = sp.subsample(self.cfg.n_sub, include_end=True)
            q, qd, qdd, mask = (
                samples.q,
                samples.qd,
                samples.qdd,
                samples.mask,
            )
        B, S, D = q.shape
        qf = q.reshape(B * S, D)
        qdf = qd.reshape(B * S, D)
        qddf = qdd.reshape(B * S, D)
        tau = self.rnea.rnea(qf, qdf, qddf).reshape(B, S, D)
        lim_hard = available_torque_torch(
            qd,
            self.tau_hard.view(1, 1, D),
            self.v_hard.view(1, 1, D),
            eta=1.0,
            saturation_scale=self.cfg.torque_saturation_scale,
            derate=self.cfg.torque_derate,
        )
        lim = self.eta.view(1, 1, D) * lim_hard
        excess = tau.abs() - lim
        hinge = F.softplus(excess / self.cfg.beta_torque) * self.cfg.beta_torque
        priced_hinge = hinge
        if "limit_alpha" in self.weights:
            priced_hinge = hinge * self.weights["limit_alpha"].view(1, 1, D)
        count = mask.sum(dim=(1, 2))
        warning_sum = (priced_hinge * mask).sum(dim=(1, 2))
        warning = warning_sum / count.clamp_min(1.0)
        over = (tau.abs() - lim_hard).clamp_min(0.0) * mask
        peak = smooth_max(
            over,
            mask,
            self.cfg.torque_peak_mode,
            self.cfg.torque_peak_p,
            self.cfg.torque_peak_temp,
            dims=(1, 2),
        )
        hard_peak = over.amax(dim=(1, 2))
        training = self.cfg.w_torque * warning
        training = training + (
            self.weights.get("torque_peak", tau.new_tensor(1.0))
            * self.cfg.w_torque_peak
            * peak
        )
        segment_peaks = []
        segment_argmax = []
        for name, index in self.segments:
            segment_over = over.index_select(-1, index)
            live = segment_over.amax(dim=(1, 2))
            segment_peaks.append(live)
            segment_argmax.append(segment_over.reshape(B, -1).argmax(dim=1))
            key = f"torque_{name}"
            training = training + self.weights.get(key, tau.new_tensor(1.0)) * live
            pen = self.weights.get(f"pen_{key}")
            if pen is not None:
                training = training + 0.5 * pen * live.square()
        pen_peak = self.weights.get("pen_torque_peak")
        if pen_peak is not None:
            training = training + 0.5 * pen_peak * peak.square()
        z = torch.zeros_like(qf)
        gravity = self.rnea.rnea(qf, z, z).reshape(B, S, D)
        velocity_bias = (
            self.rnea.rnea(qf, qdf, z).reshape(B, S, D) - gravity
        )
        inertia = (
            self.rnea.rnea(qf, z, qddf).reshape(B, S, D) - gravity
        )
        return {
            "q": q,
            "qd": qd,
            "qdd": qdd,
            "mask": mask,
            "tau": tau,
            "lim_hard": lim_hard,
            "lim": lim,
            "hinge": hinge,
            "over": over,
            "count": count,
            "warning_sum": warning_sum,
            "warning": warning,
            "peak": peak,
            "hard_peak": hard_peak,
            "segment_peaks": torch.stack(segment_peaks, dim=1),
            "segment_argmax": torch.stack(segment_argmax, dim=1),
            "training": training,
            "gravity": gravity,
            "velocity_bias": velocity_bias,
            "inertia": inertia,
            "decomposition_residual": tau - gravity - velocity_bias - inertia,
        }


def _active_signature(out: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    tau = out["tau"].detach()
    over = out["over"].detach()
    B = tau.shape[0]
    flat = over.reshape(B, -1)
    top = torch.topk(flat, k=min(2, flat.shape[1]), dim=1).values
    return {
        "over": (over > 0).detach(),
        "sign_positive": (tau >= 0).detach(),
        "global_argmax": flat.argmax(dim=1).detach(),
        "segment_argmax": out["segment_argmax"].detach(),
        "top_gap": (
            top[:, 0] - top[:, 1] if top.shape[1] > 1 else top[:, 0]
        ).detach(),
    }


def _signature_delta(
    plus: Mapping[str, Any], minus: Mapping[str, Any]
) -> dict[str, int | float]:
    return {
        "hard_threshold_crossings": int((plus["over"] != minus["over"]).sum()),
        "tau_sign_crossings": int(
            (plus["sign_positive"] != minus["sign_positive"]).sum()
        ),
        "global_argmax_changes": int(
            (plus["global_argmax"] != minus["global_argmax"]).sum()
        ),
        "segment_argmax_changes": int(
            (plus["segment_argmax"] != minus["segment_argmax"]).sum()
        ),
        "minimum_top_gap": float(
            torch.minimum(plus["top_gap"], minus["top_gap"]).min().cpu()
        ),
    }


def _directional_checks(
    plant: TorquePlant,
    *,
    q0: torch.Tensor,
    waypoints: torch.Tensor,
    v0: torch.Tensor,
    lengths: torch.Tensor,
    steps: tuple[float, ...],
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    def evaluate(x):
        ctx = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and plant.device.type == "cuda"
            else torch.autocast("cuda", enabled=False)
        )
        with ctx:
            return plant.evaluate(q0, x, v0, lengths)

    checks = {}
    objectives = ("warning", "peak", "training")
    for objective in objectives:
        x = waypoints.detach().to(dtype=plant.dtype).requires_grad_(True)
        out = evaluate(x)
        scalar = out[objective].mean()
        (grad,) = torch.autograd.grad(scalar, x)
        scale = grad.detach().abs().max().clamp_min(1e-30)
        direction = grad.detach() / scale
        autograd_dd = (grad.detach() * direction).sum()
        rows = []
        for h in steps:
            with torch.no_grad():
                plus_out = evaluate(x + h * direction)
                minus_out = evaluate(x - h * direction)
                f0 = scalar.detach()
                fp = plus_out[objective].mean()
                fm = minus_out[objective].mean()
                central = (fp - fm) / (2.0 * h)
                forward = (fp - f0) / h
                backward = (f0 - fm) / h
                sig = _signature_delta(
                    _active_signature(plus_out), _active_signature(minus_out)
                )
            denom = autograd_dd.abs().clamp_min(1e-30)
            rows.append(
                {
                    "step_rad": h,
                    "autograd": float(autograd_dd),
                    "central": float(central),
                    "forward": float(forward),
                    "backward": float(backward),
                    "central_relative_error": float(
                        (central - autograd_dd).abs() / denom
                    ),
                    "one_sided_spread_relative": float(
                        (forward - backward).abs() / denom
                    ),
                    **sig,
                }
            )
        checks[objective] = {
            "value": float(scalar.detach()),
            "gradient_max_abs": float(grad.detach().abs().max()),
            "gradient_rms": float(grad.detach().square().mean().sqrt()),
            "direction_max_abs": float(direction.abs().max()),
            "steps": rows,
        }
    return checks


def _rnea_directional_check(
    plant: TorquePlant,
    out: Mapping[str, torch.Tensor],
    steps: tuple[float, ...],
) -> dict[str, Any]:
    flat_over = out["over"].detach().reshape(-1)
    top = torch.topk(flat_over, k=min(8, flat_over.numel())).indices
    D = out["tau"].shape[-1]
    state = top // D
    q = out["q"].detach().reshape(-1, D).index_select(0, state).to(plant.dtype)
    qd = out["qd"].detach().reshape(-1, D).index_select(0, state).to(plant.dtype)
    qdd = out["qdd"].detach().reshape(-1, D).index_select(0, state).to(plant.dtype)
    q.requires_grad_(True)
    qd.requires_grad_(True)
    qdd.requires_grad_(True)
    gen = torch.Generator(device=plant.device)
    gen.manual_seed(12345)
    probe = torch.randn(
        q.shape, generator=gen, device=plant.device, dtype=plant.dtype
    )
    scalar = (plant.rnea.rnea(q, qd, qdd) * probe).mean()
    grads = torch.autograd.grad(scalar, (q, qd, qdd))
    norm = torch.stack([g.abs().max() for g in grads]).max().clamp_min(1e-30)
    directions = tuple(g.detach() / norm for g in grads)
    ad = sum((g.detach() * d).sum() for g, d in zip(grads, directions))
    rows = []
    for h in steps:
        with torch.no_grad():
            fp = (
                plant.rnea.rnea(
                    q + h * directions[0],
                    qd + h * directions[1],
                    qdd + h * directions[2],
                )
                * probe
            ).mean()
            fm = (
                plant.rnea.rnea(
                    q - h * directions[0],
                    qd - h * directions[1],
                    qdd - h * directions[2],
                )
                * probe
            ).mean()
            fd = (fp - fm) / (2.0 * h)
        rows.append(
            {
                "step": h,
                "autograd": float(ad),
                "central": float(fd),
                "relative_error": float((fd - ad).abs() / ad.abs().clamp_min(1e-30)),
            }
        )
    return {
        "states": int(q.shape[0]),
        "value": float(scalar.detach()),
        "gradient_max_abs": float(norm),
        "steps": rows,
    }


def _curvature_check(
    plant: TorquePlant,
    *,
    q0: torch.Tensor,
    waypoints: torch.Tensor,
    v0: torch.Tensor,
    lengths: torch.Tensor,
    steps: tuple[float, ...],
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    def scalar(x):
        ctx = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and plant.device.type == "cuda"
            else torch.autocast("cuda", enabled=False)
        )
        with ctx:
            return plant.evaluate(q0, x, v0, lengths)["training"].mean()

    x = waypoints.detach().to(plant.dtype).requires_grad_(True)
    f0 = scalar(x)
    (grad,) = torch.autograd.grad(f0, x, create_graph=True)
    direction = grad.detach() / grad.detach().abs().max().clamp_min(1e-30)
    (hvp,) = torch.autograd.grad(grad, x, grad_outputs=direction)
    d_hd = (direction * hvp).sum().detach()
    rows = []
    for h in steps:
        with torch.no_grad():
            fp = scalar(x + h * direction)
            fm = scalar(x - h * direction)
            fd = (fp - 2.0 * f0.detach() + fm) / (h * h)
        rows.append(
            {
                "step_rad": h,
                "hvp": float(d_hd),
                "central_second": float(fd),
                "relative_error": float(
                    (fd - d_hd).abs() / d_hd.abs().clamp_min(1e-30)
                ),
            }
        )
    return {
        "value": float(f0.detach()),
        "hvp_finite": bool(torch.isfinite(d_hd)),
        "dHd": float(d_hd),
        "steps": rows,
    }


def _model_torque_gradients(
    model,
    batch,
    plant: TorquePlant,
    lengths: torch.Tensor,
    amp_dtype: torch.dtype | None,
    *,
    model_amp_enabled: bool,
    physics_amp_enabled: bool,
    traj_head_fp32: bool | None = None,
) -> tuple[dict[str, Any], list[torch.Tensor | None]]:
    model.zero_grad(set_to_none=True)
    prior_head_mode = bool(model.cfg.traj_head_fp32)
    if traj_head_fp32 is not None:
        model.cfg.traj_head_fp32 = bool(traj_head_fp32)
    model_ctx = (
        torch.autocast("cuda", dtype=amp_dtype)
        if model_amp_enabled
        and amp_dtype is not None
        and plant.device.type == "cuda"
        else torch.autocast("cuda", enabled=False)
    )
    try:
        with model_ctx:
            enc = model.encode_context(batch["context"])
            trajectory = model(
                batch["init"], batch["state"], enc, batch["masks"]
            ).traj
    finally:
        model.cfg.traj_head_fp32 = prior_head_mode
    physics_ctx = (
        torch.autocast("cuda", dtype=amp_dtype)
        if physics_amp_enabled
        and amp_dtype is not None
        and plant.device.type == "cuda"
        else torch.autocast("cuda", enabled=False)
    )
    with physics_ctx:
        scalar = plant.evaluate(
            batch["q0"], trajectory, batch["v0"], lengths
        )["training"].mean()
    named = [(name, parameter) for name, parameter in model.named_parameters()]
    grads_raw = torch.autograd.grad(
        scalar,
        [parameter for _name, parameter in named],
        allow_unused=True,
    )
    grads = [
        None if grad is None else grad.detach().clone() for grad in grads_raw
    ]
    nonfinite = {
        name: int((~torch.isfinite(grad)).sum().item())
        for (name, _parameter), grad in zip(named, grads)
        if grad is not None and not bool(torch.isfinite(grad).all())
    }
    norm_sq = sum(
        grad.double().square().sum()
        for grad in grads
        if grad is not None and bool(torch.isfinite(grad).all())
    )
    top = sorted(
        (
            {
                "parameter": name,
                "max_abs": float(grad.abs().max()),
                "rms": float(grad.float().square().mean().sqrt()),
            }
            for (name, _parameter), grad in zip(named, grads)
            if grad is not None
        ),
        key=lambda row: row["max_abs"],
        reverse=True,
    )[:20]
    return (
        {
            "value": float(scalar.detach()),
            "gradient_norm": float(norm_sq.sqrt()),
            "nonfinite_parameters": nonfinite,
            "unused_parameters": sum(grad is None for grad in grads),
            "top_parameter_gradients": top,
        },
        grads,
    )


def _gradient_comparison(
    got: list[torch.Tensor | None], ref: list[torch.Tensor | None]
) -> dict[str, float | None]:
    dot = torch.zeros((), dtype=torch.float64, device=next(
        grad.device for grad in got if grad is not None
    ))
    got_sq = torch.zeros_like(dot)
    ref_sq = torch.zeros_like(dot)
    diff_sq = torch.zeros_like(dot)
    for left, right in zip(got, ref):
        if left is None or right is None:
            continue
        a, b = left.double(), right.double()
        dot = dot + (a * b).sum()
        got_sq = got_sq + a.square().sum()
        ref_sq = ref_sq + b.square().sum()
        diff_sq = diff_sq + (a - b).square().sum()
    ng, nr = got_sq.sqrt(), ref_sq.sqrt()
    return {
        "cosine": float(dot / (ng * nr).clamp_min(1e-30)),
        "norm_ratio": float(ng / nr.clamp_min(1e-30)),
        "relative_l2_difference": float(diff_sq.sqrt() / nr.clamp_min(1e-30)),
    }


def _localization(
    out: Mapping[str, torch.Tensor], names: tuple[str, ...], top_k: int = 16
) -> dict[str, Any]:
    over = out["over"].detach()
    B, S, D = over.shape
    values, indices = torch.topk(over.reshape(-1), k=min(top_k, over.numel()))
    rows = []
    for value, flat in zip(values.cpu(), indices.cpu()):
        flat = int(flat)
        b, rem = divmod(flat, S * D)
        s, j = divmod(rem, D)
        rows.append(
            {
                "excess_nm": float(value),
                "batch": b,
                "sample": s,
                "joint_index": j,
                "joint": names[j],
                "tau_nm": float(out["tau"][b, s, j].detach().cpu()),
                "limit_nm": float(
                    out["lim_hard"].expand_as(out["tau"])[b, s, j].detach().cpu()
                ),
                "qd_rad_s": float(out["qd"][b, s, j].detach().cpu()),
                "qdd_rad_s2": float(out["qdd"][b, s, j].detach().cpu()),
                "gravity_nm": float(out["gravity"][b, s, j].detach().cpu()),
                "velocity_bias_nm": float(
                    out["velocity_bias"][b, s, j].detach().cpu()
                ),
                "inertia_nm": float(out["inertia"][b, s, j].detach().cpu()),
            }
        )
    per_joint = {}
    for j, name in enumerate(names):
        per_joint[name] = {
            "max_abs_tau_nm": float(
                (out["tau"][..., j].abs() * out["mask"][..., j]).max().detach().cpu()
            ),
            "max_excess_nm": float(over[..., j].max().detach().cpu()),
        }
    per_sample = over.amax(dim=(0, 2)).detach().cpu()
    return {
        "top_excess": rows,
        "per_joint": per_joint,
        "top_samples": [
            {"sample": int(i), "max_excess_nm": float(per_sample[i])}
            for i in torch.argsort(per_sample, descending=True)[: min(top_k, S)]
        ],
    }


def _component_report(
    plant: TorquePlant, out: Mapping[str, torch.Tensor], top_states: int = 16
) -> dict[str, Any]:
    mask = out["mask"].detach().bool()
    residual = out["decomposition_residual"][mask]
    report = {
        "gravity": _stats(out["gravity"].abs()[mask]),
        "velocity_bias": _stats(out["velocity_bias"].abs()[mask]),
        "inertia": _stats(out["inertia"].abs()[mask]),
        "full": _stats(out["tau"].abs()[mask]),
        "decomposition_residual": _stats(residual.abs()),
    }
    B, S, D = out["tau"].shape
    flat = out["over"].detach().reshape(-1)
    indices = torch.topk(flat, k=min(top_states, flat.numel())).indices
    state_indices = torch.unique(indices // D)
    q = out["q"].detach().reshape(B * S, D).index_select(0, state_indices)
    qdd = out["qdd"].detach().reshape(B * S, D).index_select(0, state_indices)
    M = mass_matrix(plant.rnea, q)
    diag = torch.diagonal(M, dim1=-2, dim2=-1) * qdd
    inertia = out["inertia"].detach().reshape(B * S, D).index_select(
        0, state_indices
    )
    coupling = inertia - diag
    gravity = out["gravity"].detach().reshape(B * S, D).index_select(
        0, state_indices
    )
    level_a = gravity + diag
    full = out["tau"].detach().reshape(B * S, D).index_select(0, state_indices)
    report["high_excess_states"] = {
        "states": int(state_indices.numel()),
        "diag_inertia": _stats(diag.abs()),
        "coupled_inertia": _stats(coupling.abs()),
        "level_a_abs_error_vs_full": _stats((level_a - full).abs()),
        "level_a_relative_rms_vs_full": float(
            (level_a - full).square().mean().sqrt()
            / full.square().mean().sqrt().clamp_min(1e-30)
        ),
    }
    return report


def _normalization_report(
    plant: TorquePlant,
    q0: torch.Tensor,
    waypoints: torch.Tensor,
    v0: torch.Tensor,
    lengths: torch.Tensor,
    out: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    # Fixed-state masks isolate count normalization from a changed spline end BC.
    B, S, D = out["hinge"].shape
    ks = sorted(
        {
            2,
            3,
            5,
            8,
            12,
            min(25, S - 1),
            S - 1,
        }
    )
    fixed = []
    rebuilt = []
    row = int(out["hard_peak"].argmax().item())
    for k in ks:
        prefix = (
            torch.arange(S, device=plant.device).view(1, S, 1) <= int(k)
        ).to(plant.dtype)
        fixed_mask = out["mask"][row : row + 1] * prefix
        count = fixed_mask.sum()
        total = (out["hinge"][row : row + 1] * fixed_mask).sum()
        fixed.append(
            {
                "segments": k,
                "count": float(count),
                "warning_sum": float(total),
                "warning_mean": float(total / count.clamp_min(1.0)),
                "hard_peak": float(
                    (out["over"][row : row + 1] * fixed_mask).max()
                ),
            }
        )
        lens = torch.minimum(
            lengths[row : row + 1],
            torch.full(
                (1, 3), k, device=plant.device, dtype=torch.long
            ),
        )
        x = waypoints[row : row + 1].detach().to(plant.dtype).requires_grad_(True)
        rebuilt_out = plant.evaluate(
            q0[row : row + 1], x, v0[row : row + 1], lens
        )
        scalar = rebuilt_out["training"].mean()
        (grad,) = torch.autograd.grad(scalar, x)
        rebuilt.append(
            {
                "segments": k,
                "count": float(rebuilt_out["count"]),
                "warning_mean": float(rebuilt_out["warning"]),
                "hard_peak": float(rebuilt_out["hard_peak"]),
                "training": float(scalar.detach()),
                "gradient_norm": float(grad.detach().square().sum().sqrt()),
            }
        )
    return {"selected_row": row, "fixed_states": fixed, "rebuilt_spline": rebuilt}


def _compile_report(
    plant: TorquePlant, out: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    B, S, D = out["q"].shape
    q = out["q"].detach().reshape(B * S, D)[:32].clone().requires_grad_(True)
    qd = out["qd"].detach().reshape(B * S, D)[:32].clone().requires_grad_(True)
    qdd = out["qdd"].detach().reshape(B * S, D)[:32].clone().requires_grad_(True)

    def fn(a, b, c):
        return plant.rnea.rnea(a, b, c)

    eager = fn(q, qd, qdd)
    eager_grads = torch.autograd.grad(eager.square().mean(), (q, qd, qdd))
    compiled_fn = torch.compile(fn, mode="default", dynamic=False)
    qc = q.detach().clone().requires_grad_(True)
    qdc = qd.detach().clone().requires_grad_(True)
    qddc = qdd.detach().clone().requires_grad_(True)
    compiled = compiled_fn(qc, qdc, qddc)
    compiled_grads = torch.autograd.grad(
        compiled.square().mean(), (qc, qdc, qddc)
    )
    return {
        "forward": _error_stats(compiled, eager),
        "grad_q": _error_stats(compiled_grads[0], eager_grads[0]),
        "grad_qd": _error_stats(compiled_grads[1], eager_grads[1]),
        "grad_qdd": _error_stats(compiled_grads[2], eager_grads[2]),
        "compiled_finite": all(
            bool(torch.isfinite(x).all())
            for x in (compiled, *compiled_grads)
        ),
    }


def _run_model(
    model,
    batch,
    amp_dtype,
    *,
    enabled: bool,
    traj_head_fp32: bool | None = None,
):
    device = batch["q0"].device
    ctx = (
        torch.autocast("cuda", dtype=amp_dtype)
        if enabled and amp_dtype is not None and device.type == "cuda"
        else torch.autocast("cuda", enabled=False)
    )
    prior_head_mode = bool(model.cfg.traj_head_fp32)
    if traj_head_fp32 is not None:
        model.cfg.traj_head_fp32 = bool(traj_head_fp32)
    try:
        with torch.inference_mode(), ctx:
            enc = model.encode_context(batch["context"])
            out = model(batch["init"], batch["state"], enc, batch["masks"])
    finally:
        model.cfg.traj_head_fp32 = prior_head_mode
    return out.traj.detach()


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        torch.cuda.set_device(device.index or 0)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = Path(args.checkpoint)
    pool_path = Path(args.pool_cache)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = phase2_config_from_checkpoint(
        checkpoint, device=str(device), compile_model=False
    )
    cfg.compile_mode = None
    cfg.compile_model = False
    cfg.compile_cost = False
    bucket_arrays, pool_stats = _load_npz_buckets(pool_path)
    bucket_keys = tuple(sorted(int(k) for k in bucket_arrays))
    bucket = int(args.bucket if args.bucket is not None else max(bucket_keys))
    if bucket not in bucket_arrays:
        raise SystemExit(f"bucket {bucket} not in {bucket_keys}")
    scenario = _scenario_from_checkpoint(
        checkpoint, bucket_keys, bucket_arrays[bucket]
    )
    pool = Phase2Pool(
        bucket_arrays,
        cfg=scenario,
        noise=Phase2Noise(),
        device=device,
        dtype=torch.float32,
        stats=pool_stats,
    )
    model_cfg = A.model_config_from_checkpoint(checkpoint)
    model = A.build_model(model_cfg, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    costs = build_costs(
        bucket_keys, cfg.cost, device=device, dtype=torch.float32
    )
    kit = DrumKit.from_layout()
    idle = _install_idle(costs, cfg, pool, kit)
    al = _load_al(cfg, checkpoint)
    step = int(checkpoint.get("step") or 0)
    weights32 = _wtensors(
        cfg.schedule.multipliers(step, False),
        al.weights(),
        device,
        torch.float32,
        sched=cfg.schedule,
        joint_names=costs[bucket].joint_names,
        step=step,
        stage_a=False,
        pen=al.pen,
    )
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        cfg.amp_dtype
    )

    source_meta: dict[str, Any]
    if args.source == "pool":
        rng = np.random.default_rng(args.seed)
        candidate_size = (
            args.batch_size
            if args.selection == "random"
            else min(args.candidate_size, pool.size(bucket))
        )
        candidate_np = rng.choice(
            pool.size(bucket),
            size=candidate_size,
            replace=candidate_size > pool.size(bucket),
        )
        if args.selection == "random":
            indices_np = candidate_np[: args.batch_size]
            selection_scores = None
        else:
            candidate_indices = torch.as_tensor(
                candidate_np, device=device, dtype=torch.long
            )
            candidate_batch = pool.batch(
                bucket, candidate_indices, noise=False
            )
            candidate_hit = effective_hits(
                candidate_batch["hit"], candidate_batch["active"], bucket
            )
            candidate_lengths = group_lengths(candidate_hit)
            candidate_traj = _run_model(
                model, candidate_batch, amp_dtype, enabled=True
            ).float()
            selector_plant = TorquePlant(
                costs[bucket], cfg.cost, weights32, dtype=torch.float32
            )
            with torch.inference_mode():
                candidate_out = selector_plant.evaluate(
                    candidate_batch["q0"],
                    candidate_traj,
                    candidate_batch["v0"],
                    candidate_lengths,
                )
            scores = candidate_out["hard_peak"].detach().cpu()
            descending = args.selection == "highest"
            chosen = torch.argsort(scores, descending=descending)[: args.batch_size]
            indices_np = candidate_np[chosen.numpy()]
            selection_scores = scores.index_select(0, chosen).tolist()
        indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)
        batch = pool.batch(bucket, indices, noise=False)
        source_meta = {
            "selection": args.selection,
            "candidate_size": candidate_size,
            "indices": indices_np.tolist(),
            "selection_hard_peak_nm": selection_scores,
        }
    else:
        batch, bucket, source_meta = _replay_batch(
            pool=pool,
            model=model,
            costs=costs,
            cfg=cfg,
            al=al,
            bucket=bucket,
            batch_size=args.batch_size,
            ticks=args.prime_roller_ticks,
            seed=args.seed,
            step=step,
            selection=args.selection,
            candidate_size=args.candidate_size,
            weights=weights32,
            amp_dtype=amp_dtype,
            kit=kit,
            idle=idle,
        )
        # Replay may choose a fallback bucket.
        weights32 = _wtensors(
            cfg.schedule.multipliers(step, False),
            al.weights(),
            device,
            torch.float32,
            sched=cfg.schedule,
            joint_names=costs[bucket].joint_names,
            step=step,
            stage_a=False,
            pen=al.pen,
        )

    hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
    lengths = group_lengths(hit_eff)
    strike = replace(batch["strike"], hit_index=hit_eff)
    traj_amp = _run_model(
        model, batch, amp_dtype, enabled=True, traj_head_fp32=False
    )
    traj_amp_fp32_head = _run_model(
        model,
        batch,
        amp_dtype,
        enabled=True,
        traj_head_fp32=True,
    )
    traj_fp32 = _run_model(
        model, batch, amp_dtype, enabled=False, traj_head_fp32=False
    )

    plant32 = TorquePlant(costs[bucket], cfg.cost, weights32, dtype=torch.float32)
    plant64 = TorquePlant(costs[bucket], cfg.cost, weights32, dtype=torch.float64)
    q0, v0 = batch["q0"], batch["v0"]
    # ``inference_mode`` tensors cannot later become autograd leaves even after
    # a dtype-preserving ``to``.  Clone back into an ordinary tensor here.
    actual = traj_amp.float().clone()
    traj_amp_fp32_head = traj_amp_fp32_head.float().clone()
    traj_fp32 = traj_fp32.float().clone()
    out32 = plant32.evaluate(q0, actual, v0, lengths)
    out64 = plant64.evaluate(
        q0.double(), actual.double(), v0.double(), lengths
    )
    with (
        torch.autocast("cuda", dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else torch.autocast("cuda", enabled=False)
    ):
        out_amp_physics = plant32.evaluate(q0, actual, v0, lengths)
    out_model32 = plant32.evaluate(q0, traj_fp32.float(), v0, lengths)
    out_amp_fp32_head = plant32.evaluate(
        q0, traj_amp_fp32_head, v0, lengths
    )
    quantized = traj_fp32.to(torch.bfloat16).float()
    out_quantized = plant32.evaluate(q0, quantized, v0, lengths)

    # Exact torque-term reconstruction check against the full FusedCost.
    with torch.inference_mode():
        full_ct = costs[bucket].forward(
            q0,
            actual,
            v0,
            strike,
            lengths=lengths,
            elem_mask=batch["masks"].elem_mask,
            weights=weights32,
        )
    reconstruction = _error_stats(
        out32["training"], full_ct.terms["torque"]
    )

    precision = {
        key: _error_stats(out32[key], out64[key])
        for key in ("q", "qd", "qdd", "tau", "warning", "peak", "training")
    }
    # Isolate RNEA arithmetic: feed identical fp64-built states to each RNEA.
    B, S, D = out64["q"].shape
    tau64_same = out64["tau"]
    tau32_same = plant32.rnea.rnea(
        out64["q"].float().reshape(B * S, D),
        out64["qd"].float().reshape(B * S, D),
        out64["qdd"].float().reshape(B * S, D),
    ).reshape(B, S, D)
    precision["rnea_only_same_states"] = _error_stats(tau32_same, tau64_same)
    amp_effects = {
        "model_autocast_trajectory": _error_stats(traj_amp.float(), traj_fp32.float()),
        "model_fp32_then_bf16_quantized_trajectory": _error_stats(
            quantized, traj_fp32.float()
        ),
        "physics_autocast_tau_same_trajectory": _error_stats(
            out_amp_physics["tau"], out32["tau"]
        ),
        "physics_autocast_training_same_trajectory": _error_stats(
            out_amp_physics["training"], out32["training"]
        ),
        "actual_amp_model_vs_fp32_model_tau": _error_stats(
            out32["tau"], out_model32["tau"]
        ),
        "actual_amp_model_vs_fp32_model_training": _error_stats(
            out32["training"], out_model32["training"]
        ),
        "fp32_head_trajectory_vs_all_fp32_model": _error_stats(
            traj_amp_fp32_head, traj_fp32
        ),
        "fp32_head_tau_vs_all_fp32_model": _error_stats(
            out_amp_fp32_head["tau"], out_model32["tau"]
        ),
        "fp32_head_training_vs_all_fp32_model": _error_stats(
            out_amp_fp32_head["training"], out_model32["training"]
        ),
        "quantization_only_tau": _error_stats(
            out_quantized["tau"], out_model32["tau"]
        ),
    }
    model_grad_amp, grads_amp = _model_torque_gradients(
        model,
        batch,
        plant32,
        lengths,
        amp_dtype,
        model_amp_enabled=True,
        physics_amp_enabled=True,
        traj_head_fp32=False,
    )
    model_grad_model_amp, grads_model_amp = _model_torque_gradients(
        model,
        batch,
        plant32,
        lengths,
        amp_dtype,
        model_amp_enabled=True,
        physics_amp_enabled=False,
        traj_head_fp32=False,
    )
    model_grad_physics_amp, grads_physics_amp = _model_torque_gradients(
        model,
        batch,
        plant32,
        lengths,
        amp_dtype,
        model_amp_enabled=False,
        physics_amp_enabled=True,
        traj_head_fp32=False,
    )
    model_grad_fp32, grads_fp32 = _model_torque_gradients(
        model,
        batch,
        plant32,
        lengths,
        amp_dtype,
        model_amp_enabled=False,
        physics_amp_enabled=False,
        traj_head_fp32=False,
    )
    model_grad_fp32_head, grads_fp32_head = _model_torque_gradients(
        model,
        batch,
        plant32,
        lengths,
        amp_dtype,
        model_amp_enabled=True,
        physics_amp_enabled=False,
        traj_head_fp32=True,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic": "full_rnea_numerical_autograd_audit",
        "meta": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": step,
            "pool_cache": str(pool_path),
            "source": args.source,
            "source_meta": source_meta,
            "bucket": bucket,
            "batch_size": int(actual.shape[0]),
            "seed": args.seed,
            "device": str(device),
            "torch_version": torch.__version__,
            "amp_dtype": cfg.amp_dtype,
            "physics_mode": cfg.cost.physics_mode,
            "torque_physics": cfg.cost.torque_physics,
            "torque_derate": cfg.cost.torque_derate,
            "w_torque": cfg.cost.w_torque,
            "w_torque_peak": cfg.cost.w_torque_peak,
            "tensor_sha256": _digest(q0, v0, actual, lengths),
            "trajectory_dtypes": {
                "train_amp_output": str(traj_amp.dtype).replace("torch.", ""),
                "fp32_model_output": str(traj_fp32.dtype).replace("torch.", ""),
            },
        },
        "effective_torque_weights": {
            k: _number(v)
            for k, v in sorted(weights32.items())
            if k == "torque_peak"
            or k.startswith("torque_")
            or k.startswith("pen_torque")
            or k == "limit_alpha"
        },
        "torque_reconstruction_vs_fused_cost": reconstruction,
        "float32_vs_float64": precision,
        "amp_boundaries": amp_effects,
        "model_parameter_torque_gradients": {
            "model_amp_physics_amp": model_grad_amp,
            "model_amp_physics_fp32": model_grad_model_amp,
            "model_fp32_physics_amp": model_grad_physics_amp,
            "model_fp32_physics_fp32": model_grad_fp32,
            "model_amp_fp32_head_physics_fp32": model_grad_fp32_head,
            "comparisons_to_all_fp32": {
                "all_amp": _gradient_comparison(grads_amp, grads_fp32),
                "model_amp_only": _gradient_comparison(
                    grads_model_amp, grads_fp32
                ),
                "physics_amp_only": _gradient_comparison(
                    grads_physics_amp, grads_fp32
                ),
                "model_amp_fp32_head": _gradient_comparison(
                    grads_fp32_head, grads_fp32
                ),
            },
        },
        "nan_trace": _finite_stage_map(
            {
                key: out32[key]
                for key in (
                    "q",
                    "qd",
                    "qdd",
                    "tau",
                    "lim_hard",
                    "hinge",
                    "over",
                    "warning",
                    "peak",
                    "training",
                    "gravity",
                    "velocity_bias",
                    "inertia",
                )
            }
        ),
        "localization": _localization(out32, tuple(G1_UPPER_BODY_JOINTS)),
        "components": _component_report(plant32, out32),
        "directional_derivatives_fp64": _directional_checks(
            plant64,
            q0=q0.double(),
            waypoints=actual.double(),
            v0=v0.double(),
            lengths=lengths,
            steps=tuple(args.fd_steps),
        ),
        "directional_derivatives_fp32": _directional_checks(
            plant32,
            q0=q0,
            waypoints=actual,
            v0=v0,
            lengths=lengths,
            steps=tuple(args.fd_steps),
        ),
        "directional_derivatives_amp_physics": _directional_checks(
            plant32,
            q0=q0,
            waypoints=actual,
            v0=v0,
            lengths=lengths,
            steps=tuple(args.fd_steps),
            autocast_dtype=amp_dtype,
        ),
        "rnea_directional_derivative_fp64": _rnea_directional_check(
            plant64, out64, tuple(args.fd_steps)
        ),
        "curvature_fp64": _curvature_check(
            plant64,
            q0=q0.double(),
            waypoints=actual.double(),
            v0=v0.double(),
            lengths=lengths,
            steps=tuple(args.fd_steps),
        ),
        "curvature_fp32": _curvature_check(
            plant32,
            q0=q0,
            waypoints=actual,
            v0=v0,
            lengths=lengths,
            steps=tuple(args.fd_steps),
        ),
        "curvature_amp_physics": _curvature_check(
            plant32,
            q0=q0,
            waypoints=actual,
            v0=v0,
            lengths=lengths,
            steps=tuple(args.fd_steps),
            autocast_dtype=amp_dtype,
        ),
        "horizon_and_normalization": _normalization_report(
            plant32, q0, actual, v0, lengths, out32
        ),
    }
    if args.compile:
        report["compile_vs_eager_rnea"] = _compile_report(plant32, out32)
    else:
        report["compile_vs_eager_rnea"] = {
            "skipped": (
                "training has compile_cost=False; pass --compile to test isolated RNEA"
            )
        }

    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"[json] {output}")
    print(
        f"[audit] source={args.source} checkpoint_step={step} bucket={bucket} "
        f"B={actual.shape[0]} max_excess={float(out32['hard_peak'].max()):.4g} N*m"
    )
    print(
        "[precision] tau fp32/fp64 rel-rms="
        f"{precision['tau']['relative_rms']:.3e}; "
        "same-state RNEA rel-rms="
        f"{precision['rnea_only_same_states']['relative_rms']:.3e}"
    )
    for dtype_key in ("directional_derivatives_fp64", "directional_derivatives_fp32"):
        best = min(
            row["central_relative_error"]
            for row in report[dtype_key]["training"]["steps"]
        )
        print(f"[{dtype_key}] best training directional rel-error={best:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
