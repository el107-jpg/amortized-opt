#!/usr/bin/env python
"""Train the amortized terminal-pose network by backprop through kinematics.

No expert, no IK teacher -- sample a batch, predict q, price with
:class:`~amortized_opt.terminal_pose.TerminalPoseCost`, step Adam, then
update the hit/collision dual.

    python scripts/train_terminal_pose.py --steps 2000 --device cpu
    python scripts/train_terminal_pose.py --steps 1 --device cpu   # smoke
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amortized_opt.line_search import (  # noqa: E402
    LineSearchConfig,
    check_early_triggers,
    latch_triggers,
)
from amortized_opt.line_search_step import (  # noqa: E402
    init_line_search_runtime,
    needs_line_search_direction,
    resolve_known_search,
    runtime_state_dict,
    run_search_or_ordinary,
    snapshot_g_preclip,
)
from amortized_opt.starts import StartSampler  # noqa: E402
from amortized_opt.terminal_pose import (  # noqa: E402
    TerminalPoseAL,
    TerminalPoseConfig,
    TerminalPoseCost,
    TerminalPoseNet,
    collision_violation,
    sample_terminal_batch,
)

LOSS_SPEC = (
    "L = w_joint*joint + w_start*start + w_motion*motion"
    " + w_com*com + w_elev_drum*elev_drum + w_elev_cymbal*elev_cymbal"
    " + w_shaft*shaft + w_waist*waist + w_elev_floor*elev_floor"
    " + lam_hit*v_hit + (pen_hit/2)*v_hit^2"
    " + lam_col*v_col + (pen_col/2)*v_col^2"
)

#: Detached aux diagnostics the loop prints when present.
_AUX_LOG_KEYS = (
    ("elev", ("stick_elev_deg", "elev", "elev_drum")),
    ("stick_z", ("stick_z", "elev_cymbal")),
    ("shaft_s", ("shaft_s", "shaft")),
    ("com_z", ("arm_com_z", "com_z", "com")),
)


@dataclass
class TrainConfig:
    steps: int = 2000
    batch: int = 64
    lr: float = 1e-3
    #: Cosine floor.  ``None`` keeps ``lr`` constant.  At step ``steps`` the
    #: rate is this value; step 1 is ``lr``.
    lr_end: float | None = None
    device: str = "cpu"
    seed: int = 0
    log_every: int = 50
    al_every: int = 1
    grad_clip: float = 1.0
    width: int = 128
    depth: int = 2
    w_start: float = 10.0
    w_motion: float = 0.02
    collision_topk: int = 1
    start_bank_size: int = 4096
    optnet: bool = False
    n_constraints: int = 4
    centered_qp: bool = False
    mix_width: int | None = None
    qp_out: bool = False
    line_search: bool = False
    line_search_interval: int | None = None
    resume: str | None = None
    out: str = "runs/terminal_pose/last.pt"


def parse_args(argv=None) -> TrainConfig:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--lr-end",
        type=float,
        default=None,
        help="cosine floor at --steps; omit to keep --lr constant",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--al-every",
        type=int,
        default=1,
        help="dual update period (after the Adam step), like the trajectory optimizer",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--w-start", type=float, default=10.0)
    p.add_argument("--w-motion", type=float, default=0.02)
    p.add_argument("--collision-topk", type=int, default=1)
    p.add_argument(
        "--start-bank-size", type=int, default=4096,
        help="precomputed StartSampler states; 0 uses the cheap fallback sampler",
    )
    p.add_argument(
        "--optnet",
        action="store_true",
        help="quadratic-program layer between the two hidden maps",
    )
    p.add_argument(
        "--centered-qp",
        action="store_true",
        help="earlier quadratic layer: preferred pose, joint box, four equalities. Implies --optnet",
    )
    p.add_argument(
        "--n-constraints",
        type=int,
        default=4,
        help="inequality rows and equality rows in the learned QP",
    )
    p.add_argument(
        "--mix-width",
        type=int,
        default=None,
        help="SiLU width that mixes the QP solution with 8 skipped channels. Implies --optnet",
    )
    p.add_argument(
        "--qp-out",
        action="store_true",
        help="two hidden SiLU maps, then the QP solution is the pose. Implies --optnet",
    )
    p.add_argument(
        "--line-search",
        action="store_true",
        help="SaLSa step-size search. --lr is the search seed; cosine is not applied",
    )
    p.add_argument(
        "--line-search-interval",
        type=int,
        default=None,
        help=(
            "fixed SaLSa search cadence; disables unscheduled early-trigger searches "
            "and implies --line-search"
        ),
    )
    p.add_argument(
        "--resume",
        default=None,
        help="checkpoint to load (weights and augmented-Lagrangian state)",
    )
    p.add_argument("--out", default="runs/terminal_pose/last.pt")
    ns = p.parse_args(argv)
    return TrainConfig(
        steps=int(ns.steps),
        batch=int(ns.batch),
        lr=float(ns.lr),
        lr_end=None if ns.lr_end is None else float(ns.lr_end),
        device=str(ns.device),
        seed=int(ns.seed),
        log_every=int(ns.log_every),
        al_every=int(ns.al_every),
        grad_clip=float(ns.grad_clip),
        width=int(ns.width),
        depth=int(ns.depth),
        w_start=float(ns.w_start),
        w_motion=float(ns.w_motion),
        collision_topk=int(ns.collision_topk),
        start_bank_size=int(ns.start_bank_size),
        optnet=bool(ns.optnet or ns.centered_qp or ns.mix_width is not None or ns.qp_out),
        n_constraints=int(ns.n_constraints),
        centered_qp=bool(ns.centered_qp),
        mix_width=None if ns.mix_width is None else int(ns.mix_width),
        qp_out=bool(ns.qp_out),
        line_search=bool(ns.line_search or ns.line_search_interval is not None),
        line_search_interval=(
            None if ns.line_search_interval is None else int(ns.line_search_interval)
        ),
        resume=None if ns.resume is None else str(ns.resume),
        out=str(ns.out),
    )


def _aux_mean(aux: dict, aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        if key not in aux:
            continue
        t = aux[key]
        if torch.is_tensor(t):
            return float(t.detach().float().mean().item())
        return float(t)
    return None


def _al_checkpoint(al) -> dict:
    if hasattr(al, "state_dict") and callable(al.state_dict):
        return {"al": al.state_dict()}
    return {"lam": dict(al.lam), "pen": dict(al.pen)}


def _config_dict(cfg) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {}


def _sample_batch(n, rng, device, start_bank=None):
    kwargs = {}
    if start_bank is not None:
        q0_bank, qd0_bank = start_bank
        idx = torch.as_tensor(
            rng.integers(0, q0_bank.shape[0], size=int(n)),
            device=q0_bank.device,
        )
        kwargs = {
            "q_start": q0_bank.index_select(0, idx),
            "qd_start": qd0_bank.index_select(0, idx),
        }
    return sample_terminal_batch(
        n, rng=rng, device=device, dtype=torch.float32, **kwargs,
    )


def _build_start_bank(cfg: TrainConfig, device) -> tuple[torch.Tensor, torch.Tensor] | None:
    size = int(cfg.start_bank_size)
    if size <= 0:
        return None
    sampler = StartSampler(device=device, dtype=torch.float32)
    rng = np.random.default_rng(int(cfg.seed) + 7919)
    qs, qds = [], []
    for begin in range(0, size, 512):
        starts = sampler.sample(min(512, size - begin), rng=rng)
        qs.append(starts.q.detach())
        qds.append(starts.qd.detach())
    return torch.cat(qs), torch.cat(qds)


def heldout_metrics(
    net, cost, device, *, n: int = 256, seed: int = 12345, start_bank=None,
) -> dict:
    """Median contact, clearance, and angle stats on a fixed held-out draw."""
    rng = np.random.default_rng(seed)
    batch = _sample_batch(n, rng, device, start_bank)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        q = net(batch.features)
        out = cost(q, batch, None)
    if was_training:
        net.train()
    act = batch.active
    hit = out.aux["hit_dist"][act]
    elev = out.aux["stick_elev_deg"]
    pad = elev[act & ~batch.is_cymbal]
    cym_z = out.aux["stick_z"][act & batch.is_cymbal]
    viol = collision_violation(out.aux["min_sdf"])
    yaw = out.aux["waist_yaw_err_deg"].abs()
    yaw = yaw[act.any(dim=-1)]

    def med(t):
        return float(t.median().item()) if t.numel() else float("nan")

    return {
        "hit_med_mm": 1000.0 * med(hit),
        "hit_1cm": float((hit < 0.01).float().mean().item()) if hit.numel() else float("nan"),
        "viol_med_mm": 1000.0 * med(viol),
        "col_ok": float((viol <= 0).float().mean().item()),
        "pad_deg": med(pad),
        "pad_ge5": float((pad >= 5.0).float().mean().item()) if pad.numel() else float("nan"),
        "cym_z": med(cym_z),
        "cym_tip_up": float((cym_z > 0).float().mean().item()) if cym_z.numel() else float("nan"),
        "yaw_abs_deg": med(yaw),
        "start_rmse_rad": med(out.aux["start_rmse_rad"]),
        "accel_ratio_rms": med(out.aux["accel_ratio_rms"]),
    }


def grad_shares(
    net, cost, al, device, *, n: int = 32, seed: int = 12345, start_bank=None,
) -> dict:
    """Squared-norm shares of each priced term on one fresh batch."""
    rng = np.random.default_rng(seed)
    batch = _sample_batch(n, rng, device, start_bank)
    was_training = net.training
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    out = cost(net(batch.features), batch, al.cost_weights())
    norms = {}
    for key, term in out.terms.items():
        scalar = term.mean()
        if not scalar.requires_grad:
            norms[key] = 0.0
            continue
        grads = torch.autograd.grad(
            scalar, params, retain_graph=True, allow_unused=True,
        )
        sq = sum(g.detach().float().pow(2).sum() for g in grads if g is not None)
        norms[key] = float(sq.sqrt().item())
    if was_training:
        net.train()
    else:
        net.eval()
    total_sq = sum(v * v for v in norms.values())
    shares = {
        key: (v * v / total_sq if total_sq > 0 else 0.0) for key, v in norms.items()
    }
    shares["grad_norm"] = float(total_sq ** 0.5)
    return shares


class _LineSearchHost:
    """Fields ``init_line_search_runtime`` reads off the phase-2 config."""

    def __init__(self, lr: float, line_search_cfg: LineSearchConfig):
        self.lr = float(lr)
        self.line_search_cfg = line_search_cfg
        self.optimizer = "adamw"


def _line_search_step(ls_rt, net, cost, al, batch, out, cfg) -> bool:
    """SaLSa update.  Returns whether a parameter step was committed."""
    known_search, search_reasons, j = resolve_known_search(ls_rt)
    direction_due = needs_line_search_direction(
        ls_rt, known_search=known_search, j=j,
    )
    g_preclip = (
        snapshot_g_preclip(ls_rt.adapter._params) if direction_due else None
    )
    clip = float(cfg.grad_clip) if cfg.grad_clip and float(cfg.grad_clip) > 0.0 else float("inf")
    gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
    if not torch.isfinite(gnorm):
        latch_triggers(ls_rt.ctrl, ["nonfinite_gnorm"])
        return False
    viol_now = {
        "hit": float(out.violations["hit"].detach().mean().item()),
        "collision": float(out.violations["collision"].detach().mean().item()),
    }
    g2 = 0.0
    if g_preclip is not None:
        for g in g_preclip:
            if g is not None:
                g2 += float(g.detach().double().pow(2).sum().item())
    gnorm_pre = g2 ** 0.5 if g_preclip is not None else float(gnorm)
    late_reasons = (
        check_early_triggers(
            cfg=ls_rt.cfg,
            refs=ls_rt.refs,
            gnorm=gnorm_pre,
            viol_metrics=viol_now,
        )
        if ls_rt.cfg.search_on_early_triggers
        else []
    )
    run_search = bool(known_search) or (
        bool(ls_rt.cfg.search_on_early_triggers) and bool(late_reasons)
    )

    def probe_F() -> float:
        with torch.no_grad():
            q = net(batch.features)
            return float(cost(q, batch, al.cost_weights()).total.mean().item())

    result = run_search_or_ordinary(
        ls_rt,
        j=j,
        run_search=run_search,
        search_reasons=list(search_reasons) + list(late_reasons),
        F0=float(out.total.mean().detach().item()),
        g_preclip=g_preclip,
        probe_F=probe_F,
        g_objective=g_preclip,
    )
    if result.commit_side_effects and ls_rt.cfg.search_on_early_triggers:
        ls_rt.refs.update_after_valid_baseline(gnorm_pre, viol_now)
    return bool(result.executed)


def cosine_lr(step: int, steps: int, lr_start: float, lr_end: float) -> float:
    """Cosine from ``lr_start`` at step 1 to ``lr_end`` at step ``steps``."""
    span = max(int(steps) - 1, 1)
    t = min(max((int(step) - 1) / span, 0.0), 1.0)
    return float(lr_end) + (float(lr_start) - float(lr_end)) * 0.5 * (
        1.0 + math.cos(math.pi * t)
    )


def train_terminal_pose(cfg: TrainConfig) -> dict:
    """Run the differentiable terminal-pose loop; return the saved checkpoint dict."""
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    pose_cfg = TerminalPoseConfig(
        w_start=float(cfg.w_start),
        w_motion=float(cfg.w_motion),
        collision_topk=int(cfg.collision_topk),
    )
    # Net takes in_dim / width, not TerminalPoseConfig (cost weights live on the latter).
    net = TerminalPoseNet(
        width=cfg.width,
        depth=cfg.depth,
        optnet=cfg.optnet,
        n_constraints=cfg.n_constraints,
        centered_qp=cfg.centered_qp,
        mix_width=cfg.mix_width,
        qp_out=cfg.qp_out,
    ).to(device)
    cost = TerminalPoseCost(pose_cfg, device=device, dtype=torch.float32)
    start_bank = _build_start_bank(cfg, device)
    al = TerminalPoseAL()
    start_step = 0
    resume_ckpt = None
    if cfg.resume:
        resume_ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        net.load_state_dict(resume_ckpt["model"])
        if "al" in resume_ckpt:
            al.load_state_dict(resume_ckpt["al"])
        if "rng_state" in resume_ckpt:
            rng.bit_generator.state = resume_ckpt["rng_state"]
        start_step = int(resume_ckpt.get("step", 0))
    print(
        f"[init] optnet={cfg.optnet} centered_qp={cfg.centered_qp} "
        f"n_constraints={cfg.n_constraints} mix_width={cfg.mix_width} "
        f"qp_out={cfg.qp_out} line_search={cfg.line_search} "
        f"w_start={cfg.w_start:g} w_motion={cfg.w_motion:g} "
        f"collision_topk={cfg.collision_topk} start_bank={cfg.start_bank_size} "
        f"params={net.n_parameters()} "
        f"pen_hit={al.pen['hit']:.4g} pen_collision={al.pen['collision']:.4g} "
        f"pen_max_hit={al.pen_max['hit']:.4g} pen_max_collision={al.pen_max['collision']:.4g} "
        f"resume_step={start_step}",
        flush=True,
    )
    ls_rt = None
    if cfg.line_search:
        # Weight decay stays off: these runs were Adam with decay 0.  The
        # search seed is --lr, which for a resume is the cosine floor.
        opt = AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.0)
        if cfg.line_search_interval is not None:
            interval = int(cfg.line_search_interval)
            if interval < 1:
                raise ValueError(
                    f"line_search_interval must be >= 1, got {interval}"
                )
            line_search_cfg = LineSearchConfig(
                search_interval_mode="fixed",
                initial_interval=interval,
                min_interval=interval,
                max_interval=interval,
                search_on_early_triggers=False,
            )
        else:
            line_search_cfg = LineSearchConfig()
        if resume_ckpt is not None and "optimizer" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["optimizer"])
        ls_rt = init_line_search_runtime(
            phase2_cfg=_LineSearchHost(cfg.lr, line_search_cfg),
            optimizer=opt,
            objective_direction_fingerprint={"objective": "terminal_pose"},
            line_search_state=(
                None if resume_ckpt is None else resume_ckpt.get("line_search")
            ),
        )
        print(
            f"[line_search] eta_ref={ls_rt.eta_ref:.3g} "
            f"bounds=[{ls_rt.epsilon_min:.3g},{ls_rt.epsilon_max:.3g}] "
            f"seed={ls_rt.seed_provenance} "
            f"interval={ls_rt.cfg.initial_interval} "
            f"interval_mode={ls_rt.cfg.search_interval_mode} "
            f"early_triggers={int(ls_rt.cfg.search_on_early_triggers)}",
            flush=True,
        )
    else:
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        if resume_ckpt is not None and "optimizer" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["optimizer"])

    last_loss = float("nan")
    eval_history: list[dict] = []
    for step in range(1, int(cfg.steps) + 1):
        if ls_rt is None and cfg.lr_end is not None:
            lr = cosine_lr(step, cfg.steps, cfg.lr, cfg.lr_end)
            for group in opt.param_groups:
                group["lr"] = lr
        batch = _sample_batch(int(cfg.batch), rng, device, start_bank)
        q = net(batch.features)
        out = cost(q, batch, al.cost_weights())
        loss = out.total.mean()
        loss.backward()
        if ls_rt is None:
            if cfg.grad_clip and cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg.grad_clip))
            opt.step()
            stepped = True
        else:
            stepped = _line_search_step(ls_rt, net, cost, al, batch, out, cfg)
        if stepped and step % max(int(cfg.al_every), 1) == 0:
            al.update(out.violations)
        opt.zero_grad(set_to_none=True)

        last_loss = float(loss.detach().item())
        global_step = start_step + step
        if step % max(int(cfg.log_every), 1) == 0 or step == 1 or step == cfg.steps:
            v_hit = float(out.violations["hit"].detach().mean().item())
            v_col = float(out.violations["collision"].detach().mean().item())
            parts = [
                f"step={global_step}",
                f"loss={last_loss:.4g}",
                f"v_hit={v_hit:.4g}",
                f"v_collision={v_col:.4g}",
                f"lam_hit={al.lam['hit']:.4g}",
                f"lam_collision={al.lam['collision']:.4g}",
                f"pen_hit={al.pen['hit']:.4g}",
                f"pen_collision={al.pen['collision']:.4g}",
            ]
            if ls_rt is not None:
                eps = ls_rt.last_log.get("ls_epsilon")
                if eps is not None:
                    parts.append(f"ls_eps={float(eps):.4g}")
                parts.append(f"ls_searched={int(bool(ls_rt.last_log.get('ls_searched')))}")
            aux = getattr(out, "aux", None) or {}
            for label, aliases in _AUX_LOG_KEYS:
                val = _aux_mean(aux, aliases)
                if val is not None:
                    parts.append(f"{label}={val:.4g}")
            print(" ".join(parts), flush=True)
        if step % 1000 == 0 or step == cfg.steps:
            ev = heldout_metrics(net, cost, device, start_bank=start_bank)
            ev["step"] = global_step
            eval_history.append(dict(ev))
            print("[eval] " + " ".join(f"{k}={v:.4g}" for k, v in ev.items()), flush=True)

    shares = grad_shares(net, cost, al, device, start_bank=start_bank)
    print(
        "[grad] "
        + " ".join(
            f"{k}={v:.4g}" for k, v in shares.items() if k != "grad_norm"
        )
        + f" grad_norm={shares['grad_norm']:.4g}",
        flush=True,
    )

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": net.state_dict(),
        "config": _config_dict(pose_cfg),
        "step": int(start_step + cfg.steps),
        "optnet": bool(cfg.optnet),
        "in_dim": int(net.in_dim),
        "width": int(cfg.width),
        "depth": int(cfg.depth),
        "n_constraints": int(cfg.n_constraints),
        "centered_qp": bool(cfg.centered_qp),
        "mix_width": cfg.mix_width,
        "qp_out": bool(cfg.qp_out),
        "loss_spec": LOSS_SPEC,
        "grad_shares": shares,
        "eval_history": eval_history,
        "optimizer": opt.state_dict(),
        "rng_state": rng.bit_generator.state,
        "train_config": _config_dict(cfg),
        "line_search": None if ls_rt is None else runtime_state_dict(ls_rt),
        **_al_checkpoint(al),
    }
    torch.save(ckpt, out_path)
    print(f"[ckpt] {out_path}  params={net.n_parameters()}  loss={last_loss:.4g}", flush=True)
    return ckpt


def main(argv=None) -> int:
    cfg = parse_args(argv)
    train_terminal_pose(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
