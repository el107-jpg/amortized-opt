#!/usr/bin/env python
"""1-clip closed-loop MPC smoke run: apex model vs straight model as the planner.

    python scripts/mpc_smoke_apex.py --device cuda:1 \
        --ckpt straight=runs/phase1_main/best.pt apex=runs/phase1_apex/best.pt

Uses :class:`~amortized_opt.mpc_gen.MPCGenerator` **exactly as it is** -- the
known waist wind-up (README, "Known rough edges") is NOT worked around here.
The output of interest is therefore comparative: at which tick does each
planner's episode diverge, and how far does each get through the clip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.rhythm import build_context_inputs  # noqa: E402


def model_planner(model, device, *, num_drums=8, num_future=3, amp_dtype=None):
    """``(PlanRequest) -> (H, 17)``: one refinement pass from the request's init."""
    dev = torch.device(device)
    core = model.module if hasattr(model, "module") else model
    core.eval()

    @torch.no_grad()
    def plan(req):
        ctx = build_context_inputs(
            req.kit, req.snapshot, num_drums=num_drums, num_future=num_future,
            frame="env", device=dev, dtype=torch.float32,
        )
        t = lambda a: torch.as_tensor(  # noqa: E731
            np.asarray(a, float), device=dev, dtype=torch.float32
        )
        # MPCGenerator builds its batch-1 masks on the CPU; the model lives on
        # the GPU, so move the whole block rather than one field of it.
        # MPCGenerator builds its batch-1 masks on the CPU in float64; the model
        # lives on the GPU in float32, so move AND cast the whole block.
        cast = lambda v: v.to(dev, torch.float32) if v.is_floating_point() else v.to(dev)  # noqa: E731
        masks = type(req.masks)(
            **{k: cast(getattr(req.masks, k)) for k in req.masks.__dataclass_fields__}
        )
        init = t(req.init_traj)[None, :, :] * masks.elem_mask
        state = torch.cat([t(req.q), t(req.qd)])[None, :]
        amp = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp_dtype is not None and dev.type == "cuda")
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with amp:
            enc = core.encode_context(ctx)
            out = core.refine(init, state, enc, masks, iterations=1)
        return out.traj[0].float().cpu().numpy()

    return plan


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/phase1_apex/mpc_smoke.json")
    args = p.parse_args(argv)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")

    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)
    clip = clips[0]
    gen = A.MPCGenerator(kit=kit)
    starts = A.StartSampler(kit=kit, library=gen.library, fk=gen.fk,
                            rnea=gen.rnea, sdf=gen.sdf)
    st = starts.sample(1, seed=args.seed)
    q0 = st.q[0].detach().cpu().numpy().astype(float)
    qd0 = st.qd[0].detach().cpu().numpy().astype(float)

    report: dict = {"clip": clip.name if hasattr(clip, "name") else str(clip),
                    "steps_requested": args.steps, "planners": {}}
    runs = [("quintic_teacher", None)]
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = A.build_model(A.ModelConfig(**ck["model_cfg"]), dev)
        model.load_state_dict(ck["model"])
        runs.append((label, model_planner(model, dev, amp_dtype=torch.bfloat16)))

    for label, plan in runs:
        rng = np.random.default_rng(args.seed)
        try:
            ep = gen.run_episode(clip, q0, qd0, rng=rng, episode=0, planner=plan,
                                 max_steps=int(args.steps))
            q = np.asarray(ep.q_hist, float)
            rec = {
                "ticks": int(q.shape[0] - 1),
                "diverged": bool(ep.diverged),
                "diverged_step": int(ep.diverged_step),
                "hits": len(ep.hit_steps),
                "max_abs_q": float(np.nanmax(np.abs(q))),
                "waist_yaw_range": [float(np.nanmin(q[:, 0])), float(np.nanmax(q[:, 0]))],
                "samples": len(ep.samples),
            }
        except Exception as exc:  # the generator is used as-is, warts included
            rec = {"error": f"{type(exc).__name__}: {exc}"}
        report["planners"][label] = rec
        print(f"[mpc] {label}: {json.dumps(rec)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
