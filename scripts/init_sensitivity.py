#!/usr/bin/env python
"""Init-sensitivity diagnostic: is the decoder a refiner, or does it ignore its init?

    python scripts/init_sensitivity.py --device cuda:1 --scenarios 20000 \
        --ckpt straight=runs/phase1_main/best.pt apex=runs/phase1_apex_mix/best.pt

Asked for by Eric (2026-08-10, update 1).  Pure straight-init -> apex-target
training makes the target a deterministic function of the CONTEXT alone, so the
model is free to learn to ignore its initialization -- which would silently
destroy the refinement-operator property every MPC warm start depends on.  The
50/50 init mix is meant to prevent that; this measures whether it did.

The probe: perturb the input init by a fixed OU displacement of sd ``--delta``
(20 mrad by default, the same tau = 5 waypoints the training noise uses) and
measure how far the OUTPUT moves.

    sensitivity  =  RMS(dout) / RMS(dinit)        (masked, per valid element)

* ~0        -> the model ignores its init (bad: no refinement, no warm-start
               benefit, and phase 2's iterated re-feed becomes a no-op);
* ~1        -> the output tracks the init one-for-one (a pure denoiser would
               be *below* 1: it removes part of the perturbation);
* in between -> a genuine refinement operator with contraction ``1 - s``.

Every model is probed on BOTH pools (straight-line targets and apex targets)
and on every init channel, so no model is judged only on its own distribution.
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
from amortized_opt.warmstart_apex import ApexConfig, generate_apex_pool  # noqa: E402
from amortized_opt.warmstart_data import (  # noqa: E402
    NoiseConfig,
    ScenarioConfig,
    generate_pool,
    ou_kernel,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--scenarios", type=int, default=20_000)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--delta", type=float, default=0.02, help="OU sd of the probe (rad)")
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--out", default="runs/phase1_apex_mix/init_sensitivity.json")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def probe(model, pool, *, init_mode, batch, batches, delta, tau, seed, amp):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=pool.device)
    gen.manual_seed(int(seed))
    acc = {k: 0.0 for k in ("din2", "dout2", "n", "res2", "out2")}
    from amortized_opt.train_warmstart import batch_sampler

    for b, idx in batch_sampler(pool, batch, rng=rng, steps=int(batches)):
        kw = {}
        if init_mode == "hold":
            kw["hold_init"] = True
        elif init_mode is not None:
            try:
                kw["init_mode"] = init_mode
            except TypeError:
                pass
        try:
            bt = pool.batch(b, idx, generator=gen, **kw)
        except TypeError:
            bt = pool.batch(b, idx, generator=gen)  # straight pool: no init_mode
        em = bt["masks"].elem_mask.to(torch.float32)
        L = ou_kernel(b, tau, pool.device, bt["init"].dtype)
        eps = torch.randn(bt["init"].shape, device=pool.device, dtype=bt["init"].dtype,
                          generator=gen)
        d_in = float(delta) * torch.einsum("ij,bjd->bid", L, eps) * em
        ctx = (torch.autocast(device_type="cuda", dtype=amp) if amp is not None
               else torch.autocast(device_type="cuda", enabled=False))
        with ctx:
            enc = model.encode_context(bt["context"])
            o1 = model(bt["init"], bt["state"], enc, bt["masks"])
            o2 = model(bt["init"] + d_in, bt["state"], enc, bt["masks"])
        d_out = (o2.traj.float() - o1.traj.float()) * em
        res = (o1.traj.float() - bt["target"].float()) * em
        acc["din2"] += float((d_in.float() ** 2).sum())
        acc["dout2"] += float((d_out ** 2).sum())
        acc["res2"] += float((res ** 2).sum())
        moved = (o1.traj.float() - bt["init"].float()) * em
        acc["out2"] += float((moved ** 2).sum())
        acc["n"] += float(em.sum())
    n = max(acc["n"], 1.0)
    rms_in = (acc["din2"] / n) ** 0.5
    rms_out = (acc["dout2"] / n) ** 0.5
    return {
        "rms_dinit_mrad": 1000 * rms_in,
        "rms_dout_mrad": 1000 * rms_out,
        "sensitivity": rms_out / max(rms_in, 1e-12),
        "rms_residual_to_target_mrad": 1000 * (acc["res2"] / n) ** 0.5,
        "rms_output_minus_init_mrad": 1000 * (acc["out2"] / n) ** 0.5,
        "n_elements": int(n),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16

    scen = ScenarioConfig()
    noise = NoiseConfig()
    fk = A.ChainFK(device=str(dev), dtype=torch.float32)
    rnea = A.BatchedRNEA.from_urdf(with_armature=True, device=str(dev), dtype=torch.float32)

    print(f"[data] {args.scenarios} straight-target val scenarios ...", flush=True)
    straight_pool = generate_pool(
        args.scenarios, seed=args.seed, cfg=scen, noise=noise, split="eval",
        workers=args.workers, device=str(dev), fk=fk, rnea=rnea,
    )
    print(f"[data] {args.scenarios} apex-target val scenarios ...", flush=True)
    apex_pool = generate_apex_pool(
        args.scenarios, seed=args.seed, cfg=scen, apex_cfg=ApexConfig(), noise=noise,
        split="eval", workers=args.workers, device=str(dev), fk=fk, rnea=rnea,
    )

    report: dict = {"delta_rad": args.delta, "tau_steps": args.tau,
                    "scenarios": args.scenarios, "models": {}}
    rows = []
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = A.build_model(A.ModelConfig(**ck["model_cfg"]), dev)
        model.load_state_dict(ck["model"])
        model.eval()
        report["models"][label] = {}
        for pool_name, pool, modes in (
            ("straight_pool", straight_pool, [None]),
            ("apex_pool", apex_pool, ["straight", "apex", "mix"]),
        ):
            for mode in modes:
                key = f"{pool_name}/{mode or 'default'}"
                r = probe(model, pool, init_mode=mode, batch=args.batch,
                          batches=args.batches, delta=args.delta, tau=args.tau,
                          seed=args.seed, amp=amp)
                report["models"][label][key] = r
                rows.append((label, key, r))
                print(f"[probe] {label:10s} {key:26s} "
                      f"|dinit|={r['rms_dinit_mrad']:6.2f} mrad -> "
                      f"|dout|={r['rms_dout_mrad']:6.2f} mrad  "
                      f"sensitivity={r['sensitivity']:.4f}  "
                      f"|out-target|={r['rms_residual_to_target_mrad']:7.2f} mrad",
                      flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    w = max(len(l) for l, _, _ in rows) + 2
    lines = ["", "===== init sensitivity (RMS output move / RMS init move) =====",
             f"{'probe':28s}" + "".join(f"{l:>{w}s}" for l in report['models'])]
    keys = list(next(iter(report["models"].values())))
    for k in keys:
        line = f"{k:28s}"
        for l in report["models"]:
            line += f"{report['models'][l][k]['sensitivity']:{w}.4f}"
        lines.append(line)
    txt = "\n".join(lines)
    print(txt)
    out.with_suffix(".txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
