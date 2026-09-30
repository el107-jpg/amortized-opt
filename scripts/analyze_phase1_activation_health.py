#!/usr/bin/env python
"""Compare activation health and decoder-layer utility on phase-1 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A
from amortized_opt.model import Attention, DecoderBlock, EncoderBlock, SwiGLU
from amortized_opt.waist_aim import WaistAimConfig
from amortized_opt.warmstart_apex import ApexConfig, generate_apex_pool
from amortized_opt.warmstart_data import NoiseConfig, ScenarioConfig


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--label", action="append", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--draws", type=int, default=384)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--out", required=True)
    return p.parse_args()


def rms(x):
    return float(x.float().square().mean().sqrt())


def stable_rank(w):
    w = w.detach().float()
    fro2 = float(w.square().sum())
    v = torch.randn(w.shape[1], device=w.device)
    for _ in range(30):
        v = F.normalize(w.T @ (w @ v), dim=0)
    top2 = float((w @ v).square().sum())
    return fro2 / max(top2, 1e-30)


def linear_cka(x, y):
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    xy = x.T @ y
    return float(xy.square().sum() / ((x.T @ x).square().sum().sqrt() *
                                      (y.T @ y).square().sum().sqrt()).clamp_min(1e-30))


def masked_rows(x, valid):
    return x[valid].detach()


@torch.no_grad()
def analyze(label, ckpt_path, batch, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(cfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    counts = A.param_counts(model)

    records = {"ffn": {}, "attention": {}, "blocks": {}}
    handles = []
    decoder_valid = torch.cat([
        torch.ones_like(batch["masks"].token_mask[:, :1]), batch["masks"].token_mask
    ], 1)

    def ffn_hook(name):
        def hook(mod, inp, out):
            x = inp[0]
            a, b = mod.w12(x).chunk(2, -1)
            u = F.silu(a) * b
            flat = u.reshape(-1, u.shape[-1]).float()
            unit_rms = flat.square().mean(0).sqrt()
            median = unit_rms.median().clamp_min(1e-30)
            records["ffn"][name] = {
                "unit_rms_min_over_median": float(unit_rms.min() / median),
                "weak_units_lt_1pct_median": int((unit_rms < .01 * median).sum()),
                "near_zero_activation_frac": float((flat.abs() < 1e-6).float().mean()),
                "gate_a_lt_minus5_frac": float((a < -5).float().mean()),
                "output_over_input_rms": rms(out) / max(rms(x), 1e-30),
                "w3_stable_rank": stable_rank(mod.w3.weight),
            }
        return hook

    def attn_hook(name):
        def hook(mod, inp, out):
            x, kv, mask = inp
            q = mod._split(mod.q_proj(x)).float()
            k = mod._split(mod.k_proj(kv)).float()
            if mod.rope:
                q, k = mod._rope(q, k)
            logits = q @ k.transpose(-1, -2) / math.sqrt(mod.dh)
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None, None, :], -torch.inf)
            p = logits.softmax(-1)
            ent = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(-1)
            if mask is None:
                denom = math.log(max(k.shape[-2], 2))
                ent_n = ent / denom
            else:
                denom = mask.sum(-1).clamp_min(2).float().log()[:, None, None]
                ent_n = ent / denom
            pf = p.permute(1, 0, 2, 3).reshape(mod.h, -1)
            pf = F.normalize(pf, dim=1)
            sim = pf @ pf.T
            off = sim[~torch.eye(mod.h, dtype=torch.bool, device=sim.device)]
            records["attention"][name] = {
                "entropy_normalized_mean": float(ent_n.mean()),
                "max_probability_mean": float(p.max(-1).values.mean()),
                "head_probability_cosine_mean": float(off.mean()),
                "head_probability_cosine_max": float(off.max()),
                "output_over_input_rms": rms(out) / max(rms(x), 1e-30),
            }
        return hook

    def block_hook(name, valid):
        def hook(mod, inp, out):
            x = inp[0]
            delta = out - x
            records["blocks"][name] = {
                "input": masked_rows(x, valid),
                "output": masked_rows(out, valid),
                "delta_over_input_rms": rms(delta[valid]) / max(rms(x[valid]), 1e-30),
                "input_output_cosine": float(F.cosine_similarity(x[valid].float(), out[valid].float(), dim=-1).mean()),
            }
        return hook

    def encoder_block_hook(name):
        def hook(mod, inp, out):
            x, valid = inp
            delta = out - x
            records["blocks"][name] = {
                "input": masked_rows(x, valid),
                "output": masked_rows(out, valid),
                "delta_over_input_rms": rms(delta[valid]) / max(rms(x[valid]), 1e-30),
                "input_output_cosine": float(F.cosine_similarity(x[valid].float(), out[valid].float(), dim=-1).mean()),
            }
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, SwiGLU):
            handles.append(mod.register_forward_hook(ffn_hook(name)))
        elif isinstance(mod, Attention):
            handles.append(mod.register_forward_hook(attn_hook(name)))
        elif isinstance(mod, DecoderBlock):
            handles.append(mod.register_forward_hook(block_hook(name, decoder_valid)))
        elif isinstance(mod, EncoderBlock):
            handles.append(mod.register_forward_hook(encoder_block_hook(name)))

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        enc = model.encode_context(batch["context"])
        pred = model(batch["init"], batch["state"], enc, batch["masks"]).traj
    for h in handles:
        h.remove()
    em = batch["masks"].elem_mask.bool()
    base_rmse = rms((pred - batch["target"])[em])

    # Causal ablations: skip a complete layer, or zero one residual branch.
    ablations = {}
    for li, blk in enumerate(model.decoder.blocks):
        tests = {
            "whole_block": (blk, lambda m, i, o: i[0]),
            "self_attn": (blk.self_attn, lambda m, i, o: torch.zeros_like(o)),
            "cross_attn": (blk.cross_attn, lambda m, i, o: torch.zeros_like(o)),
            "ffn": (blk.ffn, lambda m, i, o: torch.zeros_like(o)),
        }
        for kind, (module, fn) in tests.items():
            # Structurally masked decoder cross-attention is represented by
            # ``None``.  It has no parameters or forward branch to hook and
            # therefore should be omitted rather than treated as an ablation.
            if module is None:
                continue
            h = module.register_forward_hook(fn)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc2 = model.encode_context(batch["context"])
                out = model(batch["init"], batch["state"], enc2, batch["masks"]).traj
            h.remove()
            rr = rms((out - batch["target"])[em])
            ablations[f"layer{li}.{kind}"] = {
                "rmse_rad": rr, "delta_rmse_mrad": 1000 * (rr - base_rmse),
                "prediction_change_mrad": 1000 * rms((out - pred)[em]),
            }

    for li, blk in enumerate(model.encoder.blocks):
        tests = {
            "whole_block": (blk, lambda m, i, o: i[0]),
            "self_attn": (blk.attn, lambda m, i, o: torch.zeros_like(o)),
            "ffn": (blk.ffn, lambda m, i, o: torch.zeros_like(o)),
        }
        for kind, (module, fn) in tests.items():
            if module is None:
                continue
            h = module.register_forward_hook(fn)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc2 = model.encode_context(batch["context"])
                out = model(batch["init"], batch["state"], enc2, batch["masks"]).traj
            h.remove()
            rr = rms((out - batch["target"])[em])
            ablations[f"encoder{li}.{kind}"] = {
                "rmse_rad": rr, "delta_rmse_mrad": 1000 * (rr - base_rmse),
                "prediction_change_mrad": 1000 * rms((out - pred)[em]),
            }

    block_names = [f"decoder.blocks.{i}" for i in range(cfg.dec_layers)]
    cka = {}
    for i in range(len(block_names)):
        for j in range(i + 1, len(block_names)):
            cka[f"layer{i}_layer{j}"] = linear_cka(
                records["blocks"][block_names[i]]["output"],
                records["blocks"][block_names[j]]["output"])
    encoder_cka = {}
    encoder_names = [f"encoder.blocks.{i}" for i in range(cfg.ctx_layers)]
    for i in range(len(encoder_names)):
        for j in range(i + 1, len(encoder_names)):
            encoder_cka[f"layer{i}_layer{j}"] = linear_cka(
                records["blocks"][encoder_names[i]]["output"],
                records["blocks"][encoder_names[j]]["output"])

    for v in records["blocks"].values():
        del v["input"]
        del v["output"]
    return {
        "label": label, "checkpoint": str(ckpt_path), "step": int(ck["step"]),
        "model_cfg": {"d_model": cfg.d_model, "dec_layers": cfg.dec_layers,
                      "dec_heads": cfg.dec_heads, "d_ctx": cfg.d_ctx,
                      "ctx_layers": cfg.ctx_layers, "ctx_heads": cfg.ctx_heads,
                      "dec_cross_attn_layers": (
                          None if cfg.dec_cross_attn_layers is None
                          else list(cfg.dec_cross_attn_layers)
                      )},
        "parameters": counts, "baseline_batch_rmse_rad": base_rmse,
        "activation": records, "decoder_output_cka": cka,
        "encoder_output_cka": encoder_cka, "ablations": ablations,
    }


def main():
    a = args()
    if len(a.ckpt) != len(a.label):
        raise SystemExit("one --label is required per --ckpt")
    dev = torch.device(a.device)
    torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")

    # Phase 2 intentionally contracts the context from three future hits to
    # one.  Build the diagnostic probe with the checkpoint's actual context
    # shape instead of assuming the Phase-1 default.  A shared probe is only
    # meaningful when all checkpoints in one invocation use the same shape.
    probe_cfgs = [
        A.model_config_from_checkpoint(
            torch.load(path, map_location="cpu", weights_only=False)
        )
        for path in a.ckpt
    ]
    num_futures = {int(cfg.num_future) for cfg in probe_cfgs}
    if len(num_futures) != 1:
        raise SystemExit(
            "checkpoints with different num_future values require separate "
            "diagnostic invocations"
        )
    num_future = num_futures.pop()
    scen = ScenarioConfig(
        horizon=50, buckets=(50,), hit_time_jitter_s=.05, hit_loudness_jitter=.2,
        idle_home_path="default", rest_frac=.25, waist_upright_ease_wp=5,
        num_future=num_future,
        waist_cfg=WaistAimConfig(future_tau=1.0, future_max=num_future),
    )
    pool = generate_apex_pool(
        a.draws, seed=18508, cfg=scen, apex_cfg=ApexConfig(), noise=NoiseConfig(),
        split="eval", workers=a.workers, device=dev, init_mode="mix", init_mix=.5,
    )
    n = min(a.batch, pool.size(50))
    idx = torch.arange(n, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20260923)
    batch = pool.batch(50, idx, generator=gen)
    reports = [analyze(l, p, batch, dev) for l, p in zip(a.label, a.ckpt)]
    payload = {"probe": {"draws": a.draws, "accepted": len(pool), "batch": n,
                         "bucket": 50, "seed": 18508, "init_mode": "mix",
                         "num_future": num_future},
               "models": reports}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
