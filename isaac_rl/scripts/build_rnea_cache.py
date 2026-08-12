#!/usr/bin/env python
"""Freeze the RNEA kinematic tree so Isaac can build it without Pinocchio.

Run this OUTSIDE Isaac (a plain `g1drum` shell).  See ``isaac_rl.rnea_cache``
for why it is needed.  It also verifies the cached RNEA against the live one on
random states, because a cache that is merely close would be a wrong-physics
bug that raises nowhere.

    python isaac_rl/scripts/build_rnea_cache.py
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import amortized_opt as A                                    # noqa: E402
from isaac_rl.rnea_cache import CACHE_PATH, build_cache, load_cached  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(CACHE_PATH))
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--tol", type=float, default=1e-6)
    a = p.parse_args(argv)

    path = build_cache(a.out)
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KiB)")

    live = A.BatchedRNEA.from_urdf(device="cpu", dtype=torch.float64)
    cached = load_cached(path, device="cpu", dtype=torch.float64)
    assert list(cached.parent) == list(live.parent)
    assert list(cached.opt2pin) == list(live.opt2pin)
    assert getattr(cached, "spatial_layout", None) == getattr(live, "spatial_layout", None)

    g = torch.Generator().manual_seed(0)
    n = len(A.G1_UPPER_BODY_JOINTS)
    lo, hi = A.urdf_position_limits()
    lo = torch.as_tensor(lo, dtype=torch.float64)
    hi = torch.as_tensor(hi, dtype=torch.float64)
    q = lo + (hi - lo) * torch.rand(a.batch, n, generator=g, dtype=torch.float64)
    qd = 2.0 * torch.randn(a.batch, n, generator=g, dtype=torch.float64)
    qdd = 20.0 * torch.randn(a.batch, n, generator=g, dtype=torch.float64)
    err = (live.rnea(q, qd, qdd) - cached.rnea(q, qd, qdd)).abs().max().item()
    print(f"max |tau_live - tau_cached| over {a.batch} random states: {err:.3e}")
    # gravity-only, which is what the env's feedforward actually uses
    z = torch.zeros_like(q)
    errg = (live.rnea(q, z, z) - cached.rnea(q, z, z)).abs().max().item()
    print(f"max |tau_grav difference|                              : {errg:.3e}")
    ok = err <= a.tol and errg <= a.tol
    print("CACHE_OK" if ok else f"CACHE_MISMATCH (tol {a.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
