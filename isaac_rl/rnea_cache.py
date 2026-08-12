"""A Pinocchio-free way to build ``BatchedRNEA`` inside the Isaac process.

The problem, measured
---------------------

``BatchedRNEA.from_urdf`` needs Pinocchio exactly ONCE, to read the kinematic
tree (``model.parents``, ``jointPlacements``, ``inertias``, the spatial layout);
evaluation afterwards is pure torch.  Inside an Isaac Sim process that one call
fails on this box::

    RuntimeError: this Pinocchio build cannot expose model.parents;
    the repo's FK-based fallback (batch_rnea._infer_parent_ids_fk) is not
    vendored here

Isaac ships its own Boost/Eigen and its ``omni.*`` extensions load before user
code, which leaves the conda Pinocchio's ``model.parents`` binding unusable.
Outside Isaac the identical call works (``isaac_rl/scripts/ref_loop.py`` builds
it fine), so this is an Isaac-process-only limitation, not a broken module.

The cost of not fixing it was concrete: in the RL-1 smoke run the
gravity/Coriolis feedforward was OFF (so the servo had to hold gravity with
position error -- a different plant from the one the per-joint lead was fitted
against), ``FusedCost`` could not be built (so the analytic half of the reward
was dead), and ``TrajectoryScorer`` could not be built (so the hit-latency
histogram had to be computed out-of-process on a dump).

The fix
-------

The kinematic tree is a **build-time constant** of the URDF.  So: build the
RNEA once in a normal process, freeze every field of the dataclass into an
``.npz``, and reconstruct from that inside Isaac.  Nothing is approximated --
the cached arrays are the same ``float64`` numbers ``from_pinocchio`` produced,
so a cache-built RNEA is bit-comparable to a live-built one (the conformance
test asserts ``<= 1e-6`` on random states, and measures ~1e-15).

The cache carries a **fingerprint** (URDF sha256 + size, joint names, armature
flag, ``BatchedRNEA.BUILD_ID``).  A stale cache is refused rather than used:
silently evaluating a stale kinematic tree would be a wrong-physics bug that
raises nowhere, which is the same class of failure as the ``sim.device``
default that cost this stage two hours.

Usage::

    # once, in any non-Isaac process (or via scripts/build_rnea_cache.py)
    build_cache()

    # inside Isaac
    rnea = build_or_load(device="cuda:1")
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, default_urdf_path
from amortized_opt.rnea import BatchedRNEA

__all__ = ["CACHE_PATH", "build_cache", "cache_fingerprint", "load_cached",
           "build_or_load", "rnea_from_cache_if_needed"]

#: Lives next to the code rather than under ``assets/`` because it is a
#: derived artifact of the vendored URDF, not an asset in its own right.
CACHE_PATH: Path = Path(__file__).resolve().parent / "rnea_cache.npz"


def cache_fingerprint(joint_names=G1_UPPER_BODY_JOINTS, urdf_path=None,
                      with_armature: bool = True) -> str:
    """Everything that would invalidate the frozen tree, in one string."""
    p = Path(urdf_path) if urdf_path else default_urdf_path()
    raw = p.read_bytes()
    h = hashlib.sha256(raw).hexdigest()[:16]
    return "|".join([
        BatchedRNEA.BUILD_ID, h, str(len(raw)),
        ",".join(joint_names), f"armature={bool(with_armature)}",
    ])


def build_cache(path: Path | str | None = None, *,
                joint_names=G1_UPPER_BODY_JOINTS, urdf_path=None,
                with_armature: bool = True) -> Path:
    """Build with Pinocchio and freeze every field.  Run OUTSIDE Isaac."""
    path = Path(path) if path else CACHE_PATH
    r = BatchedRNEA.from_urdf(urdf_path, joint_names=joint_names,
                              with_armature=with_armature, device="cpu",
                              dtype=torch.float64)
    payload = {
        "axis": r.axis.numpy(),
        "R0": r.R0.numpy(),
        "p0": r.p0.numpy(),
        "I": r.I.numpy(),
        "gravity": r.gravity.numpy(),
        "parent": np.asarray(r.parent, np.int64),
        "body_to_q": np.asarray(r.body_to_q, np.int64),
        "body_to_v": np.asarray(r.body_to_v, np.int64),
        "opt2pin": np.asarray(r.opt2pin, np.int64),
        "n_bodies": np.asarray(r.n_bodies, np.int64),
        "nq_pin": np.asarray(r.nq_pin, np.int64),
        "nv_pin": np.asarray(r.nv_pin, np.int64),
        "has_armature": np.asarray(r.armature is not None),
        "armature": (r.armature.numpy() if r.armature is not None
                     else np.zeros(0, float)),
        # ``spatial_layout`` is set as an attribute by ``from_pinocchio`` and is
        # NOT a dataclass field, so it has to be carried explicitly or the
        # reconstructed object silently loses it.
        "spatial_layout": np.asarray(getattr(r, "spatial_layout", "LA")),
        "fingerprint": np.asarray(
            cache_fingerprint(joint_names, urdf_path, with_armature)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_cached(path: Path | str | None = None, *, device="cpu",
                dtype: torch.dtype = torch.float32,
                joint_names=G1_UPPER_BODY_JOINTS, urdf_path=None,
                with_armature: bool = True, check: bool = True) -> BatchedRNEA:
    """Reconstruct from the frozen tree.  No Pinocchio, no URDF parsing."""
    path = Path(path) if path else CACHE_PATH
    d = np.load(path, allow_pickle=False)
    if check:
        want = cache_fingerprint(joint_names, urdf_path, with_armature)
        got = str(d["fingerprint"])
        if got != want:
            raise RuntimeError(
                f"stale RNEA cache at {path}:\n  cached   {got}\n  expected {want}\n"
                "rebuild it with isaac_rl/scripts/build_rnea_cache.py")
    dev = torch.device(device)
    t = lambda k: torch.as_tensor(d[k], device=dev, dtype=dtype)  # noqa: E731
    out = BatchedRNEA(
        axis=t("axis"), R0=t("R0"), p0=t("p0"), I=t("I"), gravity=t("gravity"),
        parent=[int(v) for v in d["parent"]],
        body_to_q=[int(v) for v in d["body_to_q"]],
        body_to_v=[int(v) for v in d["body_to_v"]],
        opt2pin=[int(v) for v in d["opt2pin"]],
        n_bodies=int(d["n_bodies"]),
        nq_pin=int(d["nq_pin"]),
        nv_pin=int(d["nv_pin"]),
        armature=(t("armature") if bool(d["has_armature"]) else None),
    )
    out.spatial_layout = str(d["spatial_layout"])  # type: ignore[attr-defined]
    return out


def build_or_load(*, device="cpu", dtype: torch.dtype = torch.float32,
                  path: Path | str | None = None,
                  joint_names=G1_UPPER_BODY_JOINTS, urdf_path=None,
                  with_armature: bool = True, verbose: bool = True):
    """Live Pinocchio build if it works, else the cache.  ``None`` if neither.

    Live is tried FIRST so a normal process never depends on the cache being
    fresh; the cache is the Isaac-only fallback.  Returning ``None`` rather than
    raising keeps the env constructible without the feedforward, but the caller
    is expected to log it loudly -- a silently missing feedforward changes the
    plant.

    Set ``AMORTIZED_RNEA_CACHE_ONLY=1`` to skip the live attempt entirely.  On
    this box the live build inside Isaac raises a clean ``RuntimeError`` and the
    fallback is invisible, but a Pinocchio/Isaac pairing where the failure is a
    *crash* rather than an exception would take the whole process down; the
    env var is the escape hatch, and it is also how a run pins itself to the
    committed cache for reproducibility.
    """
    if os.environ.get("AMORTIZED_RNEA_CACHE_ONLY", "").strip() not in ("", "0", "false"):
        try:
            r = load_cached(path, device=device, dtype=dtype,
                            joint_names=joint_names, urdf_path=urdf_path,
                            with_armature=with_armature)
            if verbose:
                print("[isaac_rl] RNEA from cache (AMORTIZED_RNEA_CACHE_ONLY)", flush=True)
            return r
        except Exception as exc:
            if verbose:
                print(f"[isaac_rl] RNEA unavailable: cache-only, cache={exc}", flush=True)
            return None
    try:
        return BatchedRNEA.from_urdf(urdf_path, joint_names=joint_names,
                                     with_armature=with_armature,
                                     device=device, dtype=dtype)
    except Exception as live_exc:
        try:
            r = load_cached(path, device=device, dtype=dtype,
                            joint_names=joint_names, urdf_path=urdf_path,
                            with_armature=with_armature)
            if verbose:
                print(f"[isaac_rl] RNEA from cache ({live_exc.__class__.__name__} "
                      f"on the live build)", flush=True)
            return r
        except Exception as cache_exc:
            if verbose:
                print(f"[isaac_rl] RNEA unavailable: live={live_exc}; "
                      f"cache={cache_exc}", flush=True)
            return None


@contextmanager
def rnea_from_cache_if_needed(*, verbose: bool = True):
    """Make ``BatchedRNEA.from_urdf`` fall back to the cache, process-wide.

    Some consumers build their own RNEA internally and expose no injection
    point -- ``eval_harness.TrajectoryScorer`` is the one that matters here,
    because it is what ``probe.hit_latency`` (and therefore the headline
    Isaac-vs-analytic comparison) runs on.  Rather than fork the scorer, this
    patches the single classmethod for the duration of the block, and only
    engages the cache when the live build actually raises.  Outside Isaac the
    live path still wins, so behaviour is unchanged there.
    """
    orig = BatchedRNEA.from_urdf

    @classmethod
    def _patched(cls, urdf_path=None, *, joint_names=G1_UPPER_BODY_JOINTS,
                 with_armature=True, device="cpu", dtype=torch.float32):
        try:
            return orig(urdf_path, joint_names=joint_names,
                        with_armature=with_armature, device=device, dtype=dtype)
        except Exception:
            if verbose:
                print("[isaac_rl] TrajectoryScorer/RNEA served from cache",
                      flush=True)
            return load_cached(device=device, dtype=dtype,
                               joint_names=joint_names, urdf_path=urdf_path,
                               with_armature=with_armature)

    BatchedRNEA.from_urdf = _patched
    try:
        yield
    finally:
        BatchedRNEA.from_urdf = orig
