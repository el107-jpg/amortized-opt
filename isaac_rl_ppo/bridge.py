"""The single point of contact with the parameter-noise sibling ``isaac_rl``.

Every name this package consumes from the sibling buildout is resolved HERE and
nowhere else, so when ``amortized/isaac_rl/`` churns the damage is confined to
one file.  This is a hard rule for the package, not a style preference: the
sibling is being edited by a concurrent session while this one is written.

Two sources, selected by :data:`SOURCE_ENV`
-------------------------------------------

``"sibling"`` (**default**)
    The live ``isaac_rl`` package, whose public surface is published in
    ``amortized/isaac_rl/INTERFACES.md``.  Every name in :data:`REQUIRED` was
    checked against that document; ``schedule``, ``mpc`` and ``perturb`` are
    marked STABLE there, and ``env`` / the reward WEIGHTS are marked
    PROVISIONAL (values may move; the key sets are stable).

``"pinned"``
    The snapshot of ``isaac_rl`` copied into this directory at build time
    (``isaac_rl_ppo/{env,env_cfg,mpc,reward,schedule,measure,loop,perturb}.py``),
    byte-identical to the sibling as of 2026-08-11 07:38:46.  An escape hatch
    for the case where the sibling breaks mid-run.

The default moved from ``"pinned"`` to ``"sibling"`` once ``INTERFACES.md``
landed, and the reason is concrete rather than procedural: the snapshot's
``schedule.py`` predates the sibling's **float32 clock-drift fix**.  Their old
code accumulated ``t_now += 0.02`` in float32, which flipped a boundary
comparison at t = 1.70 s on a 46 s clip and left the hit cursor one hit behind
for the rest of the episode; the fix derives the clock as ``t0 + tick * dt`` in
float64.  Since every clip time is quantized to the control grid, *every* hit
sits on that knife edge -- so the pinned snapshot would mis-schedule strikes,
which is precisely the quantity this whole project is trying to improve.

:func:`load_isaac_rl` therefore warns loudly whenever it falls back to the
snapshot, naming the modules known to be stale.  Override with::

    export AMORTIZED_PPO_ISAAC_RL_SOURCE=pinned
"""

from __future__ import annotations

import importlib
import os
import warnings
from dataclasses import dataclass

__all__ = [
    "SOURCE_ENV",
    "IsaacRlSurface",
    "load_isaac_rl",
    "load_safety",
    "resolved_source",
]

#: Environment variable selecting the import source: ``pinned`` | ``sibling``.
SOURCE_ENV = "AMORTIZED_PPO_ISAAC_RL_SOURCE"

#: The names this package consumes.  This tuple IS the compatibility contract
#: published in ``README.md``; adding to it means adding to that table.
REQUIRED: dict[str, tuple[str, ...]] = {
    "env_cfg": ("CONTROL_DT", "DECIMATION", "SIM_DT", "DrumRLEnvCfg",
                "assert_rate_consistency", "build_env_cfg"),
    "schedule": ("BatchedSchedule", "ScheduleConfig"),
    "mpc": ("IsaacMPC", "MPCConfig", "TickResult"),
    "reward": ("ContactScore", "ExecutedReward", "ExecutedStrikeScorer",
               "RewardConfig", "RewardTerms"),
    "measure": ("EpisodeView", "latency_report", "per_env_latency"),
}

#: Optional -- imported if present, silently skipped if not.
OPTIONAL: dict[str, tuple[str, ...]] = {
    "env": ("DrumRLEnvCfg", "make_env"),
    "safety": ("BatchedSafetyFilter", "SafetyConfig"),
    "loop": ("DrumMPCLoop",),
    "perturb": ("NoiseConfig", "ParameterNoise"),
}


@dataclass
class IsaacRlSurface:
    """The resolved sibling surface, as plain attributes.

    Access is by attribute (``surface.IsaacMPC``) so call sites read like
    ordinary imports while the resolution stays in one place.
    """

    source: str
    modules: dict
    names: dict

    def __getattr__(self, item):
        try:
            return self.names[item]
        except KeyError as exc:  # pragma: no cover - programming error
            raise AttributeError(
                f"{item!r} is not part of the isaac_rl surface this package "
                f"consumes (source={self.source!r}).  Add it to "
                f"bridge.REQUIRED/OPTIONAL and to the contract table in "
                f"isaac_rl_ppo/README.md."
            ) from exc

    def has(self, name: str) -> bool:
        return name in self.names


#: Modules whose pinned snapshot is KNOWN to differ from the live sibling in a
#: way that changes behaviour.  Kept explicit so the fallback warning can name
#: them rather than saying "something may be stale".
KNOWN_STALE: dict[str, str] = {
    "schedule": "predates the float32 clock-drift fix (t_now accumulated in "
                "float32 flipped a hit-boundary comparison at t=1.70s and left "
                "the cursor one hit behind for the rest of the episode)",
}


def resolved_source() -> str:
    """``"sibling"`` unless the environment pins the snapshot."""
    src = os.environ.get(SOURCE_ENV, "sibling").strip().lower()
    if src not in ("pinned", "sibling"):
        raise ValueError(
            f"{SOURCE_ENV}={src!r}; expected 'pinned' or 'sibling'")
    return src


def _module_path(source: str, name: str) -> str:
    if source == "sibling":
        return f"isaac_rl.{name}"
    return f"{__package__}.{name}"


_CACHE: dict[str, IsaacRlSurface] = {}


def load_isaac_rl(source: str | None = None, *, allow_fallback: bool = True) -> IsaacRlSurface:
    """Import the sibling surface and check the contract.

    A contract break raises ``ImportError`` naming the missing symbol -- the
    failure mode we want, because it is loud, immediate, and points at the exact
    line of ``INTERFACES.md`` that moved.  If the LIVE sibling cannot be
    imported at all (mid-edit syntax error, a module briefly absent), the pinned
    snapshot is used instead and a ``RuntimeWarning`` names what is stale.
    """
    src = source or resolved_source()
    if src in _CACHE:
        return _CACHE[src]

    if src == "sibling" and allow_fallback:
        try:
            return _load(src)
        except Exception as exc:
            stale = "; ".join(f"{m}: {why}" for m, why in KNOWN_STALE.items())
            warnings.warn(
                f"isaac_rl bridge: the live sibling failed to import ({exc!r}); "
                f"falling back to the PINNED snapshot. Known-stale modules -- "
                f"{stale}. Results from this run are not comparable to the "
                f"sibling's.", RuntimeWarning, stacklevel=2)
            return _load("pinned")
    return _load(src)


def _load(src: str) -> IsaacRlSurface:
    modules: dict = {}
    names: dict = {}
    for mod_name, symbols in REQUIRED.items():
        path = _module_path(src, mod_name)
        try:
            mod = importlib.import_module(path)
        except ImportError as exc:
            raise ImportError(
                f"isaac_rl bridge: could not import {path!r} (source={src!r}). "
                f"With source='sibling' the live buildout must be on sys.path "
                f"as the top-level package 'isaac_rl'."
            ) from exc
        modules[mod_name] = mod
        for sym in symbols:
            if not hasattr(mod, sym):
                raise ImportError(
                    f"isaac_rl bridge: {path}.{sym} is missing (source={src!r}). "
                    f"The sibling's public surface moved; see the compatibility "
                    f"contract in isaac_rl_ppo/README.md."
                )
            names[sym] = getattr(mod, sym)

    for mod_name, symbols in OPTIONAL.items():
        path = _module_path(src, mod_name)
        try:
            mod = importlib.import_module(path)
        except Exception:  # Isaac-dependent modules raise more than ImportError
            continue
        modules[mod_name] = mod
        for sym in symbols:
            if hasattr(mod, sym):
                names.setdefault(sym, getattr(mod, sym))

    surface = IsaacRlSurface(source=src, modules=modules, names=names)
    _CACHE[src] = surface
    return surface


def load_safety() -> tuple:
    """``(BatchedSafetyFilter, SafetyConfig)``, or ``(None, None)``.

    The pinned snapshot predates ``isaac_rl/safety.py`` by one second and so
    does not contain it; the live sibling does.  Either way the filter is
    optional (``DrumMPCLoop`` takes ``safety=None``) and PPO v1 runs without it
    -- a filter that silently rewrites the executed command changes the action
    the stored log-prob was computed for, which is a correctness question in its
    own right and is recorded as a DECISION NEEDING REVIEW rather than switched
    on by default.
    """
    for path in (f"{__package__}.safety", "isaac_rl.safety"):
        try:
            mod = importlib.import_module(path)
        except Exception:
            continue
        filt = getattr(mod, "BatchedSafetyFilter", None)
        conf = getattr(mod, "SafetyConfig", None)
        if filt is not None:
            return filt, conf
    warnings.warn(
        "isaac_rl_ppo.bridge: no BatchedSafetyFilter available (the pinned "
        "snapshot predates isaac_rl/safety.py). Running without the QP filter.",
        RuntimeWarning, stacklevel=2,
    )
    return None, None
