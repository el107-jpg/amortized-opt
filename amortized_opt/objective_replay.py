"""Objective-replay / RNG-freeze helpers for SaLSa line search (§8 / Bind F).

This module is the reusable substrate for evaluating

    F(theta; replay_ctx, C)

on the *same* primary (and optional BPTT) minibatch, frozen curriculum
controls ``C``, and realized external randomness, without advancing the live
training RNG streams.  It does **not** assemble the scalar objective itself —
that stays in :func:`amortized_opt.train_cost.train_phase2` / a future shared
wrapper.  Wire this in a later tranche; do not edit ``train_cost.py`` here.

---------------------------------------------------------------------------
RNG inventory in ``train_phase2`` (verified)
---------------------------------------------------------------------------

Named streams created near the top of :func:`train_phase2`::

    gen      = torch.Generator(device)   # seed cfg.seed+7919+101*rank
    sfx_gen  = torch.Generator(device)   # seed cfg.seed+4243+101*rank
    jit_gen  = torch.Generator(device)   # seed cfg.seed+5501+101*rank
    rng      = np.random.default_rng(cfg.seed + 31*rank)   # NOT random.Random

``rng`` is a :class:`numpy.random.Generator`.  There is **no**
:class:`random.Random` in the phase-2 loop today; helpers below still accept
one for completeness / plan wording.

Per-step draws that feed ``F`` (after the minibatch exists)::

    rng.choice / rng.uniform     bucket index, replay coin, BPTT coin
    gen                          frontier / fresh / replay batch assembly,
                                 TC ``sim.sample_gains``, ``_bptt_term`` draws
    jit_gen                      ``yaw_now`` Gaussian jitter
    sfx_gen                      :func:`suffix_offsets`

Global torch CPU/CUDA RNG: checkpointed only as ``torch_rng``
(:func:`save_phase2_checkpoint`).  Attention dropout
(``ModelConfig.dropout``, default ``0.0``) would consume the *global* device
RNG when ``> 0`` and ``model.training``; capture global state around forwards
even when dropout is off so enabling it later stays replay-safe.

Additional stream **outside** ``F`` once the batch is frozen::

    roller.gen                   :class:`Phase2Roller` private Generator,
                                 advanced in ``roller.tick()`` before the
                                 minibatch draw.  Do not tick the roller
                                 inside probes; optionally register it in
                                 the stream map for checkpoint parity only.

---------------------------------------------------------------------------
Integration checklist (future ``train_phase2`` edit — not done here)
---------------------------------------------------------------------------

Call sites refer to symbols in :mod:`amortized_opt.train_cost`.

1. **Construct streams** (next to ``gen`` / ``sfx_gen`` / ``jit_gen`` / ``rng``)::

       replay_mgr = ObjectiveReplayContext({
           "gen": gen, "sfx_gen": sfx_gen, "jit_gen": jit_gen, "rng": rng,
           # optional: "roller_gen": roller.gen,
       }, capture_global=True)

2. **After** minibatch assembly (post ``_sample_fresh_batch`` /
   ``replay.batch`` / frontier splice, once ``batch`` and ``use_replay`` /
   ``bucket`` / ``bs`` are known) and **before** yaw jitter or any
   ``F``-internal draw, open a record::

       rec = replay_mgr.begin_record(
           primary_batch=batch,
           controls=FrozenControls(
               weights=None,  # fill after ``_wtensors``
               al_lambda=dict(al.weights()),
               pen=dict(al.pen),
               sfx_ramp=0.0,           # fill when computed
               include_bptt=False,     # set after BPTT coin
               step=step, stage_a=stage_a,
               bucket=bucket, use_replay=use_replay,
           ),
       )
       rec.mutable_pre = snapshot_mutable_batch_fields(batch)

3. **Record realized samples as they are drawn** (prefer this over
   re-drawing from restored RNG)::

       # yaw jitter block (~train_phase2 yaw_now_jitter)
       noise = torch.randn(..., generator=jit_gen)
       rec.samples.yaw_now_jitter = noise.detach()
       # TC gains
       kp, kd, _ = sim.sample_gains(n_tc, generator=gen)
       rec.samples.tc_kp, rec.samples.tc_kd = kp.detach(), kd.detach()
       # BPTT coin
       rec.controls.include_bptt = bool(bptt_fn is not None
                                        and rng.uniform() < cfg.bptt_frac(step))
       # inside a refactored ``_bptt_term``: stash b1/b2 refs + gains + bounce
       # Suffix
       offs = suffix_offsets(..., generator=sfx_gen)
       rec.samples.suffix_offs = offs.detach()
       rec.controls.sfx_ramp = sfx_ramp
       rec.controls.weights = w   # tensor dict from ``_wtensors``

4. **After** the logical baseline ``total = loss + extra`` (and any metrics
   that must match the live step), stamp post-forward RNG and keep the
   baseline's buffer writes::

       replay_mgr.finalize_baseline(rec)

5. **Probes / late-trigger baseline replay** — never advance live streams::

       with replay_mgr.run_isolated(rec):
           restore_mutable_batch_fields(batch, rec.mutable_pre)
           # re-apply yaw jitter FROM rec.samples.yaw_now_jitter (no jit_gen)
           # re-encode, seed_waist_pr_to_sp, correct_quintic_init_arms, ...
           # evaluate F with recorded tc_kp/kd, suffix_offs, bptt payload
           F_trial = ...

   ``run_isolated`` restores named + global RNG to the post-baseline
   (live) positions on exit even if the probe accidentally draws.

6. **Do not** call ``al.update``, frontier/replay ingest, or ``roller.tick``
   inside probes.  Do not fetch a fresh BPTT second batch
   (:func:`_bptt_term` must consume ``rec.bptt``).

7. **Enablement gate**: if an enabled term cannot supply recorded samples or
   a replayable payload, raise :class:`ReplayIncompleteError` rather than
   silently omitting it from ``F``.
"""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, MutableMapping

import numpy as np
import torch

__all__ = [
    "ReplayIncompleteError",
    "GlobalTorchRngState",
    "RngBundle",
    "FrozenControls",
    "RealizedSamples",
    "BpttReplayPayload",
    "ReplayRecord",
    "ObjectiveReplayContext",
    "snapshot_torch_generator",
    "restore_torch_generator",
    "snapshot_numpy_generator",
    "restore_numpy_generator",
    "snapshot_python_random",
    "restore_python_random",
    "snapshot_global_torch_rng",
    "restore_global_torch_rng",
    "snapshot_stream",
    "restore_stream",
    "snapshot_streams",
    "restore_streams",
    "clone_dataclass_tensors",
    "snapshot_mutable_batch_fields",
    "restore_mutable_batch_fields",
]


class ReplayIncompleteError(RuntimeError):
    """Raised when an enabled objective path cannot be replayed bit-stably."""


# ----------------------------------------------------------------- snapshots --


@dataclass(frozen=True)
class GlobalTorchRngState:
    """CPU + all-device CUDA states from ``torch.get_rng_state`` / CUDA all."""

    cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...] | None = None


def snapshot_torch_generator(gen: torch.Generator) -> torch.Tensor:
    """Clone ``gen.get_state()`` so later draws cannot mutate the snapshot."""
    return gen.get_state().clone()


def restore_torch_generator(gen: torch.Generator, state: torch.Tensor) -> None:
    gen.set_state(state)


def snapshot_numpy_generator(rng: np.random.Generator) -> dict:
    """Deep-copy ``rng.bit_generator.state`` (BitGenerator state dict)."""
    return copy.deepcopy(rng.bit_generator.state)


def restore_numpy_generator(rng: np.random.Generator, state: Mapping) -> None:
    rng.bit_generator.state = copy.deepcopy(dict(state))


def snapshot_python_random(rng: random.Random) -> tuple:
    """``random.Random.getstate()`` (already an immutable-ish tuple tree)."""
    return rng.getstate()


def restore_python_random(rng: random.Random, state: tuple) -> None:
    rng.setstate(state)


def snapshot_global_torch_rng(*, capture_cuda: bool | None = None) -> GlobalTorchRngState:
    """Snapshot process-global torch RNG (CPU always; CUDA when available)."""
    cpu = torch.get_rng_state().clone()
    if capture_cuda is None:
        capture_cuda = torch.cuda.is_available()
    cuda: tuple[torch.Tensor, ...] | None = None
    if capture_cuda and torch.cuda.is_available():
        cuda = tuple(s.clone() for s in torch.cuda.get_rng_state_all())
    return GlobalTorchRngState(cpu=cpu, cuda=cuda)


def restore_global_torch_rng(state: GlobalTorchRngState) -> None:
    torch.set_rng_state(state.cpu)
    if state.cuda is not None and torch.cuda.is_available():
        # Pad / truncate defensively if device count changed (should not in-run).
        cur = torch.cuda.device_count()
        states = list(state.cuda)
        if len(states) < cur:
            # Re-seed missing devices from current state rather than crash;
            # callers that care should capture after all devices are visible.
            states = states + [
                torch.cuda.get_rng_state(i).clone() for i in range(len(states), cur)
            ]
        elif len(states) > cur:
            states = states[:cur]
        torch.cuda.set_rng_state_all(states)


def snapshot_stream(stream: Any) -> Any:
    """Dispatch snapshot for ``torch.Generator``, numpy, or ``random.Random``."""
    if isinstance(stream, torch.Generator):
        return snapshot_torch_generator(stream)
    if isinstance(stream, np.random.Generator):
        return snapshot_numpy_generator(stream)
    if isinstance(stream, random.Random):
        return snapshot_python_random(stream)
    raise TypeError(
        f"unsupported RNG stream type {type(stream)!r}; expected "
        "torch.Generator, numpy.random.Generator, or random.Random"
    )


def restore_stream(stream: Any, state: Any) -> None:
    if isinstance(stream, torch.Generator):
        restore_torch_generator(stream, state)
        return
    if isinstance(stream, np.random.Generator):
        restore_numpy_generator(stream, state)
        return
    if isinstance(stream, random.Random):
        restore_python_random(stream, state)
        return
    raise TypeError(f"unsupported RNG stream type {type(stream)!r}")


def snapshot_streams(streams: Mapping[str, Any]) -> dict[str, Any]:
    return {name: snapshot_stream(s) for name, s in streams.items()}


def restore_streams(streams: Mapping[str, Any], states: Mapping[str, Any]) -> None:
    for name, state in states.items():
        if name not in streams:
            raise KeyError(f"RNG state {name!r} has no live stream to restore")
        restore_stream(streams[name], state)


@dataclass
class RngBundle:
    """Named stream states plus optional global torch RNG."""

    named: dict[str, Any] = field(default_factory=dict)
    global_torch: GlobalTorchRngState | None = None


# ---------------------------------------------------------- frozen controls --


@dataclass
class FrozenControls:
    """Curriculum / AL controls ``C`` frozen for one search (§3 Bind F / §8).

    ``weights`` is the tensor dict produced by ``_wtensors`` (or ``None`` until
    filled).  ``al_lambda`` is ``al.weights()``; ``pen`` is ``al.pen``.
    """

    weights: Mapping[str, Any] | None = None
    al_lambda: Mapping[str, float] = field(default_factory=dict)
    pen: Mapping[str, float] = field(default_factory=dict)
    sfx_ramp: float = 0.0
    include_bptt: bool = False
    step: int = 0
    stage_a: bool = False
    bucket: int | None = None
    use_replay: bool = False
    # Extra frozen scalars the integrator may stash (w_tc, w_bptt, ...).
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RealizedSamples:
    """External random draws consumed by ``F`` — prefer these over re-RNG.

    All tensors should be ``detach()``'d (and typically left on-device).  Probes
    must feed these into TC / suffix / BPTT / yaw paths instead of calling
    ``sample_gains`` / ``suffix_offsets`` / ``torch.randn(..., generator=...)``.
    """

    tc_kp: torch.Tensor | None = None
    tc_kd: torch.Tensor | None = None
    tc_ki: torch.Tensor | None = None
    suffix_offs: torch.Tensor | None = None
    #: Additive yaw jitter noise (same shape as ``yaw_now``), not the sum.
    yaw_now_jitter: torch.Tensor | None = None
    # BPTT plant noise / gains (see ``_bptt_term``).
    bptt_kp: torch.Tensor | None = None
    bptt_kd: torch.Tensor | None = None
    bptt_ki: torch.Tensor | None = None
    bptt_e_rest: torch.Tensor | None = None
    bptt_q_noise: torch.Tensor | None = None
    bptt_qd_noise: torch.Tensor | None = None
    bptt_tip_vel_noise: torch.Tensor | None = None
    #: Catch-all for future realized draws (named explicitly in extras).
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BpttReplayPayload:
    """Frozen BPTT first/second batches — never re-sample inside a probe.

    Hold references to immutable pool storage when possible; clone only fields
    the baseline would overwrite in-place (see
    :func:`snapshot_mutable_batch_fields`).
    """

    batch1: dict[str, Any] | None = None
    batch2: dict[str, Any] | None = None
    mutable_pre1: dict[str, Any] = field(default_factory=dict)
    mutable_pre2: dict[str, Any] = field(default_factory=dict)
    src_is_replay: bool | None = None


@dataclass
class ReplayRecord:
    """One search / late-trigger objective replay context (plan ``replay_ctx``)."""

    primary_batch: dict[str, Any] | None = None
    #: Clones of fields ``train_phase2`` mutates before/during the forward.
    mutable_pre: dict[str, Any] = field(default_factory=dict)
    bptt: BpttReplayPayload | None = None
    controls: FrozenControls = field(default_factory=FrozenControls)
    samples: RealizedSamples = field(default_factory=RealizedSamples)
    #: RNG positions at :meth:`ObjectiveReplayContext.begin_record`.
    rng_pre: RngBundle | None = None
    #: RNG positions after the logical baseline forward
    #: (:meth:`ObjectiveReplayContext.finalize_baseline`).
    rng_post: RngBundle | None = None
    finalized: bool = False


# ----------------------------------------------------- mutable batch fields --

_MUTABLE_BATCH_KEYS = ("init", "state", "seed")


def clone_dataclass_tensors(obj: Any) -> Any:
    """Deep-clone tensor fields of a dataclass (e.g. ``StrikeContext``)."""
    if obj is None:
        return None
    if not hasattr(obj, "__dataclass_fields__"):
        raise TypeError(f"expected dataclass, got {type(obj)!r}")
    kwargs: dict[str, Any] = {}
    for name in obj.__dataclass_fields__:
        val = getattr(obj, name)
        if torch.is_tensor(val):
            kwargs[name] = val.detach().clone()
        else:
            kwargs[name] = val
    return type(obj)(**kwargs)


def snapshot_mutable_batch_fields(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Clone tensors that the live step may rewrite in-place.

    Covers ``init`` (``seed_waist_pr_to_sp`` / ``correct_quintic_init_arms``),
    optional ``state`` / ``seed``, ``context.yaw_now`` (jitter), and ``strike``
    (encoder SP reference rewrites — plan §14 pre-rewrite clone).
    """
    out: dict[str, Any] = {}
    for key in _MUTABLE_BATCH_KEYS:
        val = batch.get(key)
        if torch.is_tensor(val):
            out[key] = val.detach().clone()
    ctx = batch.get("context")
    if ctx is not None:
        yaw = getattr(ctx, "yaw_now", None)
        if torch.is_tensor(yaw):
            out["yaw_now"] = yaw.detach().clone()
    strike = batch.get("strike")
    if strike is not None and hasattr(strike, "__dataclass_fields__"):
        out["strike"] = clone_dataclass_tensors(strike)
    return out


def restore_mutable_batch_fields(
    batch: MutableMapping[str, Any], snap: Mapping[str, Any]
) -> None:
    """Write cloned pre-rewrite tensors back onto ``batch`` / ``context``."""
    for key in _MUTABLE_BATCH_KEYS:
        if key in snap:
            batch[key] = snap[key].clone()
    if "yaw_now" in snap:
        ctx = batch.get("context")
        if ctx is None:
            raise ReplayIncompleteError(
                "mutable snap has yaw_now but batch has no context"
            )
        ctx.yaw_now = snap["yaw_now"].clone()
    if "strike" in snap:
        batch["strike"] = clone_dataclass_tensors(snap["strike"])


# ----------------------------------------------------------- replay context --


class ObjectiveReplayContext:
    """Owns live stream handles and the begin / finalize / isolated protocol.

    Parameters
    ----------
    streams
        Mapping of names to live RNG objects (``gen``, ``sfx_gen``, ``jit_gen``,
        ``rng``, optionally ``roller_gen``, …).
    capture_global
        Also snapshot/restore process-global torch CPU/CUDA RNG (dropout / any
        bare ``torch.rand`` without a Generator).
    """

    def __init__(
        self,
        streams: Mapping[str, Any],
        *,
        capture_global: bool = True,
    ) -> None:
        self.streams: dict[str, Any] = dict(streams)
        self.capture_global = bool(capture_global)

    def register(self, name: str, stream: Any) -> None:
        self.streams[name] = stream

    def snapshot(self) -> RngBundle:
        return RngBundle(
            named=snapshot_streams(self.streams),
            global_torch=(
                snapshot_global_torch_rng() if self.capture_global else None
            ),
        )

    def restore(self, bundle: RngBundle) -> None:
        restore_streams(self.streams, bundle.named)
        if bundle.global_torch is not None:
            restore_global_torch_rng(bundle.global_torch)

    def begin_record(
        self,
        *,
        primary_batch: dict[str, Any] | None = None,
        controls: FrozenControls | None = None,
        bptt: BpttReplayPayload | None = None,
        snapshot_mutable: bool = True,
    ) -> ReplayRecord:
        """Open a replay record and capture pre-objective RNG positions.

        Typical hook: after the primary minibatch exists, before
        ``yaw_now`` jitter / TC gains / BPTT / suffix draws in
        :func:`~amortized_opt.train_cost.train_phase2`.
        """
        mutable: dict[str, Any] = {}
        if snapshot_mutable and primary_batch is not None:
            mutable = snapshot_mutable_batch_fields(primary_batch)
        return ReplayRecord(
            primary_batch=primary_batch,
            mutable_pre=mutable,
            bptt=bptt,
            controls=controls if controls is not None else FrozenControls(),
            samples=RealizedSamples(),
            rng_pre=self.snapshot(),
            rng_post=None,
            finalized=False,
        )

    def finalize_baseline(self, record: ReplayRecord) -> ReplayRecord:
        """Stamp post-baseline RNG — the live positions probes must restore to.

        Call once after the logical baseline ``loss + extra`` forward for this
        step.  Further probes use :meth:`run_isolated` so accidental draws do
        not leak past this mark.
        """
        if record.finalized:
            raise ReplayIncompleteError("finalize_baseline called twice on one record")
        record.rng_post = self.snapshot()
        record.finalized = True
        return record

    @contextmanager
    def run_isolated(
        self,
        record: ReplayRecord | None = None,
        *,
        restore_pre: bool = False,
        require_finalized: bool = True,
    ) -> Iterator[ReplayRecord | None]:
        """Run a probe without permanently advancing live RNG streams.

        On enter, snapshots the current (live) RNG.  If ``restore_pre`` and
        ``record.rng_pre`` are set, restores the pre-objective positions so a
        probe can re-draw (discouraged when realized samples exist).  On exit,
        restores the enter-time snapshot — which after
        :meth:`finalize_baseline` is the post-baseline live mark.

        Parameters
        ----------
        require_finalized
            When ``record`` is given, insist ``finalize_baseline`` already ran
            so probes restore to post-baseline rather than an ambiguous mid-step
            position.  Late-trigger baseline *re*-eval before finalize may pass
            ``require_finalized=False``.
        """
        if record is not None and require_finalized and not record.finalized:
            raise ReplayIncompleteError(
                "run_isolated(require_finalized=True) needs finalize_baseline first"
            )
        live = self.snapshot()
        try:
            if restore_pre:
                if record is None or record.rng_pre is None:
                    raise ReplayIncompleteError(
                        "restore_pre=True requires record.rng_pre"
                    )
                self.restore(record.rng_pre)
            yield record
        finally:
            self.restore(live)

    def run_isolated_fn(
        self,
        fn: Callable[[], Any],
        record: ReplayRecord | None = None,
        *,
        restore_pre: bool = False,
        require_finalized: bool = True,
    ) -> Any:
        """Convenience wrapper around :meth:`run_isolated`."""
        with self.run_isolated(
            record, restore_pre=restore_pre, require_finalized=require_finalized
        ):
            return fn()
