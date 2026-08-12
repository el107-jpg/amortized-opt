"""Episodic parameter-space-noise training: the Isaac-free half.

The plan (``docs/amortized_optimizer_plan.md``, "Final Fine Tune in RL via
IsaacSim", revised 2026-08-10) specifies an episodic scheme rather than a
step-level policy gradient:

    one epsilon per env per hit segment, redrawn at context switches; adaptive
    sigma measured as a physical tip deviation; rank hit segments by return and
    do an advantage-weighted regression of theta onto the trajectories the
    top-quantile perturbed policies produced, with the analytic differentiable
    cost mixed in as a simultaneous loss; optionally compose NES on the same
    antithetic rollouts.

Everything in this module is pure tensor bookkeeping and update math -- no
Isaac, no model, no schedule -- so the parts that are easy to get subtly wrong
(segment credit assignment, rank shaping, the antithetic NES estimator, the
advantage weighting) are unit-testable on CPU.  The driver that owns the Isaac
loop is ``isaac_rl/scripts/train_psn.py``.

Three things here are load bearing and are pinned by tests:

1. **A segment's return is the sum of the rewards of the ticks that segment
   actually produced**, and a segment is closed by the *fire* event, not by a
   fixed length.  Segments still open when the rollout window ends are
   **dropped**, not truncated -- a truncated segment's return is a biased
   sample of a shorter horizon and would systematically favour whatever the
   policy does early in a swing.

2. **Rank shaping is over segments, not over ticks.**  A long segment must not
   outvote a short one just for having accumulated more alive-bonus; the
   per-segment return is therefore reported both raw and per-tick, and the
   ranking uses the *raw* return (the whole point of the episodic formulation
   is that the hit outcome dominates), while the config exposes
   ``normalize_by_ticks`` for the A/B.

3. **The NES estimator needs the epsilon that was in force during the segment**,
   which is *not* the epsilon in the noise object after the rollout -- the loop
   redraws at the fire event that closes the segment.  ``SegmentBuffer.close``
   takes the epsilon row explicitly for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = [
    "PSNConfig",
    "SegmentBuffer",
    "SegmentStats",
    "TickBuffer",
    "advantage_weights",
    "centered_ranks",
    "nes_direction",
]


# ------------------------------------------------------------------ config --
@dataclass
class PSNConfig:
    """Update-rule knobs.  Rollout knobs live on the driver."""

    #: ``"distill"`` (the plan's preferred rule), ``"nes"`` (NES direction,
    #: realised as a distillation onto the shifted policy -- see
    #: ``train_psn.py``), or ``"both"``.
    update: str = "distill"

    # ---- distillation ----
    #: Keep segments whose return is in the top ``quantile`` of the batch.
    quantile: float = 0.25
    #: Advantage temperature for ``exp(beta * A_normalised)``.  0 = uniform
    #: weighting over the kept set (pure top-quantile behaviour cloning).
    beta: float = 2.0
    #: Cap on any single segment's weight after normalisation, as a multiple of
    #: the mean weight.  Without it one lucky segment with a +10 hit reward and
    #: a tiny batch can supply 80 % of the gradient.
    weight_clip: float = 5.0
    #: Rank the raw segment return (False) or the per-tick mean (True).
    normalize_by_ticks: bool = False

    # ---- loss mix ----
    #: Weight on the behaviour-cloning (trajectory regression) term.
    w_bc: float = 1.0
    #: Weight on the analytic ``FusedCost`` anchor.  The plan: "keep the
    #: analytic differentiable cost mixed in as a simultaneous loss so
    #: distillation cannot drift the model off constraint satisfaction."
    w_cost: float = 1.0
    #: Weight on the auxiliary heads (self-consistency + executed anchor).
    w_aux: float = 1.0
    #: Weight on the EXECUTED-state aux anchor inside ``w_aux``.
    w_aux_exec: float = 0.5
    #: Huber beta for the BC term, in radians.
    bc_huber_beta: float = 0.05

    # ---- NES ----
    #: Step size of the NES direction, expressed in units of sigma: the applied
    #: FiLM offset has RMS ``nes_step * sigma``.  Interpretable as "move the
    #: subspace mean by this many standard deviations of the search
    #: distribution", which is the only scale that is stable as sigma adapts.
    nes_step: float = 0.5

    # ---- optimisation ----
    lr: float = 1.0e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    #: Gradient steps per rollout iteration.
    steps_per_iter: int = 8
    #: Rows per gradient step.
    minibatch: int = 256


# ------------------------------------------------------------- rank shaping --
def centered_ranks(x: torch.Tensor) -> torch.Tensor:
    """Rank-transform ``x`` to evenly-spaced values in ``[-0.5, +0.5]``.

    The standard NES / OpenAI-ES utility transform.  It is not a nicety: raw
    returns here are dominated by a +10 discrete hit bonus, so an unshaped
    estimator is essentially a one-sample gradient whenever exactly one segment
    in the batch hits.  Ties are broken by index (``argsort`` of ``argsort``),
    which is fine for a utility -- adjacent ranks differ by ``1/(n-1)``.
    """
    x = x.reshape(-1)
    n = x.numel()
    if n == 0:
        return x.clone()
    if n == 1:
        return torch.zeros_like(x)
    order = torch.argsort(x)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n, device=x.device)
    return ranks.to(x.dtype) / (n - 1) - 0.5


def advantage_weights(
    returns: torch.Tensor,
    *,
    quantile: float = 0.25,
    beta: float = 2.0,
    weight_clip: float = 5.0,
) -> torch.Tensor:
    """Per-segment weights for advantage-weighted regression.  Sums to ``n``.

    Segments below the ``1 - quantile`` return quantile get exactly zero, so
    the regression only ever imitates winners (the plan's "distill the
    winners"); above it, the weight is ``exp(beta * A)`` with ``A`` the return
    standardised over the *kept* set, then **clipped at ``weight_clip``** so a
    single outlier cannot own the update.

    The kept weights have mean 1, and that is the last operation: an earlier
    version rescaled to mean 1 over all ``n`` segments *after* clipping, which
    silently multiplied the surviving weights back up by ``n / n_kept`` and
    handed a lone +1e6 outlier 32x the clip.  There is nothing to rescale for --
    the BC term is a weight-NORMALISED mean (``(loss * w).sum() / w.sum()``), so
    only the relative weights matter and ``w_bc`` is quantile-invariant anyway.
    """
    r = returns.reshape(-1).float()
    n = r.numel()
    if n == 0:
        return r.clone()
    q = float(min(max(quantile, 1e-6), 1.0))
    k = max(1, int(round(q * n)))
    thresh = torch.topk(r, k).values.min()
    keep = r >= thresh
    w = torch.zeros_like(r)
    kept = r[keep]
    if kept.numel() > 1:
        a = (kept - kept.mean()) / kept.std().clamp_min(1e-6)
    else:
        a = torch.zeros_like(kept)
    wk = torch.exp(float(beta) * a)
    wk = wk / wk.mean().clamp_min(1e-12)
    if weight_clip and weight_clip > 0:
        wk = wk.clamp(max=float(weight_clip))
    w[keep] = wk
    return w


def nes_direction(
    eps: torch.Tensor,          # (n_seg, dim) the epsilon each segment ran under
    returns: torch.Tensor,      # (n_seg,)
    *,
    sigma: float,
    step: float = 0.5,
) -> torch.Tensor:
    """``(dim,)`` NES direction, rank-shaped, normalised to ``step * sigma`` RMS.

    The textbook estimator is ``alpha / (N sigma) * sum_i R~_i eps_i``.  Two
    deviations, both deliberate:

    * the returns are **rank-shaped** (:func:`centered_ranks`) before the
      weighted sum, per the plan ("antithetic eps pairs across envs,
      rank-shaped returns");
    * the result is **renormalised** to a fixed RMS of ``step * sigma`` rather
      than scaled by ``alpha / (N sigma)``.  The raw estimator's magnitude
      depends on ``N``, on how many segments hit, and on ``sigma`` itself,
      which is adapting; a direction plus an explicit step in units of sigma
      is the only parameterisation that stays meaningful across those.
    """
    if eps.numel() == 0:
        return torch.zeros(eps.shape[-1] if eps.dim() > 1 else 0,
                           device=eps.device, dtype=eps.dtype)
    u = centered_ranks(returns.to(eps.dtype)).to(eps.device)
    d = (u.unsqueeze(-1) * eps).sum(dim=0)
    rms = d.square().mean().sqrt()
    if float(rms) <= 1e-12:
        return torch.zeros_like(d)
    return d * (float(step) * float(sigma) / rms)


# ----------------------------------------------------------------- buffers --
class TickBuffer:
    """Grow-once, preallocated store of per-(tick, env) rows.

    Keys are declared implicitly by the first :meth:`append`; every later
    append must carry the same keys with the same trailing shape.  Rows are
    stored in the dtype they arrive in, except floating rows which are cast to
    ``store_dtype`` (fp16 by default -- the buffer is ~4 KB/row at H=50 and a
    128-env x 120-tick iteration is 15 360 rows).
    """

    def __init__(self, capacity: int, *, device, store_dtype=torch.float16,
                 fp32_prefixes: tuple[str, ...] = ("ctx.", "st.")):
        self.capacity = int(capacity)
        self.device = device
        self.store_dtype = store_dtype
        #: Keys under these prefixes stay fp32.  The context tokens and the
        #: strike rows carry METRE-scale positions, and fp16 quantises those at
        #: ~0.5 mm -- the same order as the in-plane miss the whole objective is
        #: trying to resolve.  The trajectory / init / state rows are radian
        #: scale and fp16 there is ~1e-4 rad, which is well under the plant's
        #: own tracking error.
        self.fp32_prefixes = tuple(fp32_prefixes)
        self.n = 0
        self.data: dict[str, torch.Tensor] = {}

    def _dtype_for(self, key: str, v: torch.Tensor) -> torch.dtype:
        if not v.is_floating_point():
            return v.dtype
        if key.startswith(self.fp32_prefixes):
            return torch.float32
        return self.store_dtype

    def append(self, row: dict[str, torch.Tensor]) -> tuple[int, int]:
        """Append ``B`` rows.  Returns the ``[lo, hi)`` slice they occupy."""
        B = next(iter(row.values())).shape[0]
        if self.n + B > self.capacity:
            raise RuntimeError(f"TickBuffer overflow: {self.n} + {B} > {self.capacity}")
        for k, v in row.items():
            if k not in self.data:
                dt = self._dtype_for(k, v)
                self.data[k] = torch.empty((self.capacity,) + tuple(v.shape[1:]),
                                           device=self.device, dtype=dt)
            self.data[k][self.n: self.n + B] = v.to(self.data[k].dtype)
        lo, hi = self.n, self.n + B
        self.n = hi
        return lo, hi

    def gather(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.data.items()}

    def clear(self) -> None:
        self.n = 0


@dataclass
class SegmentStats:
    """What one closed hit segment contributes to the update."""

    env: int
    ret: float
    ticks: int
    rows: torch.Tensor      # indices into the TickBuffer
    eps: torch.Tensor       # (dim,) the epsilon that produced it
    hits: float = 0.0
    misses: float = 0.0


class SegmentBuffer:
    """Per-env accumulation of returns and row indices, closed on the fire event.

    The driver calls :meth:`add` once per tick with the per-env reward and the
    ``[lo, hi)`` slice the tick's rows occupy in the :class:`TickBuffer`, then
    :meth:`close` with the envs whose segment just ended and the epsilon rows
    that were in force.  Segments still open at the end of the rollout are
    discarded by :meth:`harvest` -- see the module docstring.
    """

    def __init__(self, n_envs: int, *, device):
        self.n_envs = int(n_envs)
        self.device = device
        self._ret = torch.zeros(self.n_envs, device=device)
        self._ticks = torch.zeros(self.n_envs, device=device)
        self._hits = torch.zeros(self.n_envs, device=device)
        self._miss = torch.zeros(self.n_envs, device=device)
        self._rows: list[list[int]] = [[] for _ in range(self.n_envs)]
        self.closed: list[SegmentStats] = []

    def add(self, reward: torch.Tensor, lo: int, *, hits=None, misses=None) -> None:
        """``reward`` is ``(n_envs,)``; row ``lo + b`` belongs to env ``b``."""
        self._ret += reward.detach().to(self._ret.dtype).reshape(-1)
        self._ticks += 1
        if hits is not None:
            self._hits += hits.detach().to(self._hits.dtype).reshape(-1)
        if misses is not None:
            self._miss += misses.detach().to(self._miss.dtype).reshape(-1)
        for b in range(self.n_envs):
            self._rows[b].append(lo + b)

    def close(self, envs: torch.Tensor, eps: torch.Tensor) -> int:
        """Close the segments of ``envs``; ``eps`` is the FULL ``(n_envs, dim)``.

        Returns the number of segments closed.  Envs with no accumulated ticks
        are skipped (a fire on the very first tick of a fresh segment).
        """
        ids = torch.as_tensor(envs, device=self.device).reshape(-1).long()
        if ids.numel() == 0:
            return 0
        ret = self._ret.detach().cpu()
        tks = self._ticks.detach().cpu()
        hits = self._hits.detach().cpu()
        miss = self._miss.detach().cpu()
        n = 0
        for b in ids.detach().cpu().tolist():
            b = int(b)
            if int(tks[b]) == 0:
                continue
            self.closed.append(SegmentStats(
                env=b, ret=float(ret[b]), ticks=int(tks[b]),
                rows=torch.tensor(self._rows[b], dtype=torch.long),
                eps=eps[b].detach().clone(),
                hits=float(hits[b]), misses=float(miss[b]),
            ))
            n += 1
            self._ret[b] = 0.0
            self._ticks[b] = 0.0
            self._hits[b] = 0.0
            self._miss[b] = 0.0
            self._rows[b] = []
        return n

    def harvest(self, cfg: PSNConfig):
        """``(rows, row_weights, seg_returns, seg_eps)`` for the update.

        ``rows`` indexes the :class:`TickBuffer`; ``row_weights`` is the
        segment's advantage weight broadcast to each of its ticks.  Open
        segments are dropped.
        """
        segs = self.closed
        if not segs:
            z = torch.zeros(0, device=self.device)
            return (torch.zeros(0, dtype=torch.long, device=self.device), z, z,
                    torch.zeros(0, 0, device=self.device))
        ret = torch.tensor([s.ret for s in segs], device=self.device)
        tks = torch.tensor([float(s.ticks) for s in segs], device=self.device)
        rank_on = ret / tks.clamp_min(1.0) if cfg.normalize_by_ticks else ret
        w = advantage_weights(rank_on, quantile=cfg.quantile, beta=cfg.beta,
                              weight_clip=cfg.weight_clip)
        rows, rw = [], []
        for s, wi in zip(segs, w.detach().cpu().tolist()):
            if wi <= 0.0:
                continue
            rows.append(s.rows.to(self.device))
            rw.append(torch.full((s.rows.numel(),), float(wi), device=self.device))
        if not rows:
            z = torch.zeros(0, device=self.device)
            return (torch.zeros(0, dtype=torch.long, device=self.device), z, ret,
                    torch.stack([s.eps for s in segs]))
        return (torch.cat(rows), torch.cat(rw), ret,
                torch.stack([s.eps for s in segs]))

    def reset(self) -> None:
        self.closed = []
        self._ret.zero_()
        self._ticks.zero_()
        self._hits.zero_()
        self._miss.zero_()
        self._rows = [[] for _ in range(self.n_envs)]
