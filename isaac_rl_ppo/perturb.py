"""Per-env low-rank parameter-space perturbation of the trajectory decoder.

Section "Final Fine Tune in RL via IsaacSim", revised 2026-08-10, item 1:

    What to perturb: a low-rank subspace, not raw weights.  LoRA-style
    rank-4-16 deltas on the decoder's attention out-projections and FFNs, or
    cheaper, per-layer FiLM vectors (elementwise gain/bias) -- ~10k-200k
    effective dims instead of ~10M.  This is also what makes per-env
    perturbation computationally feasible in Isaac: low-rank per-env deltas
    apply as batched matmuls inside one shared forward pass.

Why parameter space and not output space: the model is a *refinement operator*
and the temporal-consistency loss explicitly trains it to contract, so additive
trajectory noise is denoised away within one or two replans.  A perturbed theta'
is a different but self-consistent optimizer whose iterative refinement
converges to its own coherently different solution.

Design, as implemented here
---------------------------

* **Two mechanisms**, both applied *inside one shared forward pass* over the
  whole ``num_envs`` batch, so there is exactly one set of weights on the GPU:

  - :class:`FiLMPerturbation` -- per decoder block, an elementwise gain/bias
    pair on the residual stream: ``x <- x * (1 + sigma * g_b) + sigma * b_b``
    with ``(g, b)`` drawn per *env*.  ``2 * d_model`` coordinates per block,
    ``2 * L * d_model`` total (4 608 at the champion's L=6, d=384).  The
    cheaper option, and the plan's first choice.
  - :class:`LoRAPerturbation` -- rank-``r`` deltas on the decoder's attention
    out-projections and the FFN ``w3`` (the two "out" matrices of every
    block): ``y <- y + sigma * (x @ A_b) @ B_b``.  ``r * (d_in + d_out)``
    coordinates per site.  The alternative, for when the FiLM subspace turns
    out to be too cramped to express contact-timing changes (the diagnostic in
    :func:`deviation_report` is what says so).

* **Nothing in ``amortized_opt`` is modified.**  Both mechanisms attach with
  ``torch.nn.Module`` forward hooks on the existing
  :class:`~amortized_opt.model.DecoderBlock` / ``nn.Linear`` submodules, so the
  same checkpoint runs perturbed or unperturbed and the machinery detaches
  cleanly.  The *context encoder stays unperturbed* (plan item 1).

* **One epsilon per env, redrawn at context switches** (plan item 2), with
  **antithetic pairs**: env ``2k`` and ``2k+1`` carry ``+eps`` and ``-eps`` of
  the same draw.  :meth:`ParameterNoise.redraw` takes the rows to redraw, so a
  hand's re-goal redraws only the envs whose segment ended.

* **Adaptive sigma, Plappert-style, measured physically** (plan item 3):
  :class:`AdaptiveSigma` adapts a scalar multiplier so that a *measured*
  distance -- here the executed-prefix / terminal tip deviation in millimetres,
  supplied by the caller -- holds a target.  Only the machinery and the
  diagnostics live here; the tuning is the training stage's job.

Everything in this module is Isaac-free and unit-testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = [
    "AdaptiveSigma",
    "AdaptiveSigmaConfig",
    "FiLMPerturbation",
    "LoRAPerturbation",
    "NoiseConfig",
    "ParameterNoise",
    "antithetic_normal",
    "build_perturbation",
    "deviation_report",
]


# ------------------------------------------------------------------ config --
@dataclass
class NoiseConfig:
    """How the per-env parameter noise is drawn and applied."""

    #: ``"film"`` (default, cheap) or ``"lora"``.
    mode: str = "film"
    #: LoRA rank; ignored for FiLM.  Plan says 4-16.
    rank: int = 8
    #: Base scale.  The live scale is ``sigma * AdaptiveSigma.scale``.
    sigma: float = 0.0
    #: Draw env ``2k+1`` as the negation of env ``2k``.  Halves the variance of
    #: the NES gradient estimate and makes the FD interpretation exact.
    antithetic: bool = True
    #: Perturb only these decoder blocks (indices).  ``None`` = all of them.
    #: The plan's residual gaps are contact-adjacent, so restricting to the
    #: last few blocks is a cheap way to narrow the subspace; kept as a knob.
    blocks: tuple[int, ...] | None = None
    #: LoRA sites within a block.  ``attn`` = self-attention out-projection,
    #: ``cross`` = cross-attention out-projection, ``ffn`` = the SwiGLU ``w3``.
    lora_sites: tuple[str, ...] = ("attn", "cross", "ffn")
    #: RNG seed for the internal generator.
    seed: int = 0


@dataclass
class AdaptiveSigmaConfig:
    """Plappert-style adaptation of the scale, on a *physical* distance."""

    #: Target for the measured distance, millimetres of tip deviation at the
    #: hit.  The plan says "a few mm".
    target_mm: float = 3.0
    #: Multiplicative step.  Deliberately small: "adapt slowly ... sigma
    #: adaptation on a shifting policy can oscillate."
    adapt_rate: float = 1.01
    scale_min: float = 1e-3
    scale_max: float = 1e3
    #: EMA over the measured distance before comparing to the target, so one
    #: unlucky segment does not move the scale.
    ema: float = 0.9


# --------------------------------------------------------------- utilities --
def antithetic_normal(
    n: int,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
    antithetic: bool = True,
) -> torch.Tensor:
    """``(n, *shape)`` standard normals, optionally in antithetic pairs.

    With ``antithetic=True`` row ``2k+1`` is exactly ``-row 2k``.  An odd ``n``
    leaves the last row an independent draw.
    """
    if not antithetic:
        return torch.randn((n,) + tuple(shape), generator=generator,
                           device=device, dtype=dtype)
    half = (n + 1) // 2
    base = torch.randn((half,) + tuple(shape), generator=generator,
                       device=device, dtype=dtype)
    out = torch.stack([base, -base], dim=1).reshape((2 * half,) + tuple(shape))
    return out[:n]


def _pair_partner(rows: torch.Tensor) -> torch.Tensor:
    """For antithetic bookkeeping: the partner index of every row in ``rows``."""
    return rows ^ 1


# ----------------------------------------------------------------- backend --
class _Backend:
    """Common surface for the two perturbation mechanisms."""

    n_envs: int
    dim: int  # number of epsilon coordinates PER ENV

    def flat(self) -> torch.Tensor:  # (n_envs, dim)
        raise NotImplementedError

    def set_flat(self, eps: torch.Tensor) -> None:
        raise NotImplementedError

    def set_scale(self, scale: torch.Tensor | float) -> None:
        raise NotImplementedError

    def attach(self) -> None:
        raise NotImplementedError

    def detach(self) -> None:
        raise NotImplementedError


class FiLMPerturbation(_Backend):
    """Per-block elementwise gain/bias on the decoder's residual stream.

    ``x <- x * (1 + s_b * g_b) + s_b * beta_b`` after every perturbed
    :class:`DecoderBlock`, with ``(g, beta)`` a per-env draw and ``s_b`` the
    per-env live scale.  The tokens of one env all share its ``(g, beta)``, so
    the perturbation is a property of the *optimizer*, not of the waypoint --
    which is the whole point of parameter-space noise.

    The gain is centred on 1 and the bias on 0, so ``sigma = 0`` reproduces the
    unperturbed network bit-for-bit (there is a fast path that skips the hook
    body entirely in that case).
    """

    def __init__(self, decoder: nn.Module, n_envs: int, *, blocks=None,
                 device=None, dtype: torch.dtype = torch.float32):
        self.decoder = decoder
        self.n_envs = int(n_envs)
        self.d = int(decoder.cfg.d_model)
        n_blocks = len(decoder.blocks)
        self.block_ids = (
            tuple(range(n_blocks)) if blocks is None
            else tuple(int(b) % n_blocks for b in blocks)
        )
        self.L = len(self.block_ids)
        self.dim = 2 * self.L * self.d
        self.device = device
        self.dtype = dtype
        self._eps = torch.zeros(self.n_envs, self.dim, device=device, dtype=dtype)
        self._scale = torch.zeros(self.n_envs, 1, device=device, dtype=dtype)
        self._handles: list = []
        self._active = False

    # -- epsilon plumbing --
    def flat(self) -> torch.Tensor:
        return self._eps

    def set_flat(self, eps: torch.Tensor) -> None:
        if eps.shape != self._eps.shape:
            raise ValueError(f"expected {tuple(self._eps.shape)}, got {tuple(eps.shape)}")
        self._eps = eps.to(self._eps.device, self._eps.dtype)

    def set_scale(self, scale) -> None:
        s = torch.as_tensor(scale, device=self._scale.device, dtype=self._scale.dtype)
        self._scale = s.reshape(-1, 1).expand(self.n_envs, 1).contiguous() \
            if s.numel() in (1, self.n_envs) else s
        self._active = bool(torch.any(self._scale != 0.0))

    def _gb(self, k: int):
        """``(gain, bias)`` views of the flat epsilon for the ``k``-th block."""
        d = self.d
        off = 2 * k * d
        g = self._eps[:, off: off + d]
        b = self._eps[:, off + d: off + 2 * d]
        return g, b

    # -- hooks --
    def attach(self) -> None:
        self.detach()
        for k, bi in enumerate(self.block_ids):
            blk = self.decoder.blocks[bi]
            self._handles.append(blk.register_forward_hook(self._make_hook(k)))

    def _make_hook(self, k: int):
        def hook(_mod, _inp, out):
            if not self._active:
                return out
            g, b = self._gb(k)
            s = self._scale.to(out.dtype)
            gain = 1.0 + s * g.to(out.dtype)
            bias = s * b.to(out.dtype)
            # (B, T, d) * (B, 1, d)
            return out * gain.unsqueeze(1) + bias.unsqueeze(1)
        return hook

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class LoRAPerturbation(_Backend):
    """Per-env rank-``r`` deltas on the decoder's "out" projections.

    ``y <- y + s_b * ((x @ A_b) @ B_b) / sqrt(r)`` at each site, where
    ``A_b: (d_in, r)`` and ``B_b: (r, d_out)`` are per-env draws.  ``x`` is the
    hooked ``nn.Linear``'s input, so the delta is exactly a rank-``r`` additive
    weight perturbation ``W + s A B^T / sqrt(r)``, expressed as two batched
    matmuls instead of ``n_envs`` weight copies.

    The ``1/sqrt(r)`` normalization makes the induced weight-perturbation scale
    comparable across ranks, so the same ``sigma`` means roughly the same thing
    at r=4 and r=16.
    """

    _SITES = {
        "attn": ("self_attn", "o_proj"),
        "cross": ("cross_attn", "o_proj"),
        "ffn": ("ffn", "w3"),
    }

    def __init__(self, decoder: nn.Module, n_envs: int, *, rank: int = 8,
                 blocks=None, sites=("attn", "cross", "ffn"),
                 device=None, dtype: torch.dtype = torch.float32):
        self.decoder = decoder
        self.n_envs = int(n_envs)
        self.rank = int(rank)
        n_blocks = len(decoder.blocks)
        self.block_ids = (
            tuple(range(n_blocks)) if blocks is None
            else tuple(int(b) % n_blocks for b in blocks)
        )
        self.device = device
        self.dtype = dtype
        self.sites: list[tuple[nn.Linear, int, int]] = []
        for bi in self.block_ids:
            blk = decoder.blocks[bi]
            for s in sites:
                if s not in self._SITES:
                    raise ValueError(f"unknown LoRA site {s!r}")
                attr, leaf = self._SITES[s]
                lin = getattr(getattr(blk, attr), leaf)
                self.sites.append((lin, lin.in_features, lin.out_features))
        self.dim = sum(self.rank * (i + o) for _l, i, o in self.sites)
        self._eps = torch.zeros(self.n_envs, self.dim, device=device, dtype=dtype)
        self._scale = torch.zeros(self.n_envs, 1, device=device, dtype=dtype)
        self._handles: list = []
        self._active = False
        # precompute the flat offsets
        self._slices = []
        off = 0
        for _l, i, o in self.sites:
            na, nb = self.rank * i, self.rank * o
            self._slices.append((off, off + na, off + na + nb, i, o))
            off += na + nb

    def flat(self) -> torch.Tensor:
        return self._eps

    def set_flat(self, eps: torch.Tensor) -> None:
        if eps.shape != self._eps.shape:
            raise ValueError(f"expected {tuple(self._eps.shape)}, got {tuple(eps.shape)}")
        self._eps = eps.to(self._eps.device, self._eps.dtype)

    def set_scale(self, scale) -> None:
        s = torch.as_tensor(scale, device=self._scale.device, dtype=self._scale.dtype)
        self._scale = s.reshape(-1, 1).expand(self.n_envs, 1).contiguous() \
            if s.numel() in (1, self.n_envs) else s
        self._active = bool(torch.any(self._scale != 0.0))

    def _ab(self, k: int):
        a0, a1, b1, i, o = self._slices[k]
        A = self._eps[:, a0:a1].view(self.n_envs, i, self.rank)
        B = self._eps[:, a1:b1].view(self.n_envs, self.rank, o)
        return A, B

    def attach(self) -> None:
        self.detach()
        for k, (lin, _i, _o) in enumerate(self.sites):
            self._handles.append(lin.register_forward_hook(self._make_hook(k)))

    def _make_hook(self, k: int):
        r_inv = self.rank ** -0.5

        def hook(_mod, inp, out):
            if not self._active:
                return out
            x = inp[0]
            A, B = self._ab(k)
            A, B = A.to(out.dtype), B.to(out.dtype)
            s = self._scale.to(out.dtype)
            lead = x.shape[:-1]
            xf = x.reshape(x.shape[0], -1, x.shape[-1])       # (B, T, d_in)
            delta = torch.bmm(torch.bmm(xf, A), B)            # (B, T, d_out)
            delta = delta.reshape(*lead, out.shape[-1])
            sh = s.reshape(s.shape[0], *([1] * (delta.dim() - 2)), 1)
            return out + r_inv * sh * delta
        return hook

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def build_perturbation(model, n_envs: int, cfg: NoiseConfig, *,
                       device=None, dtype: torch.dtype = torch.float32) -> _Backend:
    """``model`` is an :class:`~amortized_opt.model.AmortizedOptimizer`."""
    dec = model.module.decoder if hasattr(model, "module") else model.decoder
    if cfg.mode == "film":
        return FiLMPerturbation(dec, n_envs, blocks=cfg.blocks, device=device, dtype=dtype)
    if cfg.mode == "lora":
        return LoRAPerturbation(dec, n_envs, rank=cfg.rank, blocks=cfg.blocks,
                                sites=cfg.lora_sites, device=device, dtype=dtype)
    raise ValueError(f"mode must be film|lora, got {cfg.mode!r}")


# --------------------------------------------------------- adaptive sigma ---
class AdaptiveSigma:
    """Plappert-style adaptation of a scalar multiplier on a measured distance.

    ``observe(d_mm)`` takes the *physical* deviation the caller measured -- the
    plan says "run perturbed and unperturbed policies on the same states,
    measure deviation of the executed prefix and terminal tip position" -- EMAs
    it, and nudges the scale up when the deviation is under target and down
    when it is over.  The step is multiplicative and small, so the scale moves
    on the timescale of hundreds of segments rather than of one.
    """

    def __init__(self, cfg: AdaptiveSigmaConfig | None = None, scale: float = 1.0):
        self.cfg = cfg or AdaptiveSigmaConfig()
        self.scale = float(scale)
        self.ema_mm: float | None = None
        self.n_obs = 0

    def observe(self, distance_mm: float) -> float:
        c = self.cfg
        d = float(distance_mm)
        if not (d == d):  # NaN
            return self.scale
        self.ema_mm = d if self.ema_mm is None else c.ema * self.ema_mm + (1 - c.ema) * d
        self.n_obs += 1
        if self.ema_mm < c.target_mm:
            self.scale *= c.adapt_rate
        else:
            self.scale /= c.adapt_rate
        self.scale = float(min(max(self.scale, c.scale_min), c.scale_max))
        return self.scale

    def state_dict(self) -> dict:
        return {"scale": self.scale, "ema_mm": self.ema_mm, "n_obs": self.n_obs}

    def load_state_dict(self, d: dict) -> None:
        self.scale = float(d["scale"])
        self.ema_mm = d.get("ema_mm")
        self.n_obs = int(d.get("n_obs", 0))


# ------------------------------------------------------------- the driver ---
@dataclass
class ParameterNoise:
    """One epsilon per env, redrawn at hit-segment boundaries.

    Usage inside the env loop::

        noise = ParameterNoise.build(model, num_envs, NoiseConfig(mode="film",
                                                                  sigma=0.02))
        noise.attach()
        ...
        noise.redraw(switched_env_ids)   # at every context switch
        out = model(init, state, enc, masks)   # ONE shared forward, all envs
    """

    backend: _Backend
    cfg: NoiseConfig
    adaptive: AdaptiveSigma = field(default_factory=AdaptiveSigma)
    generator: torch.Generator | None = None
    #: Per-env override of the base sigma.  Envs with 0 run the clean policy --
    #: this is how the "periodic unperturbed eval episodes" of plan item 3 and
    #: the antithetic reference are expressed without a second model copy.
    env_sigma: torch.Tensor | None = None

    @classmethod
    def build(cls, model, n_envs: int, cfg: NoiseConfig, *, device=None,
              dtype: torch.dtype = torch.float32) -> "ParameterNoise":
        backend = build_perturbation(model, n_envs, cfg, device=device, dtype=dtype)
        gen = torch.Generator(device=device or "cpu")
        gen.manual_seed(int(cfg.seed))
        env_sigma = torch.full((n_envs,), float(cfg.sigma), device=device, dtype=dtype)
        obj = cls(backend=backend, cfg=cfg, generator=gen, env_sigma=env_sigma)
        obj.redraw(None)
        obj._push_scale()
        return obj

    # -- lifecycle --
    def attach(self) -> None:
        self.backend.attach()

    def detach(self) -> None:
        self.backend.detach()

    @property
    def n_envs(self) -> int:
        return self.backend.n_envs

    @property
    def dim(self) -> int:
        return self.backend.dim

    def epsilon(self) -> torch.Tensor:
        """``(n_envs, dim)`` -- the NES coordinates, for the update rule."""
        return self.backend.flat()

    def set_sigma(self, sigma, envs: torch.Tensor | None = None) -> None:
        """Set the base sigma, globally or on a subset (0 = clean policy)."""
        s = torch.as_tensor(sigma, device=self.env_sigma.device,
                            dtype=self.env_sigma.dtype)
        if envs is None:
            self.env_sigma = s.reshape(-1).expand(self.n_envs).contiguous() \
                if s.numel() in (1, self.n_envs) else s
        else:
            self.env_sigma = self.env_sigma.clone()
            self.env_sigma[envs] = s
        self._push_scale()

    def _push_scale(self) -> None:
        self.backend.set_scale(self.env_sigma * float(self.adaptive.scale))

    def observe_deviation(self, distance_mm: float) -> float:
        """Feed the adaptive scheme; returns the new adaptive multiplier."""
        out = self.adaptive.observe(distance_mm)
        self._push_scale()
        return out

    # -- the draw --
    def redraw(self, envs: torch.Tensor | None) -> None:
        """Draw a fresh epsilon for ``envs`` (``None`` = all of them).

        Antithetic pairing is over the FIXED env index parity, so the partner
        of env ``2k`` is always env ``2k+1``.  Redrawing one member of a pair
        therefore also rewrites the other -- otherwise the pairing silently
        rots as segments end at different times, and the NES estimator loses
        the variance reduction it was set up for.
        """
        eps = self.backend.flat()
        dim = self.backend.dim
        dev, dt = eps.device, eps.dtype
        if envs is None:
            new = antithetic_normal(self.n_envs, (dim,), generator=self.generator,
                                    device=dev, dtype=dt,
                                    antithetic=self.cfg.antithetic)
            self.backend.set_flat(new)
            return
        envs = torch.as_tensor(envs, device=dev).reshape(-1).long()
        if envs.numel() == 0:
            return
        eps = eps.clone()
        if self.cfg.antithetic:
            # expand to whole pairs, then draw one base per pair
            pairs = torch.unique(envs // 2)
            base = torch.randn((pairs.numel(), dim), generator=self.generator,
                               device=dev, dtype=dt)
            lo = 2 * pairs
            hi = lo + 1
            keep = hi < self.n_envs
            eps[lo] = base
            eps[hi[keep]] = -base[keep]
        else:
            eps[envs] = torch.randn((envs.numel(), dim), generator=self.generator,
                                    device=dev, dtype=dt)
        self.backend.set_flat(eps)


# ------------------------------------------------------------ diagnostics ---
def deviation_report(
    traj_clean: torch.Tensor,
    traj_pert: torch.Tensor,
    *,
    elem_mask: torch.Tensor | None = None,
    hit_index: torch.Tensor | None = None,
    joint_names=None,
    tip_clean: torch.Tensor | None = None,
    tip_pert: torch.Tensor | None = None,
) -> dict:
    """Where does the achievable deviation concentrate?  (plan item 1)

        "check where along the trajectory and across joints the achievable
        deviation concentrates -- it should be wrist-heavy and contact-adjacent,
        not all mid-swing."

    ``traj_*`` are ``(B, H, n)`` plans from the same context, one clean and one
    perturbed.  Returns

    * ``per_joint_rad`` -- ``(n,)`` RMS deviation per joint,
    * ``per_group_rad`` -- the same pooled into waist / shoulder / elbow / wrist,
    * ``wrist_share``   -- wrist energy / total, the "wrist-heavy" number,
    * ``by_horizon_rad``-- ``(H,)`` RMS deviation per waypoint index,
    * ``contact_share`` -- energy in the last 5 waypoints before each hand's
      terminal / total, the "contact-adjacent" number (needs ``hit_index``),
    * ``tip_mm``        -- terminal tip deviation, if the caller supplies the
      two ``(B, 3)`` tip positions (this is the quantity ``AdaptiveSigma``
      wants).
    """
    from amortized_opt.constants import G1_UPPER_BODY_JOINTS

    names = tuple(joint_names or G1_UPPER_BODY_JOINTS)
    d = (traj_pert - traj_clean)
    m = torch.ones_like(d) if elem_mask is None else elem_mask.to(d.dtype)
    d = d * m
    e = d.square()
    denom = m.sum().clamp_min(1.0)

    per_joint = torch.sqrt(e.sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp_min(1.0))
    by_h = torch.sqrt(e.sum(dim=(0, 2)) / m.sum(dim=(0, 2)).clamp_min(1.0))

    groups = {
        "waist": [i for i, n in enumerate(names) if n.startswith("waist_")],
        "shoulder": [i for i, n in enumerate(names) if "shoulder" in n],
        "elbow": [i for i, n in enumerate(names) if "elbow" in n],
        "wrist": [i for i, n in enumerate(names) if "wrist" in n],
    }
    grp_energy = {k: float(e[:, :, v].sum()) for k, v in groups.items()}
    total = max(sum(grp_energy.values()), 1e-30)
    out = {
        "per_joint_rad": per_joint.detach().cpu(),
        "per_group_rad": {
            k: float((e[:, :, v].sum() / max(m[:, :, v].sum().item(), 1.0)) ** 0.5)
            for k, v in groups.items()
        },
        "group_energy_share": {k: v / total for k, v in grp_energy.items()},
        "wrist_share": grp_energy["wrist"] / total,
        "by_horizon_rad": by_h.detach().cpu(),
        "rms_rad": float((e.sum() / denom) ** 0.5),
    }
    if hit_index is not None:
        H = traj_clean.shape[1]
        k = hit_index.reshape(traj_clean.shape[0], -1).max(dim=1).values.clamp(1, H)
        idx = torch.arange(H, device=d.device).view(1, H)
        near = ((idx >= (k.view(-1, 1) - 5)) & (idx <= k.view(-1, 1))).to(e.dtype)
        out["contact_share"] = float(
            (e.sum(dim=2) * near).sum() / max(float(e.sum()), 1e-30)
        )
        out["contact_frac_of_grid"] = float(near.mean())
    if tip_clean is not None and tip_pert is not None:
        out["tip_mm"] = float(1000.0 * (tip_pert - tip_clean).norm(dim=-1).mean())
    return out
