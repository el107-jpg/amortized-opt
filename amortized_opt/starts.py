"""Start-state sampler for the direct-cost training distribution (section 5).

The plan's "Inputs for the direct cost minimization":

    the starting points will be mostly (probably like 90% of them) from places
    near on or near a drum to model drum transitions. To simulate moving
    between drums, we will also train the model to generate trajectories from
    the surfaces of a drum to a different one with a starting tip velocity that
    is facing away from the drum face (gaussian noised velocity vector to not
    overfit). Use an inertia-Weighted Pseudoinverse to convert that tip velocity
    in joint velocities.

So a sample is one of two kinds:

* **drum-departure** (default 90 %): each hand is placed on a drum it is
  allowed to play, with a tip velocity leaving that head. The pose comes from
  the vendored :class:`~amortized_opt.warmstart.GoalPoseLibrary` -- the same
  per-(hand, drum, waist-yaw) IK solutions the warm-start terminal uses, which
  is precisely "the robot has just struck this drum" -- jittered in joint space.
  The departure direction is the head normal, rotated by a Gaussian angle so
  the model never sees a perfectly axial rebound.
* **neutral** (the rest): the drummer ready pose plus joint-space Gaussian
  noise and a small random velocity.
* **cold** (off by default, see :attr:`StartConfig.cold_fraction`): a neutral
  pose whose waist yaw is redrawn UNIFORMLY over the aim law's policy range
  instead of from the library's yaw grid. The point is not the pose but the
  *independence*: every other family puts the torso somewhere the goals would
  have put it, which leaves ``yaw_now`` and the analytic aim collinear on the
  training distribution and the waist set-point head unable to tell which of
  the two it is being paid to follow (2026-08-14 waist-yaw runaway). Only the
  neutral family can carry it -- a drum-departure pose is the library's IK
  solution AT its yaw, so overriding the yaw underneath it would describe a
  robot that is not the one the arms were solved for.

Both hands are placed independently (the plan's "training will include training
data where both arms are hitting different things"), sharing one waist yaw
drawn from the library's yaw grid so the dual-arm pose stays self-consistent.

Velocity conversion is a SINGLE dynamically-consistent solve over both hands at
once: the two 3-row tip Jacobians are stacked into one 6-row task on all 17
DoF and inverted with :func:`~amortized_opt.ik.inertia_weighted_pinv`. Solving
the hands separately would make them fight over the shared waist columns;
solving them jointly lets the metric decide how to split the waist, which is
the same coupling the model has to learn.

Rejection: a sample is discarded if it is in collision (the vendored SDF's
``min_sdf``) or outside the joint box. Position samples are clipped into the
box first, so limit violations are impossible by construction and the
rejection loop only ever fires on collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
from .fk import ChainFK
from .ik import inertia_weighted_pinv, tip_jacobian
from .limits import ETA_VELOCITY, isaac_velocity_limits, urdf_position_limits
from .pd_sim import mass_matrix
from .rhythm import HANDS, DrumKit
from .sdf import DrumSceneSpec, SdfPenalty
from .waist_aim import _DEFAULT_YAW_LIMIT
from .warmstart import GoalPoseLibrary, arm_slice

__all__ = ["StartConfig", "StartBatch", "StartSampler"]

#: Sample "kind" codes stored on :class:`StartBatch`.
KIND_DRUM = 0
KIND_NEUTRAL = 1
KIND_COLD = 2


@dataclass
class StartConfig:
    """Distribution knobs. Every default is deliberately conservative."""

    #: Fraction of samples placed on/near a drum (plan: ~90 %).
    drum_fraction: float = 0.9
    #: Joint-space Gaussian jitter (rad, sd) added to a library pose.
    drum_q_jitter: float = 0.05
    #: ...and to the neutral ready pose.
    neutral_q_jitter: float = 0.18
    #: Waist jitter (rad, sd) on top of the drawn grid yaw.
    waist_jitter: float = 0.05
    #: Fraction of samples drawn COLD -- neutral pose, waist yaw uniform over
    #: +/- ``cold_yaw_limit`` and independent of everything else. 0.0 (the
    #: default) reproduces the two-family sampler exactly, including its RNG
    #: stream; the phase-2 generator turns it on per call.
    cold_fraction: float = 0.0
    #: Half-range of the cold yaw draw (rad), clipped to the URDF box. The aim
    #: law's own policy limit (``waist_aim._DEFAULT_YAW_LIMIT``, 100 deg): a
    #: torso parked outside the range the aimer is ever allowed to command is a
    #: state the closed loop cannot reach, so it would only spend samples.
    cold_yaw_limit: float = _DEFAULT_YAW_LIMIT
    #: Departure tip-speed range (m/s) -- a rebound, not a strike.
    departure_speed: tuple[float, float] = (0.3, 2.0)
    #: Gaussian sd of the departure-direction cone half-angle (rad).
    departure_angle_sigma: float = 0.35  # ~20 deg
    #: Neutral-start joint velocity, as a fraction of the per-joint limit.
    neutral_v_scale: float = 0.10
    #: Joint velocities are clipped to this fraction of the rating.
    eta_velocity: float = ETA_VELOCITY
    #: A sample is rejected when the SDF's signed distance drops below this.
    min_clearance: float = 0.0
    #: Rejection budget per sample before the neutral fallback is used.
    max_tries: int = 24
    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)

    def __post_init__(self) -> None:
        if not 0.0 <= self.drum_fraction <= 1.0:
            raise ValueError("drum_fraction must be in [0, 1]")


@dataclass
class StartBatch:
    """A batch of sampled start states.

    ``drum`` is ``(B, 2)`` with the drum index each hand departs from, or -1
    for a hand that is not on a drum (both hands of a neutral sample, and a
    hand whose library entry was missing).
    """

    q: torch.Tensor  # (B, 17)
    qd: torch.Tensor  # (B, 17)
    drum: torch.Tensor  # (B, 2) long, -1 = not on a drum
    kind: torch.Tensor  # (B,) long, KIND_*
    waist_yaw: torch.Tensor  # (B,)
    min_sdf: torch.Tensor  # (B,) clearance of the accepted sample
    tries: torch.Tensor  # (B,) how many draws each sample needed

    def __len__(self) -> int:
        return int(self.q.shape[0])

    @property
    def drum_proximal(self) -> torch.Tensor:
        return self.kind == KIND_DRUM

    def to(self, device) -> "StartBatch":
        return StartBatch(**{k: getattr(self, k).to(device) for k in self.__dataclass_fields__})


class StartSampler:
    """Seeded, reproducible sampler of dual-arm start states.

    ::

        s = StartSampler(device="cuda:1")
        batch = s.sample(1024, seed=0)        # identical for the same seed

    Everything stochastic goes through one ``numpy.random.Generator``, so a
    seed pins the whole batch including the accept/reject path.
    """

    def __init__(
        self,
        *,
        kit: DrumKit | None = None,
        library: GoalPoseLibrary | None = None,
        fk: ChainFK | None = None,
        rnea=None,
        sdf: SdfPenalty | None = None,
        config: StartConfig | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.cfg = config or StartConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.names = tuple(self.cfg.joint_names)
        self.n = len(self.names)
        self.kit = kit or DrumKit.from_layout()
        self.library = library or GoalPoseLibrary(joint_names=self.names)
        self.fk = fk or ChainFK(device=self.device, dtype=dtype)
        if rnea is None:
            from .rnea import BatchedRNEA

            rnea = BatchedRNEA.from_urdf(device=self.device, dtype=dtype)
        self.rnea = rnea
        self.sdf = sdf or SdfPenalty(
            DrumSceneSpec.from_layout(), fk=self.fk, device=self.device, dtype=dtype
        )
        lo, hi = urdf_position_limits(self.names)
        self.q_lower = np.asarray(lo, float)
        self.q_upper = np.asarray(hi, float)
        self.v_limit = self.cfg.eta_velocity * isaac_velocity_limits(self.names)
        self.ready = ready_joint_pos(self.names)
        self._arm_idx = {h: arm_slice(h, self.names) for h in HANDS}
        self._yaw_idx = self.names.index("waist_yaw_joint")
        self._yaw_grid = (
            np.asarray(self.library.yaw_grid_rad, float)
            if self.library.yaw_grid_rad
            else np.array([0.0])
        )
        # Which (hand, drum-index) pairs actually have a library pose AND are
        # eligible under the no-crossover rule.
        self._hand_drums: dict[str, list[int]] = {}
        for hand in HANDS:
            ok = [
                d
                for d in self.kit.eligible(hand)
                if self.library.has(hand, self.kit.names[d])
            ]
            if not ok:
                raise RuntimeError(f"no library poses for hand {hand!r}")
            self._hand_drums[hand] = ok

    # ------------------------------------------------------------ helpers --
    def eligible_drums(self, hand: str) -> list[int]:
        """Drum indices this hand may depart from (eligibility AND library)."""
        return list(self._hand_drums[hand])

    def _t(self, a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, float), device=self.device, dtype=self.dtype)

    def _perturb_axis(self, axis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Unit vector within a Gaussian cone about ``axis``."""
        ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(ref, axis)
        t1 /= max(float(np.linalg.norm(t1)), 1e-12)
        t2 = np.cross(axis, t1)
        theta = abs(float(rng.normal(0.0, self.cfg.departure_angle_sigma)))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        d = np.cos(theta) * axis + np.sin(theta) * (np.cos(phi) * t1 + np.sin(phi) * t2)
        return d / max(float(np.linalg.norm(d)), 1e-12)

    # ---------------------------------------------------------- proposals --
    def _propose(
        self, rng: np.random.Generator, kind: int, kit: DrumKit | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """One candidate ``(q, v_tip (2,3), drum (2,), kind)`` of the given kind.

        ``v_tip`` rows are the requested tip velocities (zero for a hand that
        is not departing a drum). The kind is decided ONCE per batch slot, not
        per retry, so a rejected drum start is re-drawn as another drum start
        and the requested ``drum_fraction`` survives rejection sampling.
        """
        cfg = self.cfg
        kit = kit if kit is not None else self.kit
        v_tip = np.zeros((2, 3))
        drums = np.full(2, -1, dtype=np.int64)
        if kind == KIND_DRUM:
            yaw = float(rng.choice(self._yaw_grid)) + float(rng.normal(0.0, cfg.waist_jitter))
            q = self.ready.copy()
            q[self._yaw_idx] = yaw
            for hi, hand in enumerate(HANDS):
                d = int(rng.choice(self._hand_drums[hand]))
                entry = self.library.select(hand, kit.names[d], yaw)
                q[self._arm_idx[hand]] = entry.arm_q
                drums[hi] = d
                speed = float(rng.uniform(*cfg.departure_speed))
                v_tip[hi] = speed * self._perturb_axis(kit.axes[d], rng)
            q = q + rng.normal(0.0, cfg.drum_q_jitter, size=self.n)
            q[self._yaw_idx] = yaw  # keep the waist on the pose the arms were solved at
            return np.clip(q, self.q_lower, self.q_upper), v_tip, drums, KIND_DRUM

        q = self.ready + rng.normal(0.0, cfg.neutral_q_jitter, size=self.n)
        if kind == KIND_COLD:
            # The yaw is drawn LAST and overwrites the jitter rather than adding
            # to it: the whole point of a cold start is that this number carries
            # no information about where the goals are, and a jitter around the
            # ready pose's yaw would leak the ready pose back in.
            lim = min(
                float(cfg.cold_yaw_limit),
                float(min(-self.q_lower[self._yaw_idx], self.q_upper[self._yaw_idx])),
            )
            q[self._yaw_idx] = float(rng.uniform(-lim, lim))
            return np.clip(q, self.q_lower, self.q_upper), v_tip, drums, KIND_COLD
        return np.clip(q, self.q_lower, self.q_upper), v_tip, drums, KIND_NEUTRAL

    # -------------------------------------------------------- velocity IK --
    def departure_velocity(
        self, q: torch.Tensor, v_tip: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        """(B, 17) joint velocities for a stacked two-hand tip-velocity task.

        ``v_tip`` is ``(B, 2, 3)`` and ``active`` ``(B, 2)`` bool. Inactive
        hands contribute a zero task row rather than being dropped, which keeps
        the shapes static; the metric then simply holds that tip still.
        """
        J = torch.cat([tip_jacobian(self.fk, q, h) for h in HANDS], dim=1)  # (B, 6, n)
        M = mass_matrix(self.rnea, q)
        P = inertia_weighted_pinv(J, M=M, damping=1e-4)
        task = (v_tip * active.unsqueeze(-1).to(v_tip.dtype)).reshape(q.shape[0], 6)
        qd = torch.matmul(P, task.unsqueeze(-1)).squeeze(-1)
        lim = self._t(self.v_limit).view(1, -1)
        return torch.clamp(qd, -lim, lim)

    # ------------------------------------------------------------- sample --
    def sample(
        self,
        batch: int,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        kit: DrumKit | None = None,
        cold_fraction: float | None = None,
    ) -> StartBatch:
        """``batch`` accepted start states. ``seed`` pins the batch exactly.

        ``kit`` overrides the nominal layout for THIS call -- pass the
        domain-randomized kit the rest of the sample will be built against
        (fix B).  Both the departure normals and the collision screen then use
        the same geometry the cost and the goal tokens will use; with the
        nominal kit the screen accepts poses that are inside a shifted drum and
        rejects poses that are not.

        ``cold_fraction`` overrides :attr:`StartConfig.cold_fraction` for THIS
        call, which is how a caller that decides cold-ness per scenario (the
        phase-2 generator) draws an all-cold chunk from the same sampler.  Cold
        slots that exhaust their rejection budget fall back to the ready pose
        and are reported as ``KIND_NEUTRAL``, not ``KIND_COLD``: the fallback
        sits at yaw 0, which is exactly the correlation a cold start exists to
        break, so it must not be mislabelled as one.
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        cfg = self.cfg
        B = int(batch)
        kit = kit if kit is not None else self.kit
        heads_env = (
            None if kit is self.kit
            else self._t(kit.positions_in("env")).unsqueeze(0)   # (1, D, 3)
        )
        # RADII travel with the centres: ``KitRandomization.per_drum_radius``
        # resizes the collision cylinder, so screening a shifted kit against
        # nominal radii reintroduces exactly the disagreement ``heads_env``
        # removes, one drum dimension over.
        radii_env = (
            None if heads_env is None
            else self._t(kit.radii).unsqueeze(0)                  # (1, D)
        )

        q_out = np.empty((B, self.n))
        vt_out = np.zeros((B, 2, 3))
        drum_out = np.full((B, 2), -1, dtype=np.int64)
        kind_out = np.empty(B, dtype=np.int64)
        tries_out = np.ones(B, dtype=np.int64)
        sdf_out = np.zeros(B)

        # The kind is fixed per slot up front: rejection sampling inside a slot
        # must not be able to shift the drum/neutral mix away from the request.
        kinds = np.where(
            rng.uniform(size=B) < cfg.drum_fraction, KIND_DRUM, KIND_NEUTRAL
        ).astype(np.int64)
        cold_f = float(cfg.cold_fraction if cold_fraction is None else cold_fraction)
        if cold_f > 0.0:
            # Drawn only when asked for, so the default sampler's RNG stream --
            # and therefore every pool generated before cold starts existed --
            # is unchanged bit for bit.
            kinds = np.where(rng.uniform(size=B) < cold_f, KIND_COLD, kinds)

        pending = list(range(B))
        # Candidates are proposed and screened in waves so the collision check
        # is one batched SDF call rather than B of them.
        attempt = 0
        while pending and attempt < cfg.max_tries:
            attempt += 1
            cand_q = np.empty((len(pending), self.n))
            cand_v = np.zeros((len(pending), 2, 3))
            cand_d = np.full((len(pending), 2), -1, dtype=np.int64)
            cand_k = np.empty(len(pending), dtype=np.int64)
            for j, b in enumerate(pending):
                cand_q[j], cand_v[j], cand_d[j], cand_k[j] = self._propose(
                    rng, int(kinds[b]), kit
                )
            qt = self._t(cand_q)
            dt_l = torch.as_tensor(cand_d[:, 0], device=self.device)
            dt_r = torch.as_tensor(cand_d[:, 1], device=self.device)
            with torch.no_grad():
                # A hand resting on the drum it has just struck is not "in
                # collision" with that drum -- exempt the stick capsule, exactly
                # as the trajectory cost does over the last waypoints of a strike.
                _, det = self.sdf.forward(
                    qt, target_drum_left=dt_l, target_drum_right=dt_r,
                    drum_head_centers=(
                        None if heads_env is None
                        else heads_env.expand(qt.shape[0], -1, -1)
                    ),
                    drum_radii=(
                        None if radii_env is None
                        else radii_env.expand(qt.shape[0], -1)
                    ),
                    return_details=True,
                )
            clear = det.min_sdf.detach().cpu().numpy()
            ok = clear >= cfg.min_clearance
            still: list[int] = []
            for j, b in enumerate(pending):
                if ok[j]:
                    q_out[b] = cand_q[j]
                    vt_out[b] = cand_v[j]
                    drum_out[b] = cand_d[j]
                    kind_out[b] = cand_k[j]
                    sdf_out[b] = clear[j]
                else:
                    tries_out[b] += 1
                    still.append(b)
            pending = still

        if pending:
            # Fallback: the ready pose is collision-free by construction.
            qt = self._t(self.ready[None, :])
            with torch.no_grad():
                _, det = self.sdf.forward(qt, return_details=True)
            base_clear = float(det.min_sdf[0])
            for b in pending:
                q_out[b] = self.ready
                vt_out[b] = 0.0
                drum_out[b] = -1
                kind_out[b] = KIND_NEUTRAL
                sdf_out[b] = base_clear

        q = self._t(q_out)
        v_tip = self._t(vt_out)
        drum = torch.as_tensor(drum_out, device=self.device)
        active = drum >= 0
        qd = self.departure_velocity(q, v_tip, active)

        # Starts that are not departing a drum get a small random joint velocity
        # instead -- neutral and cold alike, since neither has a tip task.
        neutral = torch.as_tensor(kind_out != KIND_DRUM, device=self.device)
        if bool(neutral.any()):
            noise = self._t(
                cfg.neutral_v_scale * self.v_limit[None, :] * rng.normal(size=(B, self.n))
            )
            qd = torch.where(neutral.unsqueeze(-1), noise, qd)
            lim = self._t(self.v_limit).view(1, -1)
            qd = torch.clamp(qd, -lim, lim)

        return StartBatch(
            q=q,
            qd=qd,
            drum=drum,
            kind=torch.as_tensor(kind_out, device=self.device),
            waist_yaw=q[:, self._yaw_idx].clone(),
            min_sdf=self._t(sdf_out),
            tries=torch.as_tensor(tries_out, device=self.device),
        )

    # ------------------------------------------------------------ checking --
    def violations(self, batch: StartBatch) -> dict[str, np.ndarray]:
        """Per-sample diagnostics: joint-box excess, speed excess, clearance."""
        q = batch.q.detach().cpu().numpy()
        qd = batch.qd.detach().cpu().numpy()
        below = np.maximum(self.q_lower[None, :] - q, 0.0).max(axis=1)
        above = np.maximum(q - self.q_upper[None, :], 0.0).max(axis=1)
        speed = np.maximum(np.abs(qd) - self.v_limit[None, :], 0.0).max(axis=1)
        return {
            "joint_box": np.maximum(below, above),
            "velocity": speed,
            "min_sdf": batch.min_sdf.detach().cpu().numpy(),
        }
