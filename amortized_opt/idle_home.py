"""The **home pose** an arm with no goal parks at (the idle arm).

Section "the idle arm" of ``amortized/README.md``.

An arm with no goal in the context lookahead has no terminal objective -- only
collision, limits, smoothness and the mid-range posture well price it -- so its
behaviour is underdetermined, and the known failure is the waist (serving the
active hand) sweeping the idle arm through the kit.  This module builds the
retraction target that closes that hole:

    ``q_home[hand]`` -- one 7-DoF arm posture per hand.

**It is torso-relative, and that is the whole point.**  The arm is mounted on
the torso, so a fixed vector of arm joint angles rotates rigidly with the torso
for free: park there and the arm keeps its posture relative to the body no
matter where the waist yaws.  The pose therefore does NOT depend on waist yaw,
and an idle arm asked to hold it genuinely sits still rather than tracking a
moving target while the torso turns.

What waist yaw *is* used for is validation, not indexing: the kit is world-fixed
while the arm swings with the torso, so a parked posture traces a circle through
the world and the solve has to guarantee that circle clears every drum.  The
clearance term is therefore evaluated at every yaw in a sweep and the pose must
satisfy all of them at once.  (An earlier version indexed the pose BY yaw bin.
It clears more tightly -- there is more freedom per bin -- but it asks a resting
arm to keep moving as the torso turns, it made the cost's target discontinuous
in yaw, and none of that bought anything the single pose cannot deliver.)

The pose is chosen to be:

1. **clear of the kit at every waist yaw** -- ``hinge(margin - sdf)`` against the
   vendored :class:`SdfPenalty`, swept over the yaw grid, at a ``margin`` that is
   the *safety zone* the parked stick keeps;
2. **ready** -- a quadratic anchor toward that hand's own
   :class:`~amortized_opt.warmstart.GoalPoseLibrary` strike poses, so home sits
   in the middle of the postures the arm may need next rather than in an
   arbitrary corner of the clear set;
3. **holdable** -- projected onto the narrowed posture box every step and
   carrying the same elbow-down hinge the cost does.

The result is cached to ``assets/idle_home_poses.json`` by
``scripts/build_idle_home.py`` and consumed in three places that agree by
construction: the cost (:meth:`~amortized_opt.cost.FusedCost.set_idle_home`),
the seed (:func:`~amortized_opt.warmstart.warmstart_trajectory` and
:meth:`~amortized_opt.train_cost.MPCRoller._seed_idle`) and the deployed loop
(:class:`~amortized_opt.mpc_gen.MPCGenerator`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .constants import G1_UPPER_BODY_JOINTS, default_layout_path, ready_joint_pos
from .fk import ChainFK, stick_tip_offset
from .limits import urdf_position_limits
from .sdf import DrumSceneSpec, SdfPenalty
from .warmstart import GoalPoseLibrary, arm_slice

__all__ = [
    "IdleHomeConfig",
    "IdleHomeTable",
    "build_idle_home_table",
    "default_idle_home_path",
    "load_idle_home_table",
    "resolve_idle_home",
]

_HANDS = ("left", "right")
_FRAMES: dict[str, str] = {
    "left_wrist": "left_wrist_yaw_joint",
    "right_wrist": "right_wrist_yaw_joint",
    "left_elbow": "left_elbow_joint",
    "right_elbow": "right_elbow_joint",
    "left_shoulder": "left_shoulder_pitch_joint",
    "right_shoulder": "right_shoulder_pitch_joint",
    "torso": "waist_pitch_joint",
}


def default_idle_home_path() -> Path:
    """``assets/idle_home_poses.json`` next to the drum layout."""
    return Path(default_layout_path()).with_name("idle_home_poses.json")


# ================================================================= config ====
@dataclass
class IdleHomeConfig:
    """Knobs of the offline home-pose solve.

    Nothing at *training* time reads this; only the solved pose.
    """

    #: Waist yaws the clearance constraint is ENFORCED at.  Not an index -- the
    #: pose is one vector -- but the sweep it has to survive.  129 over the full
    #: joint box is a 2.3 deg check; the swept tip circle is smooth, so this is
    #: far finer than it needs to be and costs nothing (one batched FK pass).
    n_yaw_checks: int = 129
    #: The SAFETY ZONE, metres of clearance the parked stick keeps from every
    #: drum, under-drum volume and the torso, at every yaw.  0.06 m is 12x the
    #: collision margin the cost checks (0.005 m) and leaves room for the ~2 cm
    #: of tracking error the PD plant runs at.
    margin: float = 0.06
    #: Price on the clearance hinge.  Large: this is the constraint, the anchor
    #: is only a preference inside the set it leaves.
    w_clear: float = 1.0e3
    #: Hinge exponent.  **1**, not the SDF module's default 2: a squared hinge's
    #: gradient vanishes as the violation shrinks, and the measured failure was
    #: exactly that -- the solve parked 9 mm INSIDE the torso capsule and stopped.
    #: A linear hinge pushes with constant force until the margin is met.
    power: int = 1
    #: Price on ``mean_(yaw, drum) |q_arm - q_lib(hand, drum, yaw)|^2``, the
    #: readiness anchor.  It is what keeps home near the postures the arm has to
    #: reach next instead of tucked wherever clearance is easiest.
    w_reach: float = 1.0
    #: Elbow-down hinge, same geometry as ``FusedCost._posture``.
    w_elbow: float = 10.0
    elbow_margin: float = 0.05
    #: Keep the pose off the box edge by this fraction of each joint's
    #: half-range: a home pose ON a limit has no room to be tracked into.
    box_shrink: float = 0.97
    iters: int = 800
    lr: float = 2.0e-2
    #: Independent starts, the best kept per hand.  The landscape has a
    #: torso-tucked local minimum the strike-pose centroid can fall into.
    restarts: int = 4
    restart_jitter: float = 0.35
    seed: int = 0
    #: Posture (narrowed) limits, matching ``CostConfig.box_source="posture"``.
    posture_limits: bool = True
    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)

    def validate(self) -> "IdleHomeConfig":
        if self.n_yaw_checks < 1:
            raise ValueError("n_yaw_checks must be >= 1")
        if self.margin <= 0.0:
            raise ValueError("margin must be > 0")
        if not 0.0 < self.box_shrink <= 1.0:
            raise ValueError("box_shrink must be in (0, 1]")
        return self


# ================================================================== table ====
@dataclass
class IdleHomeTable:
    """One park posture per hand, plus the sweep that validated it.

    Index 0 is the LEFT hand, 1 the RIGHT -- the same convention as
    :class:`~amortized_opt.cost.StrikeContext`.
    """

    q_arm: np.ndarray          # (2, n_arm) THE home pose, torso-relative
    joint_names: tuple[str, ...]
    #: ``(n_yaw,)`` waist yaws the clearance below was measured at.
    yaw_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: ``(2, n_yaw)`` clearance of the parked pose at each of those yaws.  The
    #: MINIMUM over this is the number that matters: it is the worst case over
    #: everywhere the torso can point.
    min_sdf: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))
    #: ``(2,)`` RMS joint distance to that hand's strike poses.
    reach_err: np.ndarray = field(default_factory=lambda: np.zeros(2))
    margin: float = 0.0
    #: ``(2, 3)`` stick-TIP position of the parked pose in the TORSO frame --
    #: fixed, because the pose is torso-relative.  Carried so the idle context
    #: token has a spatial payload without an FK pass in the data path.
    tip_torso: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))

    @property
    def n_arm(self) -> int:
        return int(self.q_arm.shape[-1])

    @property
    def worst_clearance(self) -> float:
        """Minimum clearance over both hands and every validated yaw, metres."""
        return float(np.min(self.min_sdf)) if self.min_sdf.size else float("nan")

    def arm_pose(self, hand: str) -> np.ndarray:
        """``(n_arm,)`` home pose of ``hand``.  No yaw argument: it is the same
        posture at every waist yaw, which is what "torso-relative" means."""
        return np.asarray(self.q_arm[_HANDS.index(hand)], float)

    def tip_pos(self, hand: str) -> np.ndarray:
        """``(3,)`` torso-frame stick-tip position of ``hand``'s home pose."""
        return np.asarray(self.tip_torso[_HANDS.index(hand)], float)

    def full_pose(self, yaw: float = 0.0, base: np.ndarray | None = None) -> np.ndarray:
        """``(n_dof,)`` pose with BOTH arms parked and the waist at ``yaw``."""
        names = tuple(self.joint_names)
        q = ready_joint_pos(names) if base is None else np.asarray(base, float).copy()
        q[names.index("waist_yaw_joint")] = float(yaw)
        for hand in _HANDS:
            q[arm_slice(hand, names)] = self.arm_pose(hand)
        return q

    # ------------------------------------------------------------- codecs ---
    def to_dict(self) -> dict:
        return {
            "joint_names": list(self.joint_names),
            "q_arm": np.asarray(self.q_arm, float).tolist(),
            "yaw_grid": [float(v) for v in self.yaw_grid],
            "margin": float(self.margin),
            "min_sdf": np.asarray(self.min_sdf, float).tolist(),
            "reach_err": np.asarray(self.reach_err, float).tolist(),
            "tip_torso": np.asarray(self.tip_torso, float).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IdleHomeTable":
        q = np.asarray(d["q_arm"], float)
        if q.ndim != 2:
            raise ValueError(
                "this idle-home file is the old yaw-INDEXED format "
                f"(q_arm {q.shape}); rebuild with scripts/build_idle_home.py"
            )
        return cls(
            q_arm=q,
            joint_names=tuple(d["joint_names"]),
            yaw_grid=np.asarray(d.get("yaw_grid", []), float),
            min_sdf=np.asarray(d.get("min_sdf", []), float),
            reach_err=np.asarray(d.get("reach_err", []), float),
            margin=float(d.get("margin", 0.0)),
            tip_torso=np.asarray(d.get("tip_torso", np.zeros((2, 3))), float),
        )

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path is not None else default_idle_home_path()
        p.write_text(json.dumps(self.to_dict()))
        return p

    def torch(self, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``(2, n_arm)`` home poses as a tensor on ``device``."""
        return torch.as_tensor(self.q_arm, device=device, dtype=dtype)

    def tip_tensor(self, device=None, dtype=None) -> torch.Tensor:
        """``(2, 3)`` torso-frame tip positions as a tensor.

        ``dtype`` resolves in the BODY: the ``torch`` method above shadows the
        module inside the class namespace, where defaults are evaluated.
        """
        return torch.as_tensor(
            self.tip_torso,
            device=device,
            dtype=dtype if dtype is not None else torch.float32,
        )


def load_idle_home_table(
    path: str | Path | None = None, *, joint_names=G1_UPPER_BODY_JOINTS
) -> IdleHomeTable:
    """Read the cached pose, checking it was built for ``joint_names``."""
    p = Path(path) if path is not None else default_idle_home_path()
    if not p.exists():
        raise FileNotFoundError(
            f"idle-home pose missing at {p} -- build it with "
            "`python scripts/build_idle_home.py`"
        )
    tab = IdleHomeTable.from_dict(json.loads(p.read_text()))
    if tuple(joint_names) != tuple(tab.joint_names):
        raise ValueError(
            f"idle-home pose was built for {tab.joint_names}, not {tuple(joint_names)}"
        )
    return tab


def resolve_idle_home(spec, *, joint_names=G1_UPPER_BODY_JOINTS) -> IdleHomeTable | None:
    """``None`` / ``"default"`` / a path -> an :class:`IdleHomeTable` or ``None``.

    ``"default"`` is *permissive*: it returns ``None`` (and says so) when the
    asset has not been built, so a checkout without it behaves exactly as before
    rather than dying on a missing file.  An explicit path is *strict* -- if you
    named a pose file, a typo must not silently disable the feature.
    """
    if spec is None:
        return None
    if str(spec) == "default":
        p = default_idle_home_path()
        if not p.exists():
            print(
                f"[idle-home] {p} not built; idle-arm terms stay off "
                "(run scripts/build_idle_home.py)"
            )
            return None
        return load_idle_home_table(p, joint_names=joint_names)
    return load_idle_home_table(spec, joint_names=joint_names)


# ================================================================= solver ====
def _library_anchor(
    library: GoalPoseLibrary, yaws: np.ndarray, joint_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """``(2, n_yaw, M, n_arm)`` strike poses + a ``(2, M)`` validity mask.

    The anchor set is the hand's OWN library entries -- ``left/floor_tom`` and
    ``right/hihat`` do not exist, so a hand is never asked to stay ready for a
    drum the no-crossover rule forbids it.  Entries are read at every yaw
    because the library's own pick varies with yaw; home is one pose, so it is
    anchored to the whole set at once.
    """
    per_hand = {h: library.drums(h) for h in _HANDS}
    M = max(len(v) for v in per_hand.values())
    n_arm = len(arm_slice("left", joint_names))
    anchors = np.zeros((2, len(yaws), M, n_arm))
    valid = np.zeros((2, M))
    for hi, hand in enumerate(_HANDS):
        for di, drum in enumerate(per_hand[hand]):
            valid[hi, di] = 1.0
            for yi, y in enumerate(yaws):
                anchors[hi, yi, di] = library.select(hand, drum, float(y)).arm_q
    return anchors, valid


def _elbow_drop(frames: dict, hand: str) -> torch.Tensor:
    """``z_mid(shoulder, wrist) - z_elbow``; > 0 means the elbow hangs below."""
    ze = frames[f"{hand}_elbow"][1][..., 2]
    zw = frames[f"{hand}_wrist"][1][..., 2]
    zs = frames[f"{hand}_shoulder"][1][..., 2]
    return 0.5 * (zs + zw) - ze


def build_idle_home_table(
    cfg: IdleHomeConfig | None = None,
    *,
    library: GoalPoseLibrary | None = None,
    scene: DrumSceneSpec | None = None,
    fk: ChainFK | None = None,
    urdf_path: str | Path | None = None,
    layout_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> IdleHomeTable:
    """Solve ONE park posture per hand, clear at every waist yaw in the sweep.

    The optimization variable is a single ``(2, n_arm)`` block.  It is broadcast
    across the ``n_yaw_checks`` yaw rows before FK, so every yaw contributes its
    own clearance gradient to the same parameter -- which is exactly "this one
    posture must clear the kit no matter where the torso points".

    ``dtype`` defaults to float64: this runs once, offline, and the clearance
    hinge is a difference of distances where float32 costs real accuracy.
    """
    cfg = (cfg or IdleHomeConfig()).validate()
    names = tuple(cfg.joint_names)
    dev = torch.device(device)
    library = library or GoalPoseLibrary(joint_names=names)
    fk = fk or ChainFK(
        urdf_path, joint_names=names, frames=dict(_FRAMES), device=dev, dtype=dtype
    )
    scene = scene or DrumSceneSpec.from_layout(layout_path, frame="pelvis")
    sdf = SdfPenalty(
        scene, fk=fk, margin=float(cfg.margin), power=int(cfg.power),
        device=dev, dtype=dtype,
    )

    lo_np, hi_np = urdf_position_limits(names, posture_limits=cfg.posture_limits)
    mid = 0.5 * (lo_np + hi_np)
    half = 0.5 * (hi_np - lo_np) * float(cfg.box_shrink)
    lo_np, hi_np = mid - half, mid + half
    yaw_i = names.index("waist_yaw_joint")
    N = int(cfg.n_yaw_checks)
    yaws = (
        np.linspace(float(lo_np[yaw_i]), float(hi_np[yaw_i]), N) if N > 1
        else np.zeros(1)
    )

    anchors_np, valid_np = _library_anchor(library, yaws, names)
    t = lambda a: torch.as_tensor(np.asarray(a, float), device=dev, dtype=dtype)
    anchors = t(anchors_np)          # (2, N, M, n_arm)
    a_valid = t(valid_np)            # (2, M)
    lo, hi = t(lo_np), t(hi_np)

    cols = {h: torch.as_tensor(list(arm_slice(hand, names)), device=dev)
            for h, hand in enumerate(_HANDS)}
    n_arm = int(cols[0].numel())
    lo_a = torch.stack([lo[cols[0]], lo[cols[1]]])   # (2, n_arm)
    hi_a = torch.stack([hi[cols[0]], hi[cols[1]]])

    # The yaw sweep: one row per checked yaw, waist lean at zero, arms filled in.
    q_fixed = t(np.repeat(ready_joint_pos(names)[None, :], N, axis=0))
    q_fixed[:, yaw_i] = t(yaws)
    for nm in ("waist_pitch_joint", "waist_roll_joint"):
        q_fixed[:, names.index(nm)] = 0.0

    def assemble(param: torch.Tensor) -> torch.Tensor:
        """``(2, n_arm)`` -> ``(N, n_dof)``: the SAME arms at every checked yaw."""
        q = q_fixed.clone()
        rep = param.unsqueeze(0).expand(N, 2, n_arm)
        for h in (0, 1):
            q = q.index_copy(1, cols[h], rep[:, h])
        return q

    def reach_of(param: torch.Tensor) -> torch.Tensor:
        """``(2,)`` mean squared joint distance to the strike poses, over yaw."""
        d = param.view(2, 1, 1, n_arm) - anchors            # (2, N, M, n_arm)
        num = ((d * d).sum(-1) * a_valid.view(2, 1, -1)).sum(2).mean(1)
        return num / a_valid.sum(1)

    def objective(param: torch.Tensor) -> torch.Tensor:
        frames = fk.forward(assemble(param))
        pen = sdf.forward(frames=frames)                    # (N,) one per yaw
        elbow = 0.0
        for hand in _HANDS:
            drop = _elbow_drop(frames, hand)
            elbow = elbow + (-(drop + float(cfg.elbow_margin))).clamp_min(0.0) ** 2
        return (
            float(cfg.w_clear) * pen + float(cfg.w_elbow) * elbow
        ).sum() + float(cfg.w_reach) * reach_of(param).sum()

    def measure(param: torch.Tensor):
        """``(min_sdf (2, N), reach (2,))`` of a solved pose."""
        with torch.no_grad():
            _pen, det = sdf.forward(
                frames=fk.forward(assemble(param)), return_details=True
            )
            per_hand = torch.stack([det.extras[f"min_sdf_{h}"] for h in _HANDS])
        return per_hand, reach_of(param).detach()

    # ---- restarts.  Seed 0 is the strike-pose centroid (already
    # collision-screened and elbow-down), 1 the ready pose, the rest jittered.
    centroid = (
        (anchors * a_valid.view(2, 1, -1, 1)).sum(2) / a_valid.sum(1).view(2, 1, 1)
    ).mean(1)                                               # (2, n_arm)
    ready = torch.stack([t(ready_joint_pos(names))[cols[h]] for h in (0, 1)])
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    def start(r: int) -> torch.Tensor:
        if r == 0:
            return centroid.clone()
        if r == 1:
            return ready.clone()
        noise = torch.randn(centroid.shape, generator=gen, dtype=torch.float64)
        return centroid + float(cfg.restart_jitter) * noise.to(dev, dtype)

    best_q = best_score = best_sdf = best_reach = None
    for r in range(max(int(cfg.restarts), 1)):
        p = start(r).clamp(lo_a, hi_a).requires_grad_(True)
        opt = torch.optim.Adam([p], lr=float(cfg.lr))
        for it in range(int(cfg.iters)):
            opt.zero_grad(set_to_none=True)
            loss = objective(p)
            loss.backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(lo_a, hi_a)                        # projection
            if verbose and (it % max(int(cfg.iters) // 5, 1) == 0):
                print(f"  restart {r}  iter {it:4d}  loss {float(loss.detach()):12.4f}")

        # Merge per HAND (the two hands are separable given the fixed waist), on
        # the WORST-case clearance shortfall over the sweep -- the readiness
        # anchor only breaks ties among poses that already clear everywhere.
        m_sdf, m_reach = measure(p.detach())                 # (2, N), (2,)
        score = -(
            float(cfg.w_clear) * (float(cfg.margin) - m_sdf).clamp_min(0.0).amax(dim=1)
            + float(cfg.w_reach) * m_reach
        )                                                   # (2,)
        q_r = p.detach()
        if best_q is None:
            best_q, best_score, best_sdf, best_reach = q_r, score, m_sdf, m_reach
        else:
            take = score > best_score
            best_q = torch.where(take.unsqueeze(-1), q_r, best_q)
            best_sdf = torch.where(take.unsqueeze(-1), m_sdf, best_sdf)
            best_reach = torch.where(take, m_reach, best_reach)
            best_score = torch.maximum(score, best_score)
        if verbose:
            print(f"  restart {r}: worst clearance {float(m_sdf.min()) * 1e3:.1f} mm")

    # ---- the parked stick tip, in the TORSO frame (fixed, like the pose).
    with torch.no_grad():
        q0 = assemble(best_q)[:1]                           # any yaw row will do
        frames = fk.forward(q0)
        Rt, pt = frames["torso"]
        tips = []
        for hand in _HANDS:
            R, pw = frames[f"{hand}_wrist"]
            local = torch.as_tensor(
                stick_tip_offset(hand), device=dev, dtype=dtype
            ).view(1, 3)
            world = ChainFK.points_world(R, pw, local).squeeze(1)    # (1, 3)
            # into the torso frame: R_torso^T (p_tip - p_torso)
            rel = torch.einsum("bij,bj->bi", Rt.transpose(1, 2), world - pt)
            tips.append(rel.squeeze(0))
        tip_torso = torch.stack(tips).cpu().numpy()          # (2, 3)

    return IdleHomeTable(
        q_arm=best_q.cpu().numpy(),
        joint_names=names,
        yaw_grid=yaws,
        min_sdf=best_sdf.cpu().numpy(),
        reach_err=np.sqrt(best_reach.cpu().numpy() / max(n_arm, 1)),
        margin=float(cfg.margin),
        tip_torso=tip_torso,
    )
