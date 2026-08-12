"""Differentiable analytic signed-distance penalty over the drum scene.

Vendored from ``control/safety/sdf_torch.py`` with two changes:

* **Branch-free target-drum exemption.** The original built the exemption
  one-hot with ``if bool(valid.any()): onehot[rows, tgt[rows]] = 1.0`` -- a
  device->host sync inside the hand loop. Replaced by
  ``F.one_hot(tgt.clamp(min=0), D) * (tgt >= 0)``.
* **Externally-computed FK.** ``forward(..., frames=...)`` accepts a frame dict
  from a single shared :class:`~amortized_opt.fk.ChainFK` pass so a fused cost
  head runs FK once for the SDF, the aux outputs and the torque rows. Passing
  ``q`` alone keeps the internal-FK convenience path.

The layout is loaded from ``amortized/assets/drum_kit_layout.yaml`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    HEAD_SPHERE,
    PELVIS_IN_ENV,
    STICK_RADIUS,
    TIP_RADIUS,
    TORSO_CAPSULE,
    WRIST_RADIUS,
    default_layout_path,
)
from .fk import ChainFK, stick_shaft_offsets, stick_tip_offset

_EPS = 1.0e-9


def _unit_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


@dataclass
class DrumSceneSpec:
    """Drum-kit obstacle description.

    ``head_centers`` are the striking surfaces (layout ``position``); the solid
    drum hangs ``height`` below along ``-axis``. Each drum also gets an
    under-drum no-go cylinder of depth ``under_depth`` hung ``under_clearance``
    below the head volume.

    ``frame``: "env" (layout coordinates) or "pelvis" (layout - PELVIS_IN_ENV).
    """

    names: list[str]
    head_centers: np.ndarray  # (D, 3)
    axes: np.ndarray  # (D, 3) unit head normals
    radii: np.ndarray  # (D,)
    heights: np.ndarray  # (D,)
    frame: str = "env"
    under_depth: float = 0.08
    under_clearance: float = 0.01

    @classmethod
    def from_layout(
        cls,
        layout_path: str | Path | None = None,
        *,
        frame: str = "env",
        under_depth: float = 0.08,
        under_clearance: float = 0.01,
    ) -> "DrumSceneSpec":
        import yaml

        path = Path(layout_path) if layout_path is not None else default_layout_path()
        with open(path) as f:
            layout = yaml.safe_load(f)
        # Order by the layout's drum `id` so row i == drum id i (matches the
        # target_drum_* exemption indices).
        items = list(layout["drums"].items())
        if all("id" in spec for _n, spec in items):
            items.sort(key=lambda kv: int(kv[1]["id"]))
        names, centers, axes, radii, heights = [], [], [], [], []
        for name, spec in items:
            names.append(str(name))
            centers.append(np.asarray(spec["position"], dtype=float))
            axes.append(_unit_np(np.asarray(spec.get("normal", [0.0, 0.0, 1.0]), dtype=float)))
            radii.append(float(spec["radius"]))
            heights.append(float(spec["height"]))
        head = np.stack(centers, axis=0)
        if frame == "pelvis":
            head = head - PELVIS_IN_ENV[None, :]
        elif frame != "env":
            raise ValueError(f"frame must be 'env' or 'pelvis', got {frame!r}")
        return cls(
            names=names,
            head_centers=head,
            axes=np.stack(axes, axis=0),
            radii=np.asarray(radii, dtype=float),
            heights=np.asarray(heights, dtype=float),
            frame=frame,
            under_depth=float(under_depth),
            under_clearance=float(under_clearance),
        )

    @property
    def num_drums(self) -> int:
        return len(self.names)


def point_cylinder_sdf(
    x: torch.Tensor,
    center: torch.Tensor,
    axis: torch.Tensor,
    radius: torch.Tensor,
    half_height: torch.Tensor,
) -> torch.Tensor:
    """Signed distance of points to solid capped cylinders (differentiable a.e.)."""
    u = x - center
    z = (u * axis).sum(dim=-1)
    perp = u - z.unsqueeze(-1) * axis
    rho = torch.sqrt((perp * perp).sum(dim=-1) + _EPS**2)
    d_rho = rho - radius
    d_z = z.abs() - half_height
    out = torch.sqrt(d_rho.clamp_min(0.0) ** 2 + d_z.clamp_min(0.0) ** 2 + _EPS**2) - _EPS
    inside = torch.maximum(d_rho, d_z).clamp_max(0.0)
    return out + inside


def point_capsule_sdf(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, radius: float
) -> torch.Tensor:
    ab = b - a
    denom = (ab * ab).sum(dim=-1).clamp_min(_EPS**2)
    t = (((x - a) * ab).sum(dim=-1) / denom).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    d = x - closest
    return torch.sqrt((d * d).sum(dim=-1) + _EPS**2) - radius


def point_sphere_sdf(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    d = x - center
    return torch.sqrt((d * d).sum(dim=-1) + _EPS**2) - radius


@dataclass
class SdfPenaltyDetails:
    """Diagnostics from one :class:`SdfPenalty` call (all detached)."""

    min_sdf: torch.Tensor  # (B,)
    pen_drum: torch.Tensor  # (B,)
    pen_under: torch.Tensor  # (B,)
    pen_self: torch.Tensor  # (B,)
    extras: dict = field(default_factory=dict)


class SdfPenalty:
    """hinge(margin - SDF)^p summed over collision pairs, batched + differentiable."""

    def __init__(
        self,
        scene: DrumSceneSpec | None = None,
        *,
        fk: ChainFK | None = None,
        urdf_path: str | Path | None = None,
        margin: float = 0.01,
        power: int = 2,
        clamp_depth: float | None = None,
        shaft_samples: int = 4,
        include_self: bool = True,
        w_drum: float = 1.0,
        w_under: float = 1.0,
        w_self: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.scene = scene if scene is not None else DrumSceneSpec.from_layout()
        scene = self.scene
        self.margin = float(margin)
        self.power = int(power)
        self.clamp_depth = None if clamp_depth is None else float(clamp_depth)
        self.include_self = bool(include_self)
        self.w_drum = float(w_drum)
        self.w_under = float(w_under)
        self.w_self = float(w_self)
        self.device = torch.device(device)
        self.dtype = dtype
        self.fk = fk if fk is not None else ChainFK(urdf_path, device=self.device, dtype=dtype)

        D = scene.num_drums
        t = lambda arr: torch.tensor(
            np.asarray(arr, dtype=float).tolist(), device=self.device, dtype=dtype
        )
        self.head_centers = t(scene.head_centers)  # (D, 3)
        self.axes = t(scene.axes)  # (D, 3)
        self.radii = t(scene.radii)  # (D,)
        self.half_heights = t(scene.heights / 2.0)  # (D,)
        self.under_half = float(scene.under_depth) / 2.0
        self.under_half_t = t(np.full(D, float(scene.under_depth) / 2.0))  # (D,)
        self.under_clearance = float(scene.under_clearance)
        self.num_drums = D
        off = PELVIS_IN_ENV if scene.frame == "env" else np.zeros(3)
        self.frame_offset = t(off)  # (3,)

        # Robot sample points per hand: [tip, shaft*S, wrist], local wrist frame.
        pts = {}
        for hand in ("left", "right"):
            local = np.concatenate(
                [
                    stick_tip_offset(hand)[None, :],
                    stick_shaft_offsets(hand, shaft_samples),
                    np.zeros((1, 3)),
                ],
                axis=0,
            )
            pts[hand] = t(local)  # (P, 3)
        self.local_points = pts
        self.n_points = int(pts["left"].shape[0])
        self.tip_index = 0
        tip_mask = np.zeros((self.n_points, 1))
        tip_mask[self.tip_index, 0] = 1.0
        self.tip_mask = t(tip_mask)  # (P, 1)
        self.point_radii = t(
            np.concatenate([[TIP_RADIUS], np.full(shaft_samples, STICK_RADIUS), [WRIST_RADIUS]])
        )  # (P,)
        a, b, r = TORSO_CAPSULE
        self.torso_a = t(a)
        self.torso_b = t(b)
        self.torso_r = float(r)
        hc, hr = HEAD_SPHERE
        self.head_c = t(hc)
        self.head_r = float(hr)

    # ------------------------------------------------------------------ core --
    def _cylinder_centers(self, head_centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ax = self.axes.to(head_centers.dtype)
        hh = self.half_heights.to(head_centers.dtype)
        body = head_centers - hh.unsqueeze(-1) * ax
        under_top = head_centers - (2.0 * hh + self.under_clearance).unsqueeze(-1) * ax
        under = under_top - self.under_half * ax
        return body, under

    def _hand_cyl_sdf(
        self, pts: torch.Tensor, centers: torch.Tensor, half_heights: torch.Tensor
    ) -> torch.Tensor:
        """(B, P, D) sphere-vs-solid-cylinder signed distances."""
        x = pts.unsqueeze(2)  # (B, P, 1, 3)
        c = centers.view(1, 1, -1, 3) if centers.dim() == 2 else centers.unsqueeze(1)
        ax = self.axes.to(pts.dtype).view(1, 1, -1, 3)
        sdf = point_cylinder_sdf(
            x, c, ax, self.radii.to(pts.dtype).view(1, 1, -1), half_heights.view(1, 1, -1)
        )
        return sdf - self.point_radii.to(pts.dtype).view(1, -1, 1)

    def _hinge(self, sdf: torch.Tensor) -> torch.Tensor:
        h = (self.margin - sdf).clamp_min(0.0)
        if self.clamp_depth is not None:
            h = h.clamp_max(self.clamp_depth)
        return h if self.power == 1 else h.pow(self.power)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        q: torch.Tensor | None = None,
        *,
        frames: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        drum_head_centers: torch.Tensor | None = None,  # (B, D, 3), scene frame
        target_drum_left: torch.Tensor | None = None,  # (B,) long, -1 = none
        target_drum_right: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SdfPenaltyDetails]:
        """Per-row penalty (B,). Differentiable w.r.t. ``q`` / ``frames``.

        Pass ``frames`` (the output of :meth:`ChainFK.forward`) to reuse a shared
        FK pass; otherwise ``q`` (B, 17) is run through the internal FK.
        """
        if frames is None:
            if q is None:
                raise ValueError("pass either q or frames")
            if q.dim() != 2:
                raise ValueError(f"q must be (B, n), got {tuple(q.shape)}")
            frames = self.fk.forward(q)
        ref = frames["left_wrist"][1]  # (B, 3): dtype/device/batch reference
        dtype, dev = ref.dtype, ref.device
        B = ref.shape[0]
        off = self.frame_offset.to(dtype)

        heads = self.head_centers.to(dtype) if drum_head_centers is None else drum_head_centers
        body_c, under_c = self._cylinder_centers(heads)

        D = self.num_drums
        pen_drum = ref.new_zeros(B)
        pen_under = ref.new_zeros(B)
        pen_self = ref.new_zeros(B)
        min_sdf = ref.new_full((B,), float("inf"))

        targets = {"left": target_drum_left, "right": target_drum_right}
        #: Per-hand clearance, so a caller can ask "how close is the IDLE arm"
        #: without re-running the pass with the other hand removed.  Only filled
        #: when ``return_details``; the combined ``min_sdf`` is unchanged.
        per_hand: dict[str, torch.Tensor] = {}
        for hand, frame_name in (("left", "left_wrist"), ("right", "right_wrist")):
            R, p = frames[frame_name]
            pts = ChainFK.points_world(R, p, self.local_points[hand].to(dtype)) + off

            sdf_drum = self._hand_cyl_sdf(pts, body_c, self.half_heights.to(dtype))
            sdf_under = self._hand_cyl_sdf(pts, under_c, self.under_half_t.to(dtype))

            # Target-drum exemption: this hand's TIP vs its target drum + under.
            # Branch-free -- the original's `if bool(valid.any())` was a sync.
            keep = ref.new_ones(B, self.n_points, D)
            tgt = targets[hand]
            if tgt is not None:
                tgt = tgt.to(device=dev, dtype=torch.long).clamp(-1, D - 1)
                onehot = F.one_hot(tgt.clamp(min=0), D).to(dtype) * (tgt >= 0).to(dtype).unsqueeze(-1)
                keep = keep - self.tip_mask.to(dtype).unsqueeze(0) * onehot.unsqueeze(1)
            hinge_d = self._hinge(sdf_drum) * keep
            hinge_u = self._hinge(sdf_under) * keep
            pen_drum = pen_drum + hinge_d.sum(dim=(1, 2))
            pen_under = pen_under + hinge_u.sum(dim=(1, 2))

            exempt = keep <= 0.5
            masked_d = sdf_drum.masked_fill(exempt, float("inf"))
            masked_u = sdf_under.masked_fill(exempt, float("inf"))
            hand_min = torch.minimum(masked_d.amin(dim=(1, 2)), masked_u.amin(dim=(1, 2)))
            min_sdf = torch.minimum(min_sdf, hand_min)

            if self.include_self:
                Rt, pt = frames["torso"]
                a = pt + (Rt @ self.torso_a.to(dtype).view(1, 3, 1)).squeeze(-1) + off
                b = pt + (Rt @ self.torso_b.to(dtype).view(1, 3, 1)).squeeze(-1) + off
                hc = pt + (Rt @ self.head_c.to(dtype).view(1, 3, 1)).squeeze(-1) + off
                sdf_t = point_capsule_sdf(pts, a.unsqueeze(1), b.unsqueeze(1), self.torso_r)
                sdf_h = point_sphere_sdf(pts, hc.unsqueeze(1), self.head_r)
                sdf_s = torch.minimum(sdf_t, sdf_h) - self.point_radii.to(dtype).view(1, -1)
                pen_self = pen_self + self._hinge(sdf_s).sum(dim=1)
                min_sdf = torch.minimum(min_sdf, sdf_s.amin(dim=1))
                hand_min = torch.minimum(hand_min, sdf_s.amin(dim=1))
            per_hand[hand] = hand_min

        pen = self.w_drum * pen_drum + self.w_under * pen_under + self.w_self * pen_self
        if not return_details:
            return pen
        details = SdfPenaltyDetails(
            min_sdf=min_sdf.detach(),
            pen_drum=pen_drum.detach(),
            pen_under=pen_under.detach(),
            pen_self=pen_self.detach(),
            extras={f"min_sdf_{h}": v.detach() for h, v in per_hand.items()},
        )
        return pen, details
