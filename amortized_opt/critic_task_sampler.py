"""Bounded global task sampler for online critic outlier exposure."""

from __future__ import annotations

import math
import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from .limits import ETA_VELOCITY, isaac_velocity_limits, urdf_position_limits
from .rhythm import DrumKit, head_plane_basis


class GlobalCriticTaskSampler:
    """Fresh physical-domain samples in the minimal critic feature format."""

    def __init__(self, *, max_tth_s: float = 1.0, min_active_tth_s: float = 0.08):
        names = tuple(G1_UPPER_BODY_JOINTS)
        q_lo, q_hi = urdf_position_limits(names, posture_limits=True)
        qd_hi = ETA_VELOCITY * isaac_velocity_limits(names)
        self.q_lo = torch.as_tensor(q_lo, dtype=torch.float32) / math.pi
        self.q_hi = torch.as_tensor(q_hi, dtype=torch.float32) / math.pi
        self.qd_hi = torch.as_tensor(qd_hi, dtype=torch.float32) / 10.0
        self.min_wp = int(math.ceil(float(min_active_tth_s) / CONTROL_DT))
        self.max_wp = int(math.floor(float(max_tth_s) / CONTROL_DT))
        kit = DrumKit.from_layout()
        self.positions = torch.as_tensor(kit.positions, dtype=torch.float32)
        self.radii = torch.as_tensor(kit.radii, dtype=torch.float32)
        basis = [head_plane_basis(axis) for axis in kit.axes]
        self.basis_u = torch.as_tensor(np.stack([x[0] for x in basis])).float()
        self.basis_v = torch.as_tensor(np.stack([x[1] for x in basis])).float()
        self.eligible = (
            torch.as_tensor(kit.eligible("left"), dtype=torch.long),
            torch.as_tensor(kit.eligible("right"), dtype=torch.long),
        )

    @staticmethod
    def _disk(n: int, radius, *, generator: torch.Generator):
        r = torch.sqrt(torch.rand(int(n), generator=generator)) * torch.as_tensor(radius)
        theta = 2.0 * math.pi * torch.rand(int(n), generator=generator)
        return r * theta.cos(), r * theta.sin()

    def sample(self, n: int, *, generator: torch.Generator) -> torch.Tensor:
        n = int(n)
        q = self.q_lo + torch.rand(n, len(self.q_lo), generator=generator) * (
            self.q_hi - self.q_lo
        )
        qd = (2.0 * torch.rand(n, len(self.qd_hi), generator=generator) - 1.0) * self.qd_hi
        goals = torch.zeros(n, 2, 3)
        tth = torch.zeros(n, 2)
        drums = int(self.positions.shape[0])
        kit_x, kit_y = self._disk(n * drums, 0.04, generator=generator)
        kit_x, kit_y = kit_x.reshape(n, drums), kit_y.reshape(n, drums)
        global_x, global_y = self._disk(n, 0.075, generator=generator)
        global_z = (2.0 * torch.rand(n, generator=generator) - 1.0) * 0.035
        for hand in range(2):
            eligible = self.eligible[hand]
            choice = torch.randint(0, len(eligible) + 1, (n,), generator=generator)
            active = choice < len(eligible)
            rows = active.nonzero(as_tuple=True)[0]
            if not int(rows.numel()):
                continue
            drum = eligible.index_select(0, choice.index_select(0, rows))
            pos = self.positions.index_select(0, drum).clone()
            du, dv = self._disk(
                len(rows), 0.6 * self.radii.index_select(0, drum),
                generator=generator,
            )
            pos += du[:, None] * self.basis_u.index_select(0, drum)
            pos += dv[:, None] * self.basis_v.index_select(0, drum)
            pos[:, 0] += kit_x[rows, drum] + global_x[rows]
            pos[:, 1] += kit_y[rows, drum] + global_y[rows]
            pos[:, 2] += global_z[rows]
            goals[rows, hand] = pos
            tth[rows, hand] = CONTROL_DT * torch.randint(
                self.min_wp, self.max_wp + 1, (len(rows),), generator=generator,
            ).float()
        return torch.cat((q, qd, goals.reshape(n, 6), tth), dim=1)
