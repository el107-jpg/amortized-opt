"""The control tick, assembled: schedule -> model -> safety -> plant -> reward.

The Isaac env and the Isaac-free reference driver run *the same* code path.
The only thing that differs is the ``plant``: a callable

    plant(prev_cmd, cmd, q, qd) -> (q_next, qd_next)

which is ``amortized_opt.pd_sim.BatchedPDSim.step`` for the reference driver and
"write the joint position targets, run ``decimation`` physics substeps, read the
articulation back" for Isaac.  Keeping the loop identical is what makes the
Isaac-free tests meaningful and is why the headline comparison (Isaac hit
latency vs the analytic 63 % miss rate) is apples to apples rather than two
different loops that happen to share a checkpoint.

Order of operations in one tick, and why
----------------------------------------

1. read ``(q, qd)`` from the plant;
2. ``IsaacMPC.plan`` -- context, masks, ONE shared forward (the per-env
   parameter noise rides inside it), lead, slew, box clamp;
3. the QP safety filter projects the command if it is inadmissible, and the
   intervention is COUNTED (penalized later, plan item 3);
4. the plant takes the step;
5. ``BatchedSchedule.advance`` -- the clock moves and any hit it passed fires;
6. the reward is scored on the state the plant reached, using the strike rows
   from BEFORE the advance (they describe the hit that was just executed);
7. ``IsaacMPC.commit`` shifts the plan and re-seeds fired hands;
8. the parameter-noise epsilon is redrawn for every env whose segment ended
   (plan item 2: "one eps held fixed across all replans of a hit").

Step 6 before step 7 is not cosmetic: after ``commit`` the goal has already
switched, so a reward computed then would price the NEXT strike's geometry
against the pose the plant reached for the previous one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from amortized_opt.constants import CONTROL_DT

from .mpc import IsaacMPC, TickResult
from .perturb import ParameterNoise
from .reward import ContactScore, ExecutedReward, RewardTerms
from .schedule import BatchedSchedule

__all__ = ["DrumMPCLoop", "LoopStats", "TickRecord"]


@dataclass
class LoopStats:
    """Wall-clock accounting, so ``envs x fps`` and the inference share are
    measured rather than asserted."""

    ticks: int = 0
    t_plan: float = 0.0
    t_filter: float = 0.0
    t_plant: float = 0.0
    t_reward: float = 0.0
    t_total: float = 0.0
    filter_interventions: int = 0

    def summary(self, n_envs: int, dt: float = CONTROL_DT) -> dict:
        t = max(self.t_total, 1e-9)
        return {
            "ticks": self.ticks,
            "env_steps": self.ticks * n_envs,
            "wall_s": self.t_total,
            "control_fps": self.ticks / t,
            "env_fps": self.ticks * n_envs / t,
            "realtime_factor": (self.ticks * dt) / t,
            "share_plan": self.t_plan / t,
            "share_filter": self.t_filter / t,
            "share_plant": self.t_plant / t,
            "share_reward": self.t_reward / t,
            "ms_per_tick_plan": 1000 * self.t_plan / max(self.ticks, 1),
            "ms_per_tick_plant": 1000 * self.t_plant / max(self.ticks, 1),
            "ms_per_tick_reward": 1000 * self.t_reward / max(self.ticks, 1),
            "filter_interventions": self.filter_interventions,
        }


@dataclass
class TickRecord:
    """What one tick produced, for the post-hoc latency histogram."""

    q: np.ndarray            # (B, 17) measured, END of the tick
    cmd: np.ndarray          # (B, 17) executed command
    fired: np.ndarray        # (B, 2) bool
    drum: np.ndarray         # (B, 2) long -- the goal drum at the time
    reward: np.ndarray       # (B,)


class DrumMPCLoop:
    """Assembles the pieces.  Isaac-free; the plant is injected."""

    def __init__(
        self,
        *,
        schedule: BatchedSchedule,
        mpc: IsaacMPC,
        reward: ExecutedReward,
        plant,
        noise: ParameterNoise | None = None,
        safety=None,
        impedance=None,
        record: bool = False,
    ):
        self.sched = schedule
        self.mpc = mpc
        self.reward = reward
        self.plant = plant
        self.noise = noise
        self.safety = safety
        #: Optional :class:`amortized_opt.impedance.StrikeImpedance` -- softens
        #: the striking hand's elbow/wrist gains around each hit.  Both plants
        #: take the multipliers through the same ``gain_scale`` kwarg.
        self.impedance = impedance
        self.stats = LoopStats()
        self.record = bool(record)
        self.history: list[TickRecord] = []

    # ------------------------------------------------------------- one tick --
    @torch.no_grad()
    def tick(self, q: torch.Tensor, qd: torch.Tensor):
        """One control period.  Returns ``(q, qd, RewardTerms, TickResult)``."""
        st = self.stats
        t0 = time.perf_counter()

        # ---- 1-2. plan --------------------------------------------------------
        ctx_strike = self.sched.strike_context()
        res: TickResult = self.mpc.plan(q, qd)
        # ``IsaacMPC.plan`` writes the LED command into ``traj[:, 0]``; the
        # tracking term needs the reference the plan actually nominated for
        # this instant, which is the plan's own first waypoint BEFORE the lead.
        ref0 = self.mpc.last_ref0
        cmd = res.cmd
        t1 = time.perf_counter()

        # ---- 3. safety filter -------------------------------------------------
        engaged = None
        if self.safety is not None:
            cmd, engaged = self.safety.project(cmd, q, qd)
            st.filter_interventions += int(engaged.sum())
        t2 = time.perf_counter()

        # ---- 4. plant ----------------------------------------------------------
        q_before = q
        if self.impedance is not None:
            q, qd = self.plant(self.mpc.prev_cmd, cmd, q, qd,
                               gain_scale=self.impedance.scales(res.hit_index))
        else:
            q, qd = self.plant(self.mpc.prev_cmd, cmd, q, qd)
        t3 = time.perf_counter()

        # ---- 5. clock ----------------------------------------------------------
        fired = self.sched.advance()
        if self.impedance is not None:
            self.impedance.observe_fired(fired)

        # ---- 6. reward, on the EXECUTED state ----------------------------------
        terms, cs = self.reward.step(
            q_before=q_before, q_after=q, qd_after=qd,
            ctx=ctx_strike, hit_index=res.hit_index, fired=fired, ref0=ref0,
            traj=res.traj, elem_mask=res.elem_mask,
            q_plan0=q_before, v_plan0=None,
            drum_heads=ctx_strike.drum_head_centers,
            drum_axes=self.sched.geom.get("axis"),
            drum_radii=self.sched.drum_radius,
            drum_valid=self.sched.drum_valid,
            filter_active=engaged,
        )
        t4 = time.perf_counter()

        # ---- 7. commit ----------------------------------------------------------
        self.mpc.commit(cmd, q, fired)

        # ---- 8. redraw the epsilon of every env whose segment ended --------------
        if self.noise is not None and bool(fired.any()):
            self.noise.redraw(fired.any(dim=1).nonzero(as_tuple=True)[0])

        st.ticks += 1
        st.t_plan += t1 - t0
        st.t_filter += t2 - t1
        st.t_plant += t3 - t2
        st.t_reward += t4 - t3
        st.t_total += t4 - t0
        if self.record:
            self.history.append(TickRecord(
                q=q.detach().float().cpu().numpy().copy(),
                cmd=cmd.detach().float().cpu().numpy().copy(),
                fired=fired.detach().cpu().numpy().copy(),
                drum=ctx_strike.target_drum.detach().cpu().numpy().copy(),
                reward=terms.total.detach().float().cpu().numpy().copy(),
            ))
        return q, qd, terms, res, cs

    # --------------------------------------------------------------- rollout --
    @torch.no_grad()
    def rollout(self, q: torch.Tensor, qd: torch.Tensor, steps: int):
        """``steps`` control ticks.  Returns the final state and a log dict."""
        acc: dict[str, list[float]] = {}
        for _ in range(int(steps)):
            q, qd, terms, _res, _cs = self.tick(q, qd)
            for k, v in terms.log().items():
                acc.setdefault(k, []).append(v)
        return q, qd, {k: float(np.mean(v)) for k, v in acc.items()}

    # ----------------------------------------------------------- q histories --
    def q_hist(self) -> np.ndarray:
        """``(T + 1, B, 17)`` -- what ``probe.hit_latency`` wants, transposed."""
        if not self.history:
            return np.zeros((0, 0, 0))
        return np.stack([r.q for r in self.history])

    def fired_events(self) -> list[tuple[int, int, str, int]]:
        """``(tick, env, hand, drum)`` for every scheduled hit that came due."""
        out = []
        for t, r in enumerate(self.history):
            idx = np.argwhere(r.fired)
            for b, h in idx:
                out.append((t + 1, int(b), ("left", "right")[int(h)], int(r.drum[b, h])))
        return out
