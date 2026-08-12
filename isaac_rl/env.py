"""The Isaac plant, stripped to what the amortized optimizer needs.

It is not a subclass of ``G1DrummerEnv``: that class is 6 374
lines and almost all of it -- observation assembly, the lookahead / field-grad /
credit blocks, AMP, the curriculum machinery, the action decoder and its two
dozen knobs, twenty reward terms -- exists to serve an MLP policy whose action
is 17 scalars.  Here the policy is the amortized optimizer and its action is a
whole 50 x 17 trajectory, so none of that applies and inheriting it would mean
carrying 6 000 lines of dead configuration that can still fire.

What IS carried is everything about the plant, because the per-joint reference
lead and the whole phase-2 training distribution were identified against it:

* ``get_g1_cfg(use_ideal_pd=True, pd_damping_scale=2.0, freeze_legs=True)`` --
  the composed USD with the drumsticks welded onto the wrist bodies;
* the drum-kit USD, spawned per env the same way (``AssetBase`` is abstract, so
  the ``UsdFileCfg`` spawn function is called directly);
* the two per-hand ``ContactSensor``s with per-drum filters emitted in drum-id
  order, so filter index == drum id;
* the pelvis pin, re-asserted every substep;
* the FOH position ramp across the ``decimation`` substeps with its matching
  velocity target -- this is what ``amortized_opt.pd_sim`` models analytically,
  and dropping it would invalidate the fitted lead;
* gravity/Coriolis feedforward torque, for the same reason.

The articulation cfg and the drum-kit/layout paths come from :mod:`g1_scene`,
the standalone port of what this module used to import out of ``g1_drummer``.

Not a ``DirectRLEnv``
---------------------

The RL here is *episodic parameter-space search*, not a step-level policy
gradient (plan: "We will not be using PPO for this training").  There is no
``_get_observations``, no action tensor to receive, and RSL-RL's runner is not
in the loop.  Rather than fake a ``gym`` surface for an algorithm that will
never call it, this is a plain ``DirectRLEnv`` subclass used only for its scene
plumbing, driven by :class:`~isaac_rl.loop.DrumMPCLoop` through
:meth:`DrumRLEnv.plant`.  The env's own ``step`` is never used.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF

from .env_cfg import CONTROL_DT, DECIMATION, SIM_DT, DrumRLEnvCfg, assert_rate_consistency

__all__ = ["DrumRLEnv", "make_env"]

HANDS = ("left", "right")



def try_build_rnea(device, dtype: torch.dtype = torch.float32):
    """``BatchedRNEA`` if Pinocchio works IN THIS PROCESS, else ``None``.

    Isaac Sim ships its own Boost/Eigen and its ``omni.*`` extensions load
    before user code, which on this box leaves the conda Pinocchio unable to
    expose ``model.parents``::

        RuntimeError: this Pinocchio build cannot expose model.parents

    ``BatchedRNEA.from_urdf`` needs exactly that (once, to read the kinematic
    tree; evaluation afterwards is pure torch).  Outside Isaac the same call
    works -- ``isaac_rl/scripts/ref_loop.py`` builds it fine -- so this is an
    Isaac-process-only limitation, not a broken vendored module.

    Two consequences, both reported rather than hidden:
      * the gravity/Coriolis feedforward is unavailable inside Isaac, so the
        servo has to hold gravity with position error, which is NOT the plant
        ``amortized_opt.pd_sim`` models and the per-joint lead was fitted
        against;
      * ``FusedCost`` cannot be built either, so the analytic half of the
        reward is unavailable in-process.

    FIXED (RL-1b) by :mod:`isaac_rl.rnea_cache`: the tree is a build-time
    constant of the URDF, so it is frozen to ``isaac_rl/rnea_cache.npz`` by
    ``scripts/build_rnea_cache.py`` in a normal process and reconstructed here
    with no Pinocchio at all.  Verified bit-exact (max \|tau\| difference
    0.0e+00 over 512 random states).  A live build is still tried FIRST, so a
    non-Isaac process never depends on the cache being fresh, and the cache
    carries a URDF fingerprint so a stale one is refused rather than used.

    Returns ``None`` only if BOTH paths fail, and the caller logs that loudly:
    a silently missing feedforward changes the plant.
    """
    from .rnea_cache import build_or_load
    return build_or_load(device=device, dtype=dtype)


def _isaac_imports():
    """Deferred, so the module can be imported (and the cfg inspected) headless."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    return dict(
        sim_utils=sim_utils, Articulation=Articulation, DirectRLEnv=DirectRLEnv,
        DirectRLEnvCfg=DirectRLEnvCfg, InteractiveSceneCfg=InteractiveSceneCfg,
        PhysxCfg=PhysxCfg, SimulationCfg=SimulationCfg,
        RigidBodyMaterialCfg=RigidBodyMaterialCfg,
        TerrainImporterCfg=TerrainImporterCfg, configclass=configclass,
    )


def build_isaac_cfg(cfg: DrumRLEnvCfg):
    """``DrumRLEnvCfg`` -> a live ``DirectRLEnvCfg``, at 250 Hz / decimation 5."""
    assert_rate_consistency()
    I = _isaac_imports()
    from g1_scene.g1 import get_g1_cfg
    from g1_scene.layout import DRUM_KIT_USD_PATH, DRUM_LAYOUT_PATH
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg

    @I["configclass"]
    class _Cfg(I["DirectRLEnvCfg"]):
        decimation = DECIMATION
        episode_length_s = float(cfg.episode_length_s)
        action_space = N_DOF
        observation_space = 1          # unused; DirectRLEnv insists on a number
        state_space = 0
        sim = I["SimulationCfg"](
            dt=SIM_DT,
            render_interval=DECIMATION,
            # Identical contact material to the main repo: the tip's own
            # restitution (0.88) must win the combine, and high friction has to
            # kill tangential velocity before the tip skids across the head.
            physics_material=I["RigidBodyMaterialCfg"](
                restitution=0.0,
                restitution_combine_mode="max",
                friction_combine_mode="max",
            ),
            # CCD was added at 1/120 because a 3 m/s tip travels ~17 mm per
            # physics step -- more than TIP_RADIUS = 14 mm -- so PhysX could
            # resolve the contact past first penetration.  At 1/250 that travel
            # is 12 mm, i.e. inside the tip radius, so CCD is no longer load
            # bearing.  It is kept ON anyway for this scaffold: turning it off
            # is a change to contact behaviour and belongs in the same A/B as
            # the rest of the contact-parameter randomization (RL priority 2),
            # not smuggled in with the rate change.
            physx=I["PhysxCfg"](enable_ccd=True),
        )
        robot_cfg = get_g1_cfg(
            use_ideal_pd=cfg.use_ideal_pd,
            pd_stiffness_scale=cfg.pd_stiffness_scale,
            pd_damping_scale=cfg.pd_damping_scale,
            freeze_legs=cfg.freeze_legs,
        ).replace(prim_path="/World/envs/env_.*/Robot")
        drum_kit_cfg = AssetBaseCfg(
            prim_path="/World/envs/env_.*/DrumKit",
            spawn=I["sim_utils"].UsdFileCfg(usd_path=str(DRUM_KIT_USD_PATH)),
        )
        drum_layout_file = str(DRUM_LAYOUT_PATH)
        scene = InteractiveSceneCfg(
            num_envs=int(cfg.num_envs), env_spacing=float(cfg.env_spacing),
            replicate_physics=True,
        )

    out = _Cfg()
    # THE PHYSICS DEVICE MUST BE SET EXPLICITLY.  ``SimulationCfg.device``
    # defaults to "cuda:0" and ``AppLauncher(device=...)`` does NOT override it
    # -- the repo never notices because ``parse_env_cfg(device=...)`` assigns it.
    # Left at the default, PhysX runs its GPU pipeline on cuda:0 while the app
    # and every torch tensor live on cuda:1, and on this box (cuda:0 held at
    # ~22-29 GB by concurrent sessions) ``DirectRLEnv.__init__`` then BLOCKS
    # FOREVER in scene.update() -> articulation.update() -> joint_acc ->
    # joint_vel -> omni.physics.tensors.get_dof_velocities.  It never raises.
    # Cost of finding this: the hang reproduced at 1/250 AND at 1/120, which is
    # what ruled out the rate decision as the cause.
    out.sim.device = str(cfg.device)
    # Contact sensors need the robot's own colliders reporting.
    out.robot_cfg.spawn.activate_contact_sensors = bool(cfg.contact_sensors)
    return out


def make_env(cfg: DrumRLEnvCfg):
    """Build the Isaac cfg and instantiate :class:`DrumRLEnv`."""
    return DrumRLEnv(build_isaac_cfg(cfg), rl_cfg=cfg)


def _base_class():
    from isaaclab.envs import DirectRLEnv
    return DirectRLEnv


class DrumRLEnv:  # replaced below with a real subclass at import time
    """Placeholder; see :func:`_install`.  ``make_env`` returns the real class."""


def _install():
    """Define the real class once Isaac is importable.

    Isaac Lab's ``DirectRLEnv`` cannot be imported before ``AppLauncher`` has
    started the simulation app, so the subclass is created lazily.  Callers use
    :func:`make_env`, which triggers this.
    """
    global DrumRLEnv
    Base = _base_class()

    class _DrumRLEnv(Base):
        """Scene plumbing + a batched plant callable.  Nothing else."""

        def __init__(self, cfg, rl_cfg: DrumRLEnvCfg, render_mode=None, **kw):
            self.rl_cfg = rl_cfg
            self._contact_enabled = bool(rl_cfg.contact_sensors)
            self.contact_sensors: dict = {}
            super().__init__(cfg, render_mode=render_mode, **kw)

            # ---- joint mapping.  preserve_order is load bearing: every buffer
            # in amortized_opt is in G1_UPPER_BODY_JOINTS order and the model's
            # per-joint lead vector is indexed by it.
            self.joint_ids, names = self.robot.find_joints(
                list(G1_UPPER_BODY_JOINTS), preserve_order=True)
            if tuple(names) != tuple(G1_UPPER_BODY_JOINTS):
                raise RuntimeError(
                    "articulation joint order does not match G1_UPPER_BODY_JOINTS:\n"
                    f"  got      {tuple(names)}\n  expected {G1_UPPER_BODY_JOINTS}")
            self.joint_ids_t = torch.as_tensor(self.joint_ids, device=self.device)

            self.ready_q = self.robot.data.default_joint_pos.clone()
            self.joint_pos_target = self.ready_q.clone()
            self._foh_prev = self.ready_q[:, self.joint_ids].clone()

            # pelvis pin
            root = self.robot.data.default_root_state.clone()
            self._pin_pose = root[:, :7].clone()
            self._pin_vel = torch.zeros_like(root[:, 7:])

            # gravity / Coriolis feedforward, via the vendored batched RNEA so
            # the folder stays self-contained (the repo's own path goes through
            # control/trajopt).  Armature-correct, sync-free, GPU.
            self._rnea = None
            if rl_cfg.gravity_feedforward:
                self._rnea = try_build_rnea(self.device)
                if self._rnea is None:
                    print("[isaac_rl] WARNING: gravity/Coriolis feedforward is OFF "
                          "-- see try_build_rnea", flush=True)
                else:
                    print("[isaac_rl] gravity/Coriolis feedforward ON "
                          "(BatchedRNEA available in-process)", flush=True)
            self._tau_lim = None
            self._lag_buf: torch.Tensor | None = None
            self._lag_n = 0
            self._configure_lag()

        # ------------------------------------------------------------ scene --
        def _setup_scene(self):
            import isaaclab.sim as sim_utils
            from isaaclab.assets import Articulation
            from isaaclab.terrains import TerrainImporterCfg  # noqa: F401
            from isaaclab.sim.spawners.from_files import (
                GroundPlaneCfg,
                spawn_ground_plane,
            )

            self.robot = Articulation(self.cfg.robot_cfg)
            dcfg = self.cfg.drum_kit_cfg
            for i in range(self.cfg.scene.num_envs):
                dcfg.spawn.func(f"/World/envs/env_{i}/DrumKit", dcfg.spawn,
                                translation=dcfg.init_state.pos,
                                orientation=dcfg.init_state.rot)
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
            self.scene.articulations["robot"] = self.robot
            if self._contact_enabled:
                self._setup_contact_sensors()
            light = sim_utils.DomeLightCfg(intensity=2000.0)
            light.func("/World/Light", light)

        def _setup_contact_sensors(self) -> None:
            """Two per-hand sensors, per-drum filters in DRUM-ID ORDER.

            Verbatim in structure from ``G1DrummerEnv._setup_contact_sensors``
            and for its reasons: Isaac Lab's contact filtering only works when
            the sensor's ``prim_path`` resolves to a single prim per env, and a
            single wildcard filter would leave the drum ordering to PhysX prim
            traversal -- not something to bet a reward function on.
            """
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import ContactSensor, ContactSensorCfg
            from g1_scene.g1 import LEFT_WRIST_LINK, RIGHT_WRIST_LINK
            from g1_scene.layout import ordered_drum_names

            # Asserts the ids are 0..D-1 -- filter index == drum id downstream.
            ordered = ordered_drum_names(self.cfg.drum_layout_file)
            self.contact_drum_names = ordered

            root = None
            for cand in (f"/World/envs/env_0/DrumKit/{ordered[0]}/body",
                         f"/World/envs/env_0/DrumKit/DrumKit/{ordered[0]}/body"):
                if sim_utils.find_matching_prim_paths(cand):
                    root = cand.rsplit("/", 2)[0].replace("/World/envs/env_0",
                                                          "/World/envs/env_.*")
                    break
            if root is None:
                raise RuntimeError("could not locate the drum body prims on the stage")
            filters = [f"{root}/{n}/body" for n in ordered]
            for hand, link in (("left", LEFT_WRIST_LINK), ("right", RIGHT_WRIST_LINK)):
                s = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{link}",
                    filter_prim_paths_expr=filters,
                    history_length=0, track_air_time=False,
                    max_contact_data_count_per_prim=8,
                ))
                self.scene.sensors[f"contact_{hand}"] = s
                self.contact_sensors[hand] = s

        # ------------------------------------------- DirectRLEnv obligations --
        # The episodic search never calls these; they exist so DirectRLEnv can
        # be constructed at all.  Deliberately trivial rather than approximate:
        # a plausible-looking dummy reward is worse than an obviously dead one.
        def _pre_physics_step(self, actions): self._actions = actions
        def _apply_action(self): pass
        def _get_observations(self): return {"policy": torch.zeros(self.num_envs, 1,
                                                                   device=self.device)}
        def _get_rewards(self): return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return z, z

        # ------------------------------------------------------------- plant --
        def read_state(self):
            """``(q, qd)`` in ``G1_UPPER_BODY_JOINTS`` order."""
            return (self.robot.data.joint_pos[:, self.joint_ids].clone(),
                    self.robot.data.joint_vel[:, self.joint_ids].clone())

        def _configure_lag(self) -> None:
            lo, hi = self.rl_cfg.lag_substeps
            self._lag_n = int(hi)
            if self._lag_n > 0:
                self._lag_buf = torch.zeros(self._lag_n + 1, self.num_envs, N_DOF,
                                            device=self.device)
                self._lag_k = torch.randint(int(lo), int(hi) + 1, (self.num_envs,),
                                            device=self.device)

        def _feedforward(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
            """Gravity + Coriolis at the measured state, clipped to the effort box.

            ``qdd = 0``, i.e. the inertial term is left OUT.  The repo's env can
            include it (``dynamics_ff_inertial_enable``) driven by the commanded
            acceleration; here the command is a trajectory and the honest
            commanded acceleration is available -- but including it would make
            the plant a different plant from the one ``amortized_opt.pd_sim``
            models and the lead was fitted against.  Flagged in the README as a
            deviation worth an A/B, not decided here.
            """
            if self._rnea is None:
                return None
            z = torch.zeros_like(q)
            tau = self._rnea.rnea(q, qd, z)
            if self._tau_lim is None:
                from amortized_opt.limits import isaac_effort_limits
                self._tau_lim = torch.as_tensor(
                    isaac_effort_limits(G1_UPPER_BODY_JOINTS), device=self.device,
                    dtype=tau.dtype)
            return torch.clamp(tau, -self._tau_lim, self._tau_lim)

        @torch.no_grad()
        def plant(self, prev_cmd: torch.Tensor, cmd: torch.Tensor,
                  q: torch.Tensor, qd: torch.Tensor):
            """One control tick: FOH ramp over ``decimation`` physics substeps.

            Signature matches ``BatchedPDSim.step`` so :class:`DrumMPCLoop` runs
            unchanged against either plant -- which is what makes the Isaac
            latency histogram comparable to the analytic one.
            """
            ids = self.joint_ids
            if self._lag_n > 0:
                # residual command lag, per env, in physics substeps (RL prio 3)
                self._lag_buf = torch.roll(self._lag_buf, 1, dims=0)
                self._lag_buf[0] = cmd
                idx = (self._lag_k // DECIMATION).clamp(0, self._lag_n)
                cmd = self._lag_buf.gather(
                    0, idx.view(1, -1, 1).expand(1, -1, N_DOF)).squeeze(0)
            start = prev_cmd if self.rl_cfg.foh_ramp else cmd
            for i in range(DECIMATION):
                frac = (i + 1) / DECIMATION if self.rl_cfg.foh_ramp else 1.0
                tgt = start + frac * (cmd - start)
                self.joint_pos_target[:, ids] = tgt
                self.robot.set_joint_position_target(tgt, joint_ids=ids)
                self.robot.set_joint_velocity_target(
                    (cmd - start) / CONTROL_DT, joint_ids=ids)
                if self._rnea is not None:
                    qm = self.robot.data.joint_pos[:, ids]
                    vm = self.robot.data.joint_vel[:, ids]
                    self.robot.set_joint_effort_target(self._feedforward(qm, vm),
                                                       joint_ids=ids)
                if self.rl_cfg.pin_pelvis:
                    self.robot.write_root_pose_to_sim(self._pin_pose)
                    self.robot.write_root_velocity_to_sim(self._pin_vel)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=SIM_DT)
            self._foh_prev = cmd.clone()
            return self.read_state()

        # -------------------------------------------------------- diagnostics --
        def contact_forces(self) -> torch.Tensor | None:
            """``(B, 2, D, 3)`` per-(hand, drum) contact force, or ``None``."""
            if not self.contact_sensors:
                return None
            L = self.contact_sensors["left"].data.force_matrix_w
            R = self.contact_sensors["right"].data.force_matrix_w
            return torch.stack((L[:, 0], R[:, 0]), dim=1)

        # ------------------------------------------------------------- reset --
        @torch.no_grad()
        def reset_to(self, env_ids, q: torch.Tensor, qd: torch.Tensor | None = None):
            """Teleport ``env_ids`` to ``q`` (17 DoF), zero velocity by default."""
            env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1).long()
            if env_ids.numel() == 0:
                return
            jp = self.robot.data.default_joint_pos.clone()
            jv = torch.zeros_like(jp)
            jp[:, self.joint_ids] = q
            if qd is not None:
                jv[:, self.joint_ids] = qd
            self.robot.write_joint_state_to_sim(jp[env_ids], jv[env_ids], None, env_ids)
            self.robot.write_root_pose_to_sim(self._pin_pose[env_ids], env_ids)
            self.robot.write_root_velocity_to_sim(self._pin_vel[env_ids], env_ids)
            self.joint_pos_target[env_ids] = jp[env_ids]
            self._foh_prev[env_ids] = q[env_ids] if q.shape[0] == self.num_envs \
                else q
            self.scene.write_data_to_sim()

    DrumRLEnv = _DrumRLEnv
    return _DrumRLEnv


def _make(cfg_isaac, rl_cfg):
    cls = _install()
    return cls(cfg_isaac, rl_cfg=rl_cfg)


def make_env(cfg: DrumRLEnvCfg):  # noqa: F811  (real definition)
    """Build the Isaac cfg and instantiate the env.  Requires a running app."""
    return _make(build_isaac_cfg(cfg), cfg)
