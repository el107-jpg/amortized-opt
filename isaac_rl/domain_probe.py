"""Known-geometry and unmodeled-dynamics stress tests for direct-PID playback.

USD variants live under the experiment directory. Source assets and trained
weights are untouched. All random draws are explicit and persisted.
"""
from pathlib import Path
import json

import numpy as np
import torch

from amortized_opt.fk import StickSpec, stick_tip_offset, stick_axis_local, stick_contact_offset, stick_mount_transform


def overlay_stage(source, destination):
    from pxr import Usd
    source = str(Path(source).resolve())
    stage = Usd.Stage.CreateNew(str(destination))
    stage.GetRootLayer().subLayerPaths = [source]
    original = Usd.Stage.Open(source)
    stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    return stage


def drum_variant(source, destination, names, offsets):
    from pxr import UsdGeom, Gf
    stage = overlay_stage(source, destination)
    paths = {}
    for name, delta in zip(names, offsets):
        matches = [p for p in stage.Traverse() if p.GetName() == name and p.GetTypeName() == "Xform"]
        if len(matches) != 1:
            raise ValueError(f"Expected one physical drum prim for {name}")
        prim = matches[0]
        ops = [o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()
               if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if len(ops) != 1:
            raise ValueError(f"Unexpected drum transform for {name}")
        ops[0].Set(Gf.Vec3d(*(np.asarray(ops[0].Get())+delta)))
        paths[name] = str(prim.GetPath())
    stage.GetRootLayer().Save()
    return paths


def robot_variant(source, destination, sticks):
    from pxr import UsdGeom, Gf
    from scipy.spatial.transform import Rotation
    stage = overlay_stage(source, destination)
    for hand in ("left", "right"):
        sp = StickSpec(hand, **sticks[hand])
        matches = [p for p in stage.Traverse() if p.GetName() == hand+"_stick"]
        if len(matches) != 1:
            raise ValueError(f"Expected one physical {hand} stick prim")
        ops = [o for o in UsdGeom.Xformable(matches[0]).GetOrderedXformOps()
               if o.GetOpType() == UsdGeom.XformOp.TypeOrient]
        if len(ops) != 1:
            raise ValueError("Unexpected stick mount transform")
        position_ops = [o for o in UsdGeom.Xformable(matches[0]).GetOrderedXformOps()
                        if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if len(position_ops) != 1:
            raise ValueError("Unexpected stick mount position")
        position_ops[0].Set(Gf.Vec3d(*stick_mount_transform(hand, sp)[1]))
        x, y, z, w = Rotation.from_matrix(sp.rotation()).as_quat()
        ops[0].Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    stage.GetRootLayer().Save()


def prepare_scene(env_cfg, profile):
    from amortized_opt.rhythm import DrumKit
    out = Path(profile["out"])
    kit = DrumKit.from_layout()
    offsets = np.asarray(profile["drum_offsets_m"])
    paths = drum_variant(env_cfg.drum_kit_cfg.spawn.usd_path, out/"drums.usda", kit.names, offsets)
    env_cfg.drum_kit_cfg.spawn.usd_path = str(out/"drums.usda")
    robot_variant(env_cfg.robot_cfg.spawn.usd_path, out/"robot.usda", profile["sticks"])
    env_cfg.robot_cfg.spawn.usd_path = str(out/"robot.usda")
    env_cfg.dr_enable = False  # own explicit draws, unaffected by discovery gates
    env_cfg.drum_pos_jitter_m = 0.
    for hand in ("left", "right"):
        setattr(env_cfg, hand+"_stick_tip_offset",
                stick_tip_offset(hand, StickSpec(hand, **profile["sticks"][hand])).tolist())
    (out/"physical_assets.json").write_text(json.dumps(dict(
        drum_prim_paths=paths, offsets_m=offsets.tolist(), sticks=profile["sticks"],
        drum_asset=str(out/"drums.usda"), robot_asset=str(out/"robot.usda")), indent=2)+"\n")


def install_geometry(u, profile):
    from amortized_opt.rhythm import DrumKit
    from pxr import UsdGeom, Gf
    import omni.usd
    out = Path(profile["out"])
    if Path(u.cfg.robot_cfg.spawn.usd_path) != out/"robot.usda":
        raise RuntimeError("Robot variant was overwritten before scene creation")
    if Path(u.cfg.drum_kit_cfg.spawn.usd_path) != out/"drums.usda":
        raise RuntimeError("Drum variant was overwritten before scene creation")
    stage = omni.usd.get_context().get_stage()
    verified = {}
    for hand in ("left", "right"):
        matches = [p for p in stage.Traverse() if p.GetName() == hand+"_stick"
                   and "/env_0/" in str(p.GetPath())]
        if len(matches) != 1:
            raise RuntimeError("Cannot verify the spawned physical stick")
        prim = matches[0]
        sp = StickSpec(hand, **profile["sticks"][hand])
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        actual_tip = np.asarray(matrix.Transform(Gf.Vec3d(0., 0., sp.tip_sigma)))
        np.testing.assert_allclose(actual_tip, stick_tip_offset(hand, sp), atol=1e-6, rtol=0)
        verified[hand+"_physical_tip_local"] = actual_tip.tolist()
    for name, delta in zip(DrumKit.from_layout().names, profile["drum_offsets_m"]):
        matches = [p for p in stage.Traverse() if p.GetName() == name and "/env_0/" in str(p.GetPath())]
        if len(matches) != 1:
            raise RuntimeError("Cannot verify the spawned physical drum")
        actual = matches[0].GetAttribute("xformOp:translate").Get()
        np.testing.assert_allclose(actual, delta, atol=1e-6, rtol=0)
    (out/"geometry_verification.json").write_text(json.dumps(dict(
        verified=True, drum_transforms_match_inputs=True, **verified), indent=2)+"\n")
    nominal = torch.tensor(DrumKit.from_layout().positions, device=u.device, dtype=u.drum_pos.dtype)
    torch.testing.assert_close(u.drum_pos, nominal, atol=2e-5, rtol=0)
    delta = torch.tensor(profile["drum_offsets_m"], device=u.device, dtype=u.drum_pos.dtype)
    u.drum_pos += delta
    u.drum_pos_env[:] = u.drum_pos
    probes = []
    for hand in ("left", "right"):
        sp = StickSpec(hand, **profile["sticks"][hand])
        n = int(u.cfg.shaft_hit_samples)
        sigma = sp.tip_sigma-np.arange(1, n+1)*float(u.cfg.shaft_hit_span_m)/max(n, 1)
        probes.append([stick_tip_offset(hand, sp), *[stick_contact_offset(hand, s, sp) for s in sigma]])
    u.stick_probe_offsets.copy_(torch.tensor(np.asarray(probes), device=u.device, dtype=u.stick_probe_offsets.dtype))


def install_policy_geometry(agent, profile):
    """Thread measured stick geometry into terminal features and diagnostic cost."""
    if not any(abs(v) > 0 for s in profile["sticks"].values() for v in s.values()):
        return
    specs = [StickSpec(hand, **profile["sticks"][hand]) for hand in ("left", "right")]
    axis = torch.tensor(np.stack([stick_axis_local(s.hand, s) for s in specs]),
                        device=agent.device, dtype=torch.float32)[None]
    tip = torch.tensor(np.stack([stick_tip_offset(s.hand, s) for s in specs]),
                       device=agent.device, dtype=torch.float32)[None]
    length = torch.tensor([[s.length for s in specs]], device=agent.device)
    valid = torch.ones(1, 2, device=agent.device, dtype=torch.bool)
    original_context = agent.schedule.context
    original_strike = agent.schedule.strike_context
    def context(*args, **kwargs):
        ctx = original_context(*args, **kwargs)
        ctx.stick_axis, ctx.stick_tip, ctx.stick_length, ctx.stick_valid = axis, tip, length, valid
        return ctx
    def strike(*args, **kwargs):
        from amortized_opt.cost import stick_cost_from_context
        ctx = original_strike(*args, **kwargs)
        cost = agent.mpc.descent_cost
        points, axes, sigma = stick_cost_from_context(axis, tip, length, valid, cost.n_points-2)
        ctx.stick_points_local, ctx.stick_axis_local, ctx.stick_tip_sigma = points, axes, sigma
        return ctx
    agent.schedule.context = context
    agent.schedule.strike_context = strike


def apply_dynamics(agent, profile):
    """Perturb real IdealPD gains and PhysX masses/friction after reset."""
    robot = agent.u.robot
    ids = agent._joint_ids
    dynamics = profile["dynamics"]
    rng = np.random.default_rng(profile["seed"]+704)
    ks = rng.uniform(1-dynamics["gain_range"], 1+dynamics["gain_range"], len(ids))
    ds = rng.uniform(1-dynamics["gain_range"], 1+dynamics["gain_range"], len(ids))
    if dynamics["gain_range"]:
        # These saved bases are also the source used by contact compliance,
        # preventing that policy from silently restoring nominal distal gains.
        changed = []
        for act, local, cols, kp, kd in agent._compliance_gain_map:
            kp, kd = kp.clone(), kd.clone()
            kp[:, local] *= kp.new_tensor(ks)[cols]
            kd[:, local] *= kd.new_tensor(ds)[cols]
            act.stiffness.copy_(kp)
            act.damping.copy_(kd)
            changed.append((act, local, cols, kp, kd))
        if not changed:
            raise RuntimeError("Gain probe requires the contact-compliance gain map")
        agent._compliance_gain_map = changed
    body_ids = [i for i,n in enumerate(robot.body_names)
                if any(s in n for s in ("shoulder", "elbow", "wrist"))]
    mass_scale = rng.uniform(1-dynamics["mass_range"], 1+dynamics["mass_range"], len(body_ids))
    if dynamics["mass_range"]:
        mass = robot.root_physx_view.get_masses().clone()
        mass[:, body_ids] *= torch.tensor(mass_scale, device=mass.device, dtype=mass.dtype)
        robot.root_physx_view.set_masses(mass, torch.tensor([0], dtype=torch.int32))
        measured = robot.root_physx_view.get_masses()
        torch.testing.assert_close(measured, mass, atol=1e-6, rtol=1e-5)
    friction = rng.uniform(0, dynamics["friction_max"], len(ids))
    if dynamics["friction_max"]:
        writer = getattr(robot, "write_joint_friction_coefficient_to_sim", None)
        if writer is None:
            raise RuntimeError("Joint-friction write API unavailable; cannot claim this randomizer applied")
        writer(torch.tensor(friction[None], device=agent.u.device, dtype=torch.float32), joint_ids=ids)
    return dict(stiffness_scale=ks.tolist(), damping_scale=ds.tolist(),
                mass_body_names=[robot.body_names[i] for i in body_ids], mass_scale=mass_scale.tolist(),
                friction_coefficient=friction.tolist(), geometry_observed=True,
                latency_randomization=False, sensor_noise_randomization=False)
