import numpy as np
from pxr import Usd, UsdGeom, Gf

from amortized_opt.fk import StickSpec, stick_tip_offset
from isaac_rl.domain_probe import drum_variant, robot_variant


def test_drum_collision_geometry_moves_with_parent_without_changing_source(tmp_path):
    src = tmp_path/"source.usda"
    stage = Usd.Stage.CreateNew(str(src))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    parent = UsdGeom.Xform.Define(stage, "/World/DrumKit/snare")
    parent.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    body = UsdGeom.Cylinder.Define(stage, "/World/DrumKit/snare/body")
    body.AddTranslateOp().Set(Gf.Vec3d(.45, 0, .88))
    stage.GetRootLayer().Save()
    before = src.read_bytes()
    target = tmp_path/"variant.usda"
    delta = np.array([[.10, -.04, .03]])
    drum_variant(src, target, ["snare"], delta)
    changed = Usd.Stage.Open(str(target))
    xyz = UsdGeom.XformCache().GetLocalToWorldTransform(
        changed.GetPrimAtPath("/World/DrumKit/snare/body")).ExtractTranslation()
    np.testing.assert_allclose(xyz, [.55, -.04, .91], atol=1e-8)
    assert src.read_bytes() == before


def test_rotated_physical_tip_matches_policy_kinematics(tmp_path):
    src = tmp_path/"robot_source.usda"
    stage = Usd.Stage.CreateNew(str(src))
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    for hand in ("left", "right"):
        stick = UsdGeom.Xform.Define(stage, "/Robot/"+hand+"_stick")
        stick.AddTranslateOp().Set(Gf.Vec3d(.2, -.06 if hand == "left" else .06, 0))
        stick.AddOrientOp().Set(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
        tip = UsdGeom.Sphere.Define(stage, "/Robot/"+hand+"_stick/tip")
        tip.AddTranslateOp().Set(Gf.Vec3d(0, 0, .2))
    stage.GetRootLayer().Save()
    specs = {"left": dict(tilt=.1, roll=-.12), "right": dict(tilt=-.08, roll=.16)}
    target = tmp_path/"robot_variant.usda"
    robot_variant(src, target, specs)
    changed = Usd.Stage.Open(str(target))
    for hand in ("left", "right"):
        position = UsdGeom.XformCache().GetLocalToWorldTransform(
            changed.GetPrimAtPath("/Robot/"+hand+"_stick/tip")).ExtractTranslation()
        np.testing.assert_allclose(position, stick_tip_offset(hand, StickSpec(hand, **specs[hand])), atol=1e-7)
