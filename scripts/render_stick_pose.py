#!/usr/bin/env python
"""Render the G1 + drumsticks at a static pose from several camera angles.

A look-at-what-you-have tool for the stick geometry in
``amortized_opt/constants.py``: the mount rotation is stored as a quaternion,
so "what angle are the sticks at" is not readable from the source.  This spawns
the real composed USD (the same one training and Isaac playback use, sticks
authored by ``g1_scene.g1``) at a chosen pose and takes stills from an orbit
plus a close-up of each wrist.

    python scripts/render_stick_pose.py --pose ready --out runs/render/sticks

Needs the Isaac preamble (see CLAUDE.md).  Pick a GPU with free VRAM through
``CUDA_VISIBLE_DEVICES`` -- a "NoneType has no attribute
create_articulation_view" crash is really VRAM exhaustion on *physical* GPU 0,
and ``--device`` does not move the allocation.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pose", default="ready",
                   choices=("ready", "idle_home", "zero"),
                   help="ready = G1_READY_JOINT_OFFSETS (the pose the frozen "
                        "columns hold and the eval grid jitters around); "
                        "idle_home = the park posture from assets/; zero = URDF neutral")
    p.add_argument("--out", default="runs/render/sticks", help="output directory")
    p.add_argument("--angles", default="front,3quarter,side,top",
                   help="comma-separated names from play_isaac.ANGLE_OFFSETS")
    p.add_argument("--cam-zoom", type=float, default=1.0,
                   help="camera distance multiplier, as in play_isaac.py")
    p.add_argument("--no-wrist", action="store_true",
                   help="skip the two wrist close-ups (the named angles only)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--px-height", type=int, default=720)
    p.add_argument("--no-kit", action="store_true", help="omit the drum kit")
    p.add_argument("--device", default="cuda:0",
                   help="Isaac device AFTER CUDA_VISIBLE_DEVICES masking")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- Isaac must be launched before anything imports isaaclab.envs --------
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app

    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    import imageio.v3 as iio

    from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
    from g1_scene.g1 import get_g1_cfg
    from g1_scene.layout import DRUM_KIT_USD_PATH

    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- scene --------------------------------------------------------------
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0, device=args.device))
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.92, 0.95)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0)
    )
    # A key light so the sticks cast a readable shadow instead of flat ambient.
    sim_utils.DistantLightCfg(intensity=1800.0, angle=1.0).func(
        "/World/key", sim_utils.DistantLightCfg(intensity=1800.0, angle=1.0),
        orientation=(0.85, 0.15, 0.50, 0.0),
    )

    robot_cfg = get_g1_cfg().replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    if not args.no_kit:
        sim_utils.UsdFileCfg(usd_path=str(DRUM_KIT_USD_PATH)).func(
            "/World/DrumKit", sim_utils.UsdFileCfg(usd_path=str(DRUM_KIT_USD_PATH))
        )

    # ---- cameras -------------------------------------------------------------
    # Aim is baked into ``OffsetCfg`` at spawn.  ``Camera.set_world_poses_from_view``
    # -- Isaac's own look-at, which would be the obvious choice -- goes through
    # the FABRIC write path, and that raises here: the kernel calls
    # ``wp.transform_compose``, which this warp (1.7.1) does not have.
    #
    # The convention matters and cost a round the first time.  A rotation whose
    # columns are [right, up, -forward] is OPENGL (-Z forward, +Y up), NOT
    # ``"world"`` (+X forward, +Z up).  Handing an OpenGL matrix to
    # ``convention="world"`` aimed every camera at the floor.
    def look_at_quat(eye, target):
        """(w, x, y, z) for a camera at ``eye`` looking at ``target``, OPENGL."""
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-6:            # looking straight down
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, up, -fwd], axis=1)     # columns: x, y, z
        t = np.trace(R)
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                 (R[1, 0] - R[0, 1]) / s]
        else:
            i = int(np.argmax(np.diag(R)))
            if i == 0:
                s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                     (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
            elif i == 1:
                s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                     0.25 * s, (R[1, 2] + R[2, 1]) / s]
            else:
                s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        q = np.asarray(q, float)
        return tuple(float(v) for v in q / np.linalg.norm(q))
    # The SAME table and look-at point ``scripts/play_isaac.py`` records video
    # with, so these stills frame the robot exactly like the existing playback.
    ANGLE_OFFSETS = {
        "front": [0.0, 2.2, 1.1],
        "3quarter": [1.6, 1.2, 1.25],
        "side": [-2.2, 0.0, 1.1],
        "top": [0.3, 0.3, 2.8],
    }
    CAM_TARGET = np.array([0.30, 0.0, 0.95])          # play_isaac's cam_target

    shots: list[tuple[str, tuple, tuple, float]] = []   # name, eye, target, fov
    wanted = [a.strip() for a in args.angles.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in ANGLE_OFFSETS]
    if unknown:
        raise SystemExit(f"unknown angle(s) {unknown}; choose from {sorted(ANGLE_OFFSETS)}")
    for name in wanted:
        eye = np.asarray(ANGLE_OFFSETS[name], float) * float(args.cam_zoom)
        shots.append((name, tuple(eye), tuple(CAM_TARGET), 50.0))
    # Wrist close-ups: tight FOV, near the hands, so the mount angle is legible.
    # Not from play_isaac -- it has no close-up -- but that is the whole point
    # of this script, so they are additive and can be turned off.
    if not args.no_wrist:
        for hand, sgn in (("left", +1.0), ("right", -1.0)):
            tgt = np.array([0.28, sgn * 0.22, 1.02])
            shots.append((f"wrist_{hand}", tuple(tgt + np.array([0.55, sgn * 0.42, 0.30])),
                          tuple(tgt), 30.0))
            shots.append((f"wrist_{hand}_top", tuple(tgt + np.array([0.05, sgn * 0.05, 0.75])),
                          tuple(tgt), 30.0))

    cams = []
    for name, eye, tgt, fov in shots:
        # Focal length from horizontal FOV on a 24 mm-wide aperture.  Isaac's
        # own viewport sits near 48 deg; the first pass used 24 deg, which is a
        # 56 mm telephoto and put the camera inside the drum kit.
        f = 12.0 / math.tan(math.radians(fov) / 2.0)
        cfg = CameraCfg(
            prim_path=f"/World/cam_{name}",
            update_period=0.0,
            width=int(args.width), height=int(args.px_height),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=f, focus_distance=2.0,
                horizontal_aperture=24.0, clipping_range=(0.05, 50.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in eye),
                                       rot=look_at_quat(eye, tgt),
                                       convention="opengl"),
        )
        cams.append((name, Camera(cfg)))

    sim.reset()

    # ---- pose ---------------------------------------------------------------
    names = list(robot.data.joint_names)
    q = robot.data.default_joint_pos.clone()
    if args.pose != "zero":
        if args.pose == "ready":
            target = dict(zip(G1_UPPER_BODY_JOINTS, ready_joint_pos(G1_UPPER_BODY_JOINTS)))
        else:
            from amortized_opt.idle_home import resolve_idle_home
            from amortized_opt.warmstart import arm_slice

            tbl = resolve_idle_home("default", joint_names=G1_UPPER_BODY_JOINTS)
            if tbl is None:
                raise SystemExit("idle-home table not built; run scripts/build_idle_home.py")
            base = np.asarray(ready_joint_pos(G1_UPPER_BODY_JOINTS), float)
            for hand in ("left", "right"):
                base[arm_slice(hand, G1_UPPER_BODY_JOINTS)] = tbl.arm_pose(hand)
            target = dict(zip(G1_UPPER_BODY_JOINTS, base))
        miss = [n for n in target if n not in names]
        if miss:
            print(f"[warn] joints not in the articulation: {miss}", flush=True)
        for n, v in target.items():
            if n in names:
                q[:, names.index(n)] = float(v)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.reset()

    # Settle the render graph: cameras need a few rendered frames before the
    # RTX buffers hold the posed robot rather than the spawn state.
    for _ in range(12):
        sim.step(render=True)
        for _n, c in cams:
            c.update(dt=0.0)

    # ---- capture ------------------------------------------------------------
    written = []
    for name, cam in cams:
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        rgb = rgb[..., :3]
        path = out_dir / f"{args.pose}_{name}.png"
        iio.imwrite(path, rgb)
        written.append(path)
        print(f"[shot] {path}", flush=True)

    # ---- the numbers the pictures are of ------------------------------------
    from amortized_opt.constants import STICK_LENGTH, STICK_RADIUS
    from amortized_opt.fk import stick_axis_local, stick_mount_transform

    print(f"\n[stick] LENGTH {STICK_LENGTH} m  RADIUS {STICK_RADIUS} m  (tip at L/2 "
          f"= {STICK_LENGTH / 2:.3f} m from the mount)")
    for hand in ("left", "right"):
        R, p = stick_mount_transform(hand)
        ax = stick_axis_local(hand)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.trace(R) - 1) / 2))))
        print(f"[stick] {hand:5s} mount {np.round(p, 4)}  rot {ang:.2f} deg  "
              f"axis(handle->tip) {np.round(ax, 4)}  "
              f"[to wrist x {math.degrees(math.acos(ax[0])):.2f} deg, "
              f"y {math.degrees(math.acos(ax[1])):.2f} deg, "
              f"z {math.degrees(math.acos(ax[2])):.2f} deg]")

    print(f"\n[done] {len(written)} stills -> {out_dir}", flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
