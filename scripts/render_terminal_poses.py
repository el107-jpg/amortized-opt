#!/usr/bin/env python
"""Isaac stills of the trained terminal-pose network on the nominal kit.

One reached hit per drum, from the striking side.  Kit and stick jitter are
off so the pose matches the USD the renderer spawns.

    python scripts/render_terminal_poses.py --ckpt runs/terminal_pose/ablate_pen/optnet_gpu.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CASES = (
    ("left", "snare"),
    ("right", "snare"),
    ("left", "hihat"),
    ("right", "floor_tom"),
    ("left", "cymbal_left"),
    ("right", "cymbal_right"),
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="runs/terminal_pose/ablate_pen/optnet_gpu.pt")
    p.add_argument("--out", default="runs/terminal_pose/isaac")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args(argv)


def look_at_quat(eye, target):
    """(w, x, y, z) OpenGL camera rotation.  Same convention as render_stick_pose."""
    import numpy as np

    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], axis=1)
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


def select_poses(ckpt_path: str, device: str):
    """Nominal-kit poses.  Each case is a reached hit near the median miss."""
    import numpy as np
    import torch

    from amortized_opt.fk import StickRandomization
    from amortized_opt.rhythm import KitRandomization
    from amortized_opt.terminal_pose import (
        TerminalPoseCost, TerminalPoseNet, sample_terminal_batch,
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state = ck["model"]
    if "in_dim" in ck:
        in_dim = int(ck["in_dim"])
    else:
        first = "pre.weight" if bool(ck.get("optnet", True)) else "hidden.0.weight"
        in_dim = int(state[first].shape[1])
    net = TerminalPoseNet(
        in_dim=in_dim,
        width=int(ck.get("width", 128)),
        depth=int(ck.get("depth", 2)),
        optnet=bool(ck.get("optnet", True)),
        n_constraints=int(ck.get("n_constraints", 4)),
        centered_qp=bool(ck.get("centered_qp", False)),
        mix_width=ck.get("mix_width"),
        qp_out=bool(ck.get("qp_out", False)),
    ).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    cost = TerminalPoseCost(device=dev)
    batch = sample_terminal_batch(
        2048,
        rng=np.random.default_rng(12345),
        kit_rand=KitRandomization(
            per_drum_xy=0.0, global_xy=0.0, global_z=0.0,
            per_drum_tilt=0.0, per_drum_radius=0.0,
        ),
        stick_rand=StickRandomization(),
        idle_frac=0.0,
        device=dev,
    )
    qs = []
    with torch.no_grad():
        for i in range(0, batch.features.shape[0], 64):
            qs.append(net(batch.features[i:i + 64, :net.in_dim]))
        q = torch.cat(qs, 0)
        out = cost(q, batch, None)
    hit = out.aux["hit_dist"].detach().cpu().numpy()
    elev = out.aux["stick_elev_deg"].detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    active = batch.active.detach().cpu().numpy()
    drum_index = batch.drum_index.detach().cpu().numpy()
    names = list(batch.drum_names)
    name_i = {n: i for i, n in enumerate(names)}
    hand_i = {"left": 0, "right": 1}
    chosen = []
    for hand, drum in CASES:
        hi = hand_i[hand]
        m = np.where(active[:, hi] & (drum_index[:, hi] == name_i[drum]))[0]
        reached = m[hit[m, hi] < 0.02]
        pool = reached if len(reached) else m
        if len(pool) == 0:
            raise SystemExit(f"no sample for {hand} {drum}")
        med = np.median(hit[pool, hi])
        j = int(pool[np.argmin(np.abs(hit[pool, hi] - med))])
        other = 1 - hi
        other_name = names[int(drum_index[j, other])] if active[j, other] else "idle"
        chosen.append({
            "hand": hand,
            "drum": drum,
            "q": q_np[j].astype(np.float64),
            "miss_mm": float(1000 * hit[j, hi]),
            "elev": float(elev[j, hi]),
            "other": other_name,
        })
        print(
            f"[pose] {hand} {drum} miss {chosen[-1]['miss_mm']:.1f} mm "
            f"elev {chosen[-1]['elev']:+.1f} other={other_name}",
            flush=True,
        )
    return chosen


def main(argv=None) -> int:
    args = parse_args(argv)
    cases = select_poses(args.ckpt, args.device)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app

    import numpy as np
    import torch
    import imageio.v3 as iio
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from PIL import Image, ImageDraw

    from amortized_opt.constants import G1_UPPER_BODY_JOINTS
    from g1_scene.g1 import get_g1_cfg
    from g1_scene.layout import DRUM_KIT_USD_PATH

    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0, device=args.device))
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.92, 0.95))
    light.func("/World/light", light)
    key = sim_utils.DistantLightCfg(intensity=1800.0, angle=1.0)
    key.func("/World/key", key, orientation=(0.85, 0.15, 0.50, 0.0))

    robot = Articulation(get_g1_cfg().replace(prim_path="/World/Robot"))
    kit = sim_utils.UsdFileCfg(usd_path=str(DRUM_KIT_USD_PATH))
    kit.func("/World/DrumKit", kit)

    # One camera per side.  Offsets are baked at spawn; look-at cannot be
    # moved later on this warp build.
    # Behind the striking shoulder, looking forward onto the kit, so the arm
    # sits between the camera and the drum instead of behind it.
    cam_spec = {
        "left": (np.array([-0.85, 1.25, 1.55]), np.array([0.42, 0.18, 0.95])),
        "right": (np.array([-0.85, -1.25, 1.55]), np.array([0.42, -0.12, 0.92])),
    }
    cams = {}
    fov = 48.0
    focal = 12.0 / math.tan(math.radians(fov) / 2.0)
    for side, (eye, tgt) in cam_spec.items():
        cfg = CameraCfg(
            prim_path=f"/World/cam_{side}",
            update_period=0.0,
            width=int(args.width),
            height=int(args.height),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal, focus_distance=2.0,
                horizontal_aperture=24.0, clipping_range=(0.05, 50.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in eye),
                rot=look_at_quat(eye, tgt),
                convention="opengl",
            ),
        )
        cams[side] = Camera(cfg)

    sim.reset()
    names = list(robot.data.joint_names)
    upper_idx = []
    for joint in G1_UPPER_BODY_JOINTS:
        if joint not in names:
            raise SystemExit(f"articulation is missing {joint}")
        upper_idx.append(names.index(joint))
    root = robot.data.root_pos_w[0].detach().cpu().numpy()
    print(f"[root] pelvis {np.round(root, 4)}  expected [0, 0, 0.8]", flush=True)

    from amortized_opt.constants import PELVIS_IN_ENV
    from amortized_opt.fk import ChainFK
    from amortized_opt.cost import FusedCost

    fk = ChainFK(
        joint_names=G1_UPPER_BODY_JOINTS,
        frames=dict(FusedCost.FRAMES),
        device="cpu",
        dtype=torch.float64,
    )
    body_names = list(robot.data.body_names)
    wrist_body = {"left": body_names.index("left_wrist_yaw_link"),
                  "right": body_names.index("right_wrist_yaw_link")}

    def hold(q):
        """Photograph the commanded joints.  ``sim.step`` integrates the arm
        drives toward the default stance, so the picture would not be the
        model's output.  ``render`` refreshes the viewport with physics paused."""
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        for _ in range(6):
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            sim.forward()
            sim.render()

    written = []
    for case in cases:
        q = robot.data.default_joint_pos.clone()
        pose = torch.as_tensor(case["q"], device=q.device, dtype=q.dtype)
        for col, value in zip(upper_idx, pose):
            q[:, col] = value
        cam = cams[case["hand"]]
        hold(q)
        cam.update(dt=0.0)
        got = robot.data.joint_pos[0, upper_idx].detach().cpu().numpy()
        cmd = np.asarray(case["q"], float)
        # Wrist link in the world versus the analytic wrist joint frame.
        hi = 0 if case["hand"] == "left" else 1
        frames = fk.forward(torch.as_tensor(cmd, dtype=torch.float64).view(1, -1))
        R_fk, p_fk = frames[f"{case['hand']}_wrist"]
        p_fk_w = p_fk[0].numpy() + np.asarray(PELVIS_IN_ENV, float)
        p_sim = robot.data.body_pos_w[0, wrist_body[case["hand"]]].detach().cpu().numpy()
        quat = robot.data.body_quat_w[0, wrist_body[case["hand"]]].detach().cpu().numpy()
        # body_quat_w is wxyz
        print(
            f"[check] {case['hand']} {case['drum']}  "
            f"joint err {np.abs(got - cmd).max():.4f} rad  "
            f"wrist |sim-fk| {np.linalg.norm(p_sim - p_fk_w) * 1000:.1f} mm",
            flush=True,
        )
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        rgb = rgb[..., :3]
        path = out_dir / f"{case['hand']}_{case['drum']}.png"
        iio.imwrite(path, rgb)
        written.append((path, case))
        print(f"[shot] {path}", flush=True)

    # Labeled 2x3 sheet.
    thumbs = []
    for path, case in written:
        im = Image.open(path).convert("RGB")
        bar_h = 36
        canvas = Image.new("RGB", (im.width, im.height + bar_h), (250, 250, 248))
        canvas.paste(im, (0, bar_h))
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{case['hand']} {case['drum'].replace('_', ' ')}    "
            f"miss {case['miss_mm']:.1f} mm    stick {case['elev']:+.0f}°"
        )
        draw.text((12, 8), title, fill=(20, 20, 20))
        thumbs.append(canvas)
    cols, rows = 2, 3
    w, h = thumbs[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet_path = out_dir / "gallery.png"
    sheet.save(sheet_path)
    print(f"[sheet] {sheet_path}", flush=True)

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
