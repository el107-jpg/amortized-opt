"""Bind registered upstream drum tasks to this decoder's shared scene geometry."""
from __future__ import annotations

import importlib
import numpy as np
from pathlib import Path
from g1_scene import g1 as geometry

GEOMETRY_VERSION = 'decoder_shared_constants_v1'
DRUM_GEOMETRY_VERSION = 'thick_cymbal_top_entry_side_5mm_v1'
CONTACT_SCORING_VERSION = 'distal20cm_capsule_top_side5mm_contact_v5'
ROOT = Path(__file__).resolve().parents[1]
DRUM_LAYOUT_PATH = ROOT/'assets/drum_kit_layout.yaml'
DRUM_USD_PATH = ROOT/'assets/drums/simple_drum_kit_6drums.usd'


def bind_registered_task_geometry(config_module=None, env_module=None):
    """Call after launching Isaac, before constructing registered task cfgs.

    The external task otherwise rebuilds a different articulation in post_init
    and samples shaft contacts using its own stale mount constants.
    """
    if config_module is None:
        config_module = importlib.import_module('g1_drummer.tasks.direct.g1_drum.g1_drummer_env_cfg')
    if env_module is None:
        env_module = importlib.import_module('g1_drummer.tasks.direct.g1_drum.g1_drummer_env')
    config_module.get_g1_cfg = geometry.get_g1_cfg
    env_module.stick_shaft_offsets_in_wrist = geometry.stick_shaft_offsets_in_wrist
    if hasattr(env_module,'ContactStrikeTracker'):
        from amortized_opt.rhythm import DrumKit
        from isaac_rl.cymbal_contact_scoring import make_capsule_tracker
        env_module.ContactStrikeTracker=make_capsule_tracker(
            env_module.ContactStrikeTracker,DrumKit.from_layout(DRUM_LAYOUT_PATH))


def configure_drum_kit_geometry(cfg):
    """Install the same local thick-collider kit layout and USD for Isaac."""
    if not DRUM_LAYOUT_PATH.is_file() or not DRUM_USD_PATH.is_file():
        raise FileNotFoundError('Local drum-kit layout/USD is missing')
    cfg.drum_layout_file=str(DRUM_LAYOUT_PATH)
    cfg.drum_kit_cfg.spawn.usd_path=str(DRUM_USD_PATH)


def configure_tip_geometry(cfg):
    cfg.left_stick_tip_offset = list(geometry.TIP_OFFSET_IN_WRIST_LEFT)
    cfg.right_stick_tip_offset = list(geometry.TIP_OFFSET_IN_WRIST_RIGHT)
    cfg.stick_tip_offset = list(geometry.TIP_OFFSET_IN_WRIST_LEFT)
    cfg.shaft_hit_span_m = geometry.SHAFT_HIT_SPAN_M


def mount_positions():
    return [list(geometry.STICK_MOUNT_POS_LEFT), list(geometry.STICK_MOUNT_POS_RIGHT)]


def audit_drum_kit(path=DRUM_USD_PATH,layout=DRUM_LAYOUT_PATH):
    """Verify actual USD collider thickness and head-centre alignment."""
    import yaml
    from pxr import Usd,UsdPhysics
    stage=Usd.Stage.Open(str(path))
    if stage is None:raise RuntimeError(f'Cannot open drum kit USD {path}')
    specs=yaml.safe_load(Path(layout).read_text())['drums']
    result={}
    for name,spec in specs.items():
        prim=stage.GetPrimAtPath(f'/World/DrumKit/{name}/body')
        if (not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or not prim.GetAttribute('physics:kinematicEnabled').Get()):
            raise RuntimeError(f'{name} is not a colliding kinematic rigid body')
        height=float(prim.GetAttribute('height').Get())
        radius=float(prim.GetAttribute('radius').Get())
        if abs(height-float(spec['height']))>1e-6 or abs(radius-float(spec['radius']))>1e-6:
            raise RuntimeError(f'{name} USD and layout geometry differ')
        if name in ('hihat','cymbal_left','cymbal_right') and abs(height-.10)>1e-6:
            raise RuntimeError(f'{name} must retain 10 cm physical thickness')
        center=np.asarray(spec['position'],float)
        normal=np.asarray(spec.get('normal',[0,0,1]),float);normal/=np.linalg.norm(normal)
        body=np.asarray(prim.GetAttribute('xformOp:translate').Get(),float)
        if np.abs(body-(center-normal*height/2)).max()>1e-6:
            raise RuntimeError(f'{name} USD top face differs from decoder head centre')
        result[name]=dict(height_m=height,radius_m=radius,collider=True)
    return result


def audit_composed_sticks(path, *, stage=None):
    """Verify the actual USD colliders' transforms and sizes, not only probes."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if stage is None:
        stage = Usd.Stage.Open(str(path))
    result = {}
    for hand in ('left','right'):
        matches = [p for p in stage.Traverse() if p.GetName() == f'{hand}_stick']
        if len(matches) != 1:
            raise RuntimeError(f'Expected one authored {hand} stick in {path}')
        prim = matches[0]
        tip = stage.GetPrimAtPath(prim.GetPath().AppendChild('tip'))
        shaft = stage.GetPrimAtPath(prim.GetPath().AppendChild('shaft'))
        tip_local = tip.GetAttribute('xformOp:translate').Get()
        transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        authored = np.asarray(transform.Transform(Gf.Vec3d(*tip_local)))
        expected = np.asarray(getattr(geometry,f'TIP_OFFSET_IN_WRIST_{hand.upper()}'))
        error = float(np.abs(authored-expected).max())
        if error > 1e-6 or not tip.HasAPI(UsdPhysics.CollisionAPI) or not shaft.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f'{hand} physical collider does not match decoder geometry')
        tip_radius = float(tip.GetAttribute('radius').Get())
        shaft_radius = float(shaft.GetAttribute('radius').Get())
        if abs(tip_radius-geometry.TIP_RADIUS) > 1e-7 or abs(shaft_radius-geometry.STICK_RADIUS) > 1e-7:
            raise RuntimeError('Physical stick radii differ from decoder geometry')
        result[hand] = dict(authored_tip_offset_local_m=authored.tolist(),
            decoder_tip_offset_local_m=expected.tolist(), max_coordinate_error_m=error,
            tip_radius_m=tip_radius, shaft_radius_m=shaft_radius)
    return result
