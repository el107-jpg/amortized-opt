#!/usr/bin/env python3
"""Private rollout worker: no optimizer, resampling or independent chart selection."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    folder = parser.parse_args().folder
    from isaac_rl.chart_cma_distributed import atomic_json
    from scripts.train_chart_cma import setup_isaac
    import torch
    config = json.loads((folder/'setup.json').read_text())
    checkpoint = Path(config['checkpoint'])
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    launcher = env = None
    try:
        args = SimpleNamespace(**config['args'])
        launcher, env, evaluator = setup_isaac(args, checkpoint, source, config['seed_clip'])
        from isaac_rl.decoder_scene import (mount_positions,GEOMETRY_VERSION,
            DRUM_GEOMETRY_VERSION,CONTACT_SCORING_VERSION,DRUM_LAYOUT_PATH,DRUM_USD_PATH,
            audit_composed_sticks,audit_drum_kit)
        u = env.unwrapped
        if (Path(u.cfg.drum_kit_cfg.spawn.usd_path).resolve()!=DRUM_USD_PATH.resolve()
                or not getattr(u.contact_tracker,'_capsule_tracker',False)):
            raise RuntimeError('Registered Isaac task did not install thick kit/capsule scoring')
        drum_audit=audit_drum_kit()
        geometry = dict(tip_offsets_local_m=torch.cat((u.left_stick_tip_offset,u.right_stick_tip_offset)).cpu().tolist(),
            mount_positions_local_m=mount_positions(),
            pelvis_in_env_m=(u.robot.data.root_pos_w[0]-u.scene.env_origins[0]).cpu().tolist())
        atomic_json(folder/'ready.json', dict(device=args.device, envs=args.envs,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), tip_geometry=geometry,
            simulation_geometry_version=GEOMETRY_VERSION, robot_usd=str(u.cfg.robot_cfg.spawn.usd_path),
            collider_geometry_audit=audit_composed_sticks(u.cfg.robot_cfg.spawn.usd_path),
            rollout_runtime_version=(CONTACT_SCORING_VERSION
                if getattr(args,'clear_reset_contacts',False) and getattr(args,'match_decoder_speed',False)
                else CONTACT_SCORING_VERSION+'_unmatched'),
            drum_geometry_version=DRUM_GEOMETRY_VERSION,
            drum_layout_sha256=hashlib.sha256(DRUM_LAYOUT_PATH.read_bytes()).hexdigest(),
            drum_usd_sha256=hashlib.sha256(DRUM_USD_PATH.read_bytes()).hexdigest(),
            drum_kit_audit=drum_audit,
            fixed_hit_scoring=getattr(u,'_fixed_hit_scoring',None),decoder_amp=str(evaluator.mpc.cfg.amp_dtype),
            impact_speed_target=(dict(version='decoder_linear_impact_speed_v1',speed_min_m_s=evaluator.mpc.sched.cfg.speed_min,
                speed_max_m_s=evaluator.mpc.sched.cfg.speed_max) if evaluator.match_decoder_speed else None)))
        request = 1
        while not (folder/'stop').exists():
            key = f'request_{request:06d}'
            path = folder/(key+'.pkl')
            if not path.exists():
                time.sleep(.1)
                continue
            try:
                payload = pickle.loads(path.read_bytes())
                if isinstance(payload, dict):
                    episodes = payload['episodes']
                    options = dict(initial_states=payload['initial_states'],
                                   collect_post_hits=payload['collect_post_hits'],
                                   collect_prefirst_snapshots=payload.get('collect_prefirst_snapshots',False),
                                   command_overrides=payload.get('command_overrides'))
                else:
                    episodes, options = payload, {}
                def progress(offset, total, tick, ticks):
                    atomic_json(folder/(key+'_progress.json'), dict(offset=offset, episodes=total, tick=tick, ticks=ticks))
                started = time.monotonic()
                reports = evaluator.evaluate(episodes, progress, **options)
                atomic_json(folder/(key+'.json'), dict(reports=reports, elapsed_s=time.monotonic()-started))
                path.unlink()
            except BaseException as exc:
                atomic_json(folder/(key+'.json'), dict(error=repr(exc)))
                raise
            request += 1
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == '__main__':
    main()
