from types import SimpleNamespace
import os
from pathlib import Path
import numpy as np
import pytest

from isaac_rl.decoder_scene import (bind_registered_task_geometry,configure_tip_geometry,
    configure_drum_kit_geometry,audit_drum_kit)
from g1_scene import g1
from amortized_opt.fk import stick_tip_offset, stick_contact_offset


def test_registered_task_rebuild_and_probes_use_shared_geometry():
    old=lambda:None
    cfg_module=SimpleNamespace(get_g1_cfg=old)
    env_module=SimpleNamespace(stick_shaft_offsets_in_wrist=old)
    bind_registered_task_geometry(cfg_module,env_module)
    assert cfg_module.get_g1_cfg is g1.get_g1_cfg
    assert env_module.stick_shaft_offsets_in_wrist is g1.stick_shaft_offsets_in_wrist
    cfg=SimpleNamespace(left_stick_tip_offset=[999]*3,right_stick_tip_offset=[999]*3)
    configure_tip_geometry(cfg)
    np.testing.assert_allclose(cfg.left_stick_tip_offset,stick_tip_offset('left'),atol=1e-12)
    np.testing.assert_allclose(cfg.right_stick_tip_offset,stick_tip_offset('right'),atol=1e-12)
    for hand in ('left','right'):
        probes=env_module.stick_shaft_offsets_in_wrist(hand,n_samples=4)
        for i,probe in enumerate(probes,1):
            sigma=g1.STICK_LENGTH/2-i*g1.SHAFT_HIT_SPAN_M/4
            np.testing.assert_allclose(probe,stick_contact_offset(hand,sigma),atol=1e-12)


def test_registered_contact_tracker_uses_local_capsule_rule():
    class Tracker:pass
    cfg_module=SimpleNamespace(get_g1_cfg=lambda:None)
    env_module=SimpleNamespace(stick_shaft_offsets_in_wrist=lambda *_args:None,
        ContactStrikeTracker=Tracker)
    bind_registered_task_geometry(cfg_module,env_module)
    assert env_module.ContactStrikeTracker is not Tracker
    assert env_module.ContactStrikeTracker._capsule_tracker


def test_editing_constants_invalidates_composed_usd_cache(tmp_path,monkeypatch):
    from amortized_opt import constants
    source=tmp_path/'stock.usd';output=tmp_path/'composed.usd';definitions=tmp_path/'constants.py'
    for path in (source,output,definitions):path.write_text('test')
    now=max(Path(g1.__file__).stat().st_mtime,os.path.getmtime(source))+10
    os.utime(output,(now,now));os.utime(definitions,(now+1,now+1))
    monkeypatch.setattr(constants,'__file__',str(definitions))
    monkeypatch.setattr(g1,'_composed_usd_path',lambda freeze:output)
    calls=[]
    monkeypatch.setattr(g1,'_compose_g1_with_sticks',lambda src,out,freeze_legs:calls.append((src,out,freeze_legs)))
    assert g1._ensure_composed_usd(source,freeze_legs=True)==output
    assert calls==[(source,output,True)]


def test_old_geometry_cached_labels_are_rejected(tmp_path):
    import json
    from scripts.train_neural_hand_plans import load_cached_panel
    (tmp_path/'config.json').write_text(json.dumps({'checkpoint_sha256':'test_decoder'}))
    with pytest.raises(ValueError,match='old/unverified simulator geometry'):
        load_cached_panel(tmp_path,'test_decoder')


def test_thick_cymbal_layout_and_usd_are_bound_together():
    cfg=SimpleNamespace(drum_layout_file='old',drum_kit_cfg=SimpleNamespace(
        spawn=SimpleNamespace(usd_path='old')))
    configure_drum_kit_geometry(cfg)
    result=audit_drum_kit(cfg.drum_kit_cfg.spawn.usd_path,cfg.drum_layout_file)
    assert {result[name]['height_m'] for name in ('hihat','cymbal_left','cymbal_right')}=={.1}
    assert result['snare']['height_m']==.16


def test_fixed_deadline_labels_are_rejected_for_random_deadline_training(tmp_path):
    import json
    from isaac_rl.decoder_scene import GEOMETRY_VERSION
    from scripts.train_neural_hand_plans import load_cached_panel
    (tmp_path/'config.json').write_text(json.dumps({'checkpoint_sha256':'test_decoder',
        'simulation_geometry_version':GEOMETRY_VERSION,'first_hit_policy':'fixed_one_second'}))
    with pytest.raises(ValueError,match='different first-hit timing policy'):
        load_cached_panel(tmp_path,'test_decoder','tier_random')
