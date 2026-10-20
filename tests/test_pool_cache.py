"""The ``--pool-cache`` schema + content fingerprint (``scripts/train_phase2.py``).

An old cache used to reload without complaint and silently zero every context
field it predates -- ``aim_valid=False`` / ``stick_valid=False`` on every row.
Under ``--waist-sp-head`` that turns the learned anchor into "pin yaw to 0"
while the ``sp_res`` log reads a healthy 0.0.  The npz now carries the
``CONTEXT_FIELDS`` names as a marker, and a missing or different marker is a
hard failure with instructions, not a quiet degradation.

Kit DR / mix / seed / lead were a second silent failure class: the schema
marker is name-only, so a path reused with different ``--drum-tilt-deg`` etc.
trained on the wrong distribution.  ``_fingerprint`` closes that.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import train_phase2 as tp  # noqa: E402  (the SCRIPT, not the train loop)
from amortized_opt.phase2_data import Phase2Noise, Phase2ScenarioConfig  # noqa: E402
from amortized_opt.rhythm import KitRandomization  # noqa: E402
from amortized_opt.warmstart_data import CONTEXT_FIELDS  # noqa: E402
from amortized_opt.waist_aim import WaistAimConfig  # noqa: E402


def test_saved_line_search_can_be_disabled_for_targeted_teacher_training(tmp_path):
    saved = tmp_path/'args.json'
    saved.write_text(json.dumps({'line_search': True}))
    args = tp.parse_args(['--args-json', str(saved), '--no-line-search',
                          '--dls-terminal-goal-pairs', 'right:hihat'])
    assert args.line_search is False
    assert args.dls_terminal_goal_pairs == ['right:hihat']


def _args(**kw):
    base = dict(
        scenarios=100,
        seed=0,
        legacy_pool=False,
        rhythm_idle_frac=0.3,
        post_hit_frac=0.6,
        far_fetched_frac=0.1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _scen(**kw):
    base = dict(
        horizon=25,
        buckets=(25,),
        num_future=1,
        rhythm_idle_frac=0.3,
        post_hit_frac=0.6,
        far_fetched_frac=0.1,
        eval_like_frac=0.0,
        cold_start_frac=0.0,
        pair_frac=0.0,
        rest_frac=0.0,
        lead_s=1.0,
        synth_transition_frac=0.25,
        tip_speed_max=4.0,
        hit_time_jitter_s=0.05,
        hit_loudness_jitter=0.2,
        rare_drum_boost=0.25,
        kit_randomization=KitRandomization(
            per_drum_tilt=math.radians(3.0),
            per_drum_radius=0.15,
        ),
        waist_cfg=WaistAimConfig(future_tau=1.0, future_max=1),
        idle_cylinder_path="default",
        idle_settle_wp=5,
    )
    base.update(kw)
    return Phase2ScenarioConfig(**base)


def _try_load(tmp_path, fingerprint=None, **files):
    cache = tmp_path / "pool.npz"
    np.savez(cache, **files)
    return tp._load_npz_buckets(cache, fingerprint=fingerprint)


def test_the_marker_is_the_context_fields():
    assert json.loads(tp._pool_schema()) == list(CONTEXT_FIELDS)


def test_fingerprint_is_stable_for_the_same_settings():
    args, scen = _args(), _scen()
    assert tp._pool_fingerprint(args, scen) == tp._pool_fingerprint(args, scen)


def test_fingerprint_moves_when_tilt_changes():
    args = _args()
    a = tp._pool_fingerprint(args, _scen())
    b = tp._pool_fingerprint(
        args,
        _scen(kit_randomization=KitRandomization(
            per_drum_tilt=math.radians(5.0), per_drum_radius=0.15,
        )),
    )
    assert a != b


def test_cache_identity_includes_all_hand_goal_coverage_and_old_mapping_is_explicit():
    assert tp._pool_fingerprint(_args(),_scen(all_hand_drum_goals=True)) != tp._pool_fingerprint(_args(),_scen(all_hand_drum_goals=False))
    assert tp.parse_args([]).all_hand_drum_goals
    assert not tp.parse_args(['--no-all-hand-drum-goals']).all_hand_drum_goals


def test_fresh_selector_resume_rate_override_is_explicit_and_positive():
    assert tp.parse_args(['--line-search-legacy-resume-seed','3e-5']).line_search_legacy_resume_seed==3e-5
    with pytest.raises(SystemExit):tp.parse_args(['--line-search-legacy-resume-seed','0'])


def test_startup_coverage_checks_actual_active_targets_and_single_hand_lookahead():
    import torch
    from amortized_opt.rhythm import DrumKit
    kit=DrumKit.from_layout();pairs=[(h,d) for h,hand in enumerate(('left','right')) for d in kit.eligible(hand)]
    drums=torch.full((len(pairs),2),-1,dtype=torch.long);active=torch.zeros_like(drums,dtype=torch.bool)
    for row,(h,d) in enumerate(pairs):drums[row,h]=d;active[row,h]=True
    class Pool:
        data={50:dict(drum=drums,active=active)}
        def __len__(self):return len(drums)
    report=tp.training_goal_coverage(Pool(),Phase2ScenarioConfig())
    assert not report['missing_active_targets']
    assert len(report['expected_hand_drum_pairs'])==13
    assert report['synthetic_source_destination_pairs']==85
    active[pairs.index((1,kit.index('hihat'))),1]=False
    with pytest.raises(RuntimeError,match='right:hihat'):
        tp.training_goal_coverage(Pool(),Phase2ScenarioConfig())


def test_a_cache_without_the_marker_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="no schema marker"):
        _try_load(tmp_path, **{"25/q0": np.zeros((2, 17), np.float32)})


def test_a_cache_with_a_stale_marker_is_refused(tmp_path):
    stale = json.dumps([f for f in CONTEXT_FIELDS if not f.startswith("aim")])
    with pytest.raises(RuntimeError, match="different context schema"):
        _try_load(tmp_path, _schema=np.array(stale),
                  **{"25/q0": np.zeros((2, 17), np.float32)})


def test_a_cache_without_fingerprint_is_refused_when_required(tmp_path):
    fp = tp._pool_fingerprint(_args(), _scen())
    with pytest.raises(RuntimeError, match="no content fingerprint"):
        _try_load(
            tmp_path,
            fingerprint=fp,
            _schema=np.array(tp._pool_schema()),
            **{"25/q0": np.zeros((2, 17), np.float32)},
        )


def test_a_cache_with_wrong_fingerprint_is_refused(tmp_path):
    args, scen = _args(), _scen()
    fp = tp._pool_fingerprint(args, scen)
    other = tp._pool_fingerprint(_args(seed=99), scen)
    with pytest.raises(RuntimeError, match="different pool settings"):
        _try_load(
            tmp_path,
            fingerprint=fp,
            _schema=np.array(tp._pool_schema()),
            _fingerprint=np.array(other),
            **{"25/q0": np.zeros((2, 17), np.float32)},
        )


def test_a_current_cache_still_loads(tmp_path):
    args, scen = _args(), _scen()
    fp = tp._pool_fingerprint(args, scen)
    cache = tmp_path / "pool.npz"
    np.savez(
        cache,
        _schema=np.array(tp._pool_schema()),
        _fingerprint=np.array(fp),
        _stats=np.array(json.dumps({"accepted": 2})),
        **{"25/q0": np.zeros((2, 17), np.float32)},
    )
    pool = tp.load_or_make_pool(
        SimpleNamespace(pool_cache=str(cache), post_hit_cache=None,
                        legacy_pool=False, rhythm_idle_frac=0.3,
                        far_fetched_frac=0.1, scenarios=100, seed=0),
        scen, Phase2Noise(), "cpu",
    )
    assert pool.size(25) == 2
    assert pool.stats.get("accepted") == 2
