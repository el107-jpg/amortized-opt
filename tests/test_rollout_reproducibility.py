from types import SimpleNamespace
from isaac_rl.rollout_reproducibility import freeze_initial_hit_scoring,scoring_state


def test_fixed_scoring_preserves_initial_kernel_and_blocks_anneal_updates():
    env=SimpleNamespace(cfg=SimpleNamespace(correct_time_sigma_start=.4,correct_time_sigma_end=.01,
        velocity_match_shape_start=1.,velocity_match_shape=6.,beat_done_quality=.25),
        curriculum_progress=0.,timing_progress=0.,speed_progress=0.,discovery_unlock_step=-1,approach_step_count=0)
    settings=freeze_initial_hit_scoring(env)
    assert settings['timing_sigma_s']==.4 and settings['velocity_match_shape']==1.
    env.timing_progress=.9;env.speed_progress=.8;env.approach_step_count=100000;env._step_curriculum()
    for fraction in [0.,.3,1.]:
        cfg=env.cfg
        assert cfg.correct_time_sigma_start+(cfg.correct_time_sigma_end-cfg.correct_time_sigma_start)*fraction==.4
        assert cfg.velocity_match_shape_start+(cfg.velocity_match_shape-cfg.velocity_match_shape_start)*fraction==1.
    assert scoring_state(env)['done_quality']==.25


def test_runtime_reproducibility_flags_preserve_resume_recipe_and_allow_explicit_override(tmp_path):
    import json
    from scripts.train_chart_cma import parse_args
    (tmp_path/'config.json').write_text(json.dumps({'args':dict(freeze_hit_scoring=True,decoder_fp32=True,rollout_audit=False)}))
    flags=['--out',str(tmp_path),'--resume',str(tmp_path/'last.pkl')]
    assert parse_args(flags).freeze_hit_scoring and parse_args(flags).decoder_fp32
    assert not parse_args(flags+['--no-decoder-fp32']).decoder_fp32
    assert parse_args(flags+['--rollout-audit']).rollout_audit
    assert not parse_args(flags).clear_reset_contacts  # Legacy resumes preserve their archived runtime.
    assert parse_args(['--out',str(tmp_path)]).clear_reset_contacts
    assert not parse_args(['--out',str(tmp_path),'--no-clear-reset-contacts']).clear_reset_contacts
    assert parse_args(['--out',str(tmp_path)]).freeze_hit_scoring
    assert parse_args(['--out',str(tmp_path)]).match_decoder_speed
    assert not parse_args(flags).match_decoder_speed
    assert parse_args(flags+['--match-decoder-speed']).match_decoder_speed


def test_corrected_and_legacy_cached_outcomes_cannot_be_combined():
    import pytest
    from isaac_rl.hand_plan_dataset import require_matching_rollout_runtime
    require_matching_rollout_runtime({}, {})
    runtime=dict(rollout_runtime_version='cleared_contact_reset_v1',fixed_hit_scoring={'timing_sigma_s':.4},decoder_amp='torch.bfloat16')
    require_matching_rollout_runtime({'rollout_runtime':runtime},{'rollout_runtime':dict(runtime)})
    with pytest.raises(ValueError,match='regenerate labels'):
        require_matching_rollout_runtime({}, {'rollout_runtime':runtime})


def test_reset_force_clear_discards_old_reports_and_preserves_other_sensor_state():
    import torch
    from isaac_rl.rollout_reproducibility import clear_reset_contact_buffers
    data=SimpleNamespace(force_matrix_w=torch.tensor([[[3.,4.,0.]]]),net_forces_w=torch.ones(1,3),
        net_forces_w_history=torch.ones(1,2,3),other=torch.ones(1))
    env=SimpleNamespace(contact_sensors={'left':SimpleNamespace(data=data)})
    assert clear_reset_contact_buffers(env)==5.
    assert data.force_matrix_w.count_nonzero()==0 and data.net_forces_w.count_nonzero()==0
    assert data.net_forces_w_history.count_nonzero()==0 and data.other.item()==1.
