"""Explicit fixed hit-credit settings and read-only rollout diagnostics."""


def scoring_state(env):
    cfg=env.cfg
    return dict(timing_sigma_s=float(getattr(env,'last_correct_time_sigma',cfg.correct_time_sigma_start)),
        velocity_match_shape=float(getattr(env,'last_velocity_match_shape',cfg.velocity_match_shape_start)),
        done_quality=float(cfg.beat_done_quality),
        curriculum_progress=float(env.curriculum_progress),timing_progress=float(env.timing_progress),
        speed_progress=float(env.speed_progress),discovery_unlock_step=int(env.discovery_unlock_step),
        approach_step_count=int(env.approach_step_count))


def freeze_initial_hit_scoring(env):
    """Preserve fresh-scene completion criteria independently of PPO curriculum.

    Disabling adaptive_curriculum alone would enable the legacy clock anneal.
    Equal start/end endpoints explicitly keep the timing/speed kernels fixed.
    Only this evaluator's env instance is changed; external PPO tasks are not.
    """
    cfg=env.cfg
    sigma=float(cfg.correct_time_sigma_start)
    shape=float(cfg.velocity_match_shape_start or cfg.velocity_match_shape)
    cfg.correct_time_sigma_start=cfg.correct_time_sigma_end=sigma
    cfg.velocity_match_shape_start=cfg.velocity_match_shape=shape
    env._step_curriculum=lambda:None
    env._fixed_hit_scoring=dict(version='fixed_initial_contact_quality_v1',timing_sigma_s=sigma,
        velocity_match_shape=shape,done_quality=float(cfg.beat_done_quality))
    return env._fixed_hit_scoring


def clear_reset_contact_buffers(env):
    """A teleport has no freshly solved contact force; discard pre-reset reports.

    Reading data primes the lazy sensor cache. The next real physics substep
    marks it stale again and provides forces for the installed starting pose.
    """
    peaks=[]
    for sensor in env.contact_sensors.values():
        data=sensor.data
        if data.force_matrix_w is not None:
            peaks.append(float(data.force_matrix_w.norm(dim=-1).max()))
            data.force_matrix_w.zero_()
        for key in ('net_forces_w','net_forces_w_history'):
            values=getattr(data,key,None)
            if values is not None:values.zero_()
    return max(peaks,default=0.)
