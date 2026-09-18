"""Source-of-truth rhythm launcher safety/default contract."""

from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "_run_rhythm_bellman.sh"


def test_cpu_defaults_to_smaller_eager_actor_batches():
    text = SCRIPT.read_text()
    assert 'if [[ "$DEV" == "cpu" ]]; then\n    BATCH=128' in text
    assert 'if [[ "$LINE_SEARCH" == "1" || "$DEV" == "cpu" ]]; then' in text
    assert 'COMPILE_MODEL="${COMPILE_MODEL:-0}"' in text


def test_line_search_ablation_defaults_and_strict_opt_ins():
    text = SCRIPT.read_text()
    assert 'RESET_OPTIMIZER="${RESET_OPTIMIZER:-0}"' in text
    assert 'LOAD_FLAGS+=(--reset-optimizer)' in text
    assert "RESET_OPTIMIZER=1 requires RESUME=" in text
    assert "[run] resume controls:" in text
    defaults = {
        "LINE_SEARCH_STRICT_ACTUAL_UPDATE": "0",
        "LINE_SEARCH_REJECT_CURRENT_PROBE_INCREASE": "0",
        "LINE_SEARCH_SKIP_ON_PROBE_EXHAUSTION": "0",
        "LINE_SEARCH_FORCE_AL_RESET": "0",
        "LINE_SEARCH_ACTUAL_DIAGNOSTIC_EVERY": "0",
    }
    for name, value in defaults.items():
        assert f'{name}="${{{name}:-{value}}}"' in text

    flags = (
        "--line-search-strict-actual-update",
        "--line-search-reject-current-probe-increase",
        "--line-search-skip-on-probe-exhaustion",
        "--line-search-force-al-reset",
        "--line-search-actual-diagnostic-every",
    )
    for flag in flags:
        assert flag in text
    assert "[run] line-search safety:" in text
    assert "[run] line-search CLI:" in text


def test_direct_recipe_and_non_line_search_arm_remain_default_safe():
    text = SCRIPT.read_text()
    assert 'LINE_SEARCH="${LINE_SEARCH:-1}"' in text
    assert 'BPTT_START_FRAC="${BPTT_START_FRAC:-2}"' in text
    assert 'BPTT_START_FRAC="${BPTT_START_FRAC:-0.6667}"' in text
    assert 'WAIST_PR_SP="${WAIST_PR_SP:-0}"' in text
    assert 'WAIST_PR_EMA="${WAIST_PR_EMA:-0}"' in text
    assert 'W_VELOCITY="${W_VELOCITY:-0}"' in text
    assert 'W_VELOCITY_PEAK="${W_VELOCITY_PEAK:-0}"' in text
    assert 'REPLAY_VIOL_DOWNWEIGHT="${REPLAY_VIOL_DOWNWEIGHT:-0.05}"' in text
    assert 'REPLAY_VIOL_UNIFORM_MIX="${REPLAY_VIOL_UNIFORM_MIX:-0.05}"' in text
    assert '--replay-viol-uniform-mix "$REPLAY_VIOL_UNIFORM_MIX"' in text
    assert "replay_viol_sampling_beta=${REPLAY_VIOL_DOWNWEIGHT}" in text
    assert "replay_viol_uniform_mix=${REPLAY_VIOL_UNIFORM_MIX}" in text
    assert 'REPLAY_PROFICIENCY="${REPLAY_PROFICIENCY:-0}"' in text


def test_timing_critic_uses_replay_frozen_line_search_by_default():
    text = SCRIPT.read_text()
    start = text.index(
        'if [[ "$TIMING_CRITIC_MODE" == "assist" || '
        '"$TIMING_CRITIC_MODE" == "control" ]]'
    )
    end = text.index("# Learned task-time feasibility model", start)
    timing_branch = text[start:end]
    assist_branch = timing_branch.split("else", 1)[0]
    assert 'VOXEL_POOL="${VOXEL_POOL:-0}"' in assist_branch
    assert "LINE_SEARCH=" not in assist_branch
    assert 'LINE_SEARCH="${LINE_SEARCH:-1}"' in timing_branch
    assert 'TORQUE_GRAD_CAP="${TORQUE_GRAD_CAP:-0.7}"' in text
    assert 'if [[ -z "${GRAD_BALANCE+x}" ]]; then' in text
    assert 'GRAD_BALANCE=""' in text
