"""Accepted phase-1 coverage gates and fresh-head checkpoint behavior."""

from dataclasses import asdict
import pytest
import torch

from amortized_opt.model import AmortizedOptimizer, ModelConfig, model_config_from_checkpoint
from amortized_opt.rhythm import DrumKit
from amortized_opt.warmstart import GoalPoseLibrary
from amortized_opt.warmstart_data import ScenarioConfig, phase1_goal_coverage, phase1_synthetic_goal_clips


def test_phase1_clips_include_new_goals_and_all_directional_pairs():
    kit = DrumKit.from_layout()
    clips = phase1_synthetic_goal_clips(kit, GoalPoseLibrary(), ScenarioConfig())
    pairs = {(hit.hand, hit.drum) for clip in clips for hit in clip.hits}
    expected = {(hand, d) for hand in ("left", "right") for d in kit.eligible(hand)}
    assert pairs == expected
    assert len(clips) == 255
    assert len(expected) == 13


def test_coverage_counts_only_accepted_active_targets():
    kit = DrumKit.from_layout()
    rows = [(h, d) for h, hand in enumerate(("left", "right")) for d in kit.eligible(hand)]
    drum = torch.full((len(rows), 2), -1, dtype=torch.long)
    active = torch.zeros_like(drum, dtype=torch.bool)
    for i, (h, d) in enumerate(rows):
        drum[i, h] = d
        active[i, h] = True

    class Pool:
        data = {50: {"drum": drum, "active": active}}
        def __len__(self):
            return len(drum)

    coverage = phase1_goal_coverage(Pool(), ScenarioConfig())
    assert not coverage["missing_targets"]
    assert coverage["active_goal_counts"]["right"]["hihat"] == 1
    assert coverage["active_goal_counts"]["left"]["cymbal_right"] == 1
    active[rows.index((1, kit.names.index("hihat"))), 1] = False
    with pytest.raises(RuntimeError, match="right:hihat"):
        phase1_goal_coverage(Pool(), ScenarioConfig())


def test_new_terminal_head_disables_legacy_hihat_override_and_saves_policy(tmp_path):
    cfg = ModelConfig(terminal_head=True, d_ctx=32, ctx_heads=4,
                      d_model=32, dec_heads=4, dec_layers=1)
    model = AmortizedOptimizer(cfg)
    path = tmp_path / "terminal.pt"
    torch.save({"model": model.terminal_initializer.head.state_dict(),
                "all_hand_drum_goals": True}, path)
    model.load_terminal_head(path)
    assert not model.terminal_initializer.hihat_library_fallback
    saved = {"model_cfg": asdict(cfg), "model": model.state_dict()}
    restored = AmortizedOptimizer(model_config_from_checkpoint(saved))
    assert not restored.terminal_initializer.hihat_library_fallback


def test_apex_training_defaults_to_clean_lines():
    from scripts.train_phase1_apex import parse_args
    args = parse_args([])
    assert args.init_mode == "straight_clean"
    assert args.all_hand_drum_goals
