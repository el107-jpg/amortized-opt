import math

import pytest
import torch

from isaac_rl.chart_cma_reward import (distance_to_hit_region, proximity_credit,
    NearMissTracker)


def test_distance_to_disk_and_half_disk_is_euclidean():
    points = torch.tensor([[0., 0., .03], [1.2, 0., 0.], [-.2, .5, 0.], [-1., 2., .3]])
    centers = torch.zeros_like(points)
    normals = torch.tensor([0., 0., 1.]).expand_as(points)
    radii = torch.ones(4)
    halves = torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    distance = distance_to_hit_region(points, centers, normals, radii, halves)
    torch.testing.assert_close(distance, torch.tensor([.03, .2, .2, math.sqrt(2.09)]))
    # Axis tilt changes the plane, without losing the Euclidean interpretation.
    assert distance_to_hit_region(torch.tensor([[.04, 0., 0.]]), centers[:1],
        torch.tensor([[1., 0., 0.]]), radii[:1], halves[:1]).item() == pytest.approx(.04)


def test_partial_credit_is_bounded_monotone_and_never_added_to_hits():
    distance = torch.tensor([0., .01, .05, .1, float('inf'), 0., 0.])
    credited = torch.tensor([False, False, False, False, False, True, False])
    valid = torch.tensor([True, True, True, True, True, True, False])
    partial, quality = proximity_credit(distance, credited, valid)
    assert partial[0] == .25
    assert partial[1] == pytest.approx(.25*math.exp(-.2))
    assert partial[0] > partial[1] > partial[2] > partial[3] > partial[4]
    assert partial[5] == 0 and quality[5] == 1
    assert partial[6] == 0 and quality[6] == 0
    assert quality.sum() <= valid.sum()


def tracker():
    # The first target belongs to the left hand, the second to the right.
    return NearMissTracker(torch.tensor([[1., 2.]]), torch.tensor([[0, 1]]),
        torch.tensor([[True, True]]), torch.zeros(1, 2, 3),
        torch.tensor([[[0., 0., 1.], [0., 0., 1.]]]), torch.ones(1, 2),
        torch.zeros(1, 2, 3))


def test_tracker_uses_only_assigned_hand_and_scheduled_window():
    t = tracker()
    # An early pass at the target must not credit the later scheduled hit.
    t.update(.5, torch.zeros(1, 2, 3), torch.tensor([True]))
    assert torch.isinf(t.closest).all()
    # The wrong hand at distance zero must not improve left-hand proximity.
    t.update(1., torch.tensor([[[0., 0., .1], [0., 0., 0.]]]), torch.tensor([True]))
    assert t.closest[0, 0] == pytest.approx(.1)
    assert torch.isinf(t.closest[0, 1])
    t.update(1.02, torch.tensor([[[0., 0., .02], [0., 0., 0.]]]), torch.tensor([True]))
    assert t.closest[0, 0] == pytest.approx(.02)
    # Finished episodes do not acquire credit from later simulation steps.
    t.update(2., torch.zeros(1, 2, 3), torch.tensor([False]))
    assert torch.isinf(t.closest[0, 1])
    partial, quality = t.credits(torch.tensor([[True, False]]))
    assert partial[0, 0] == 0 and quality[0, 0] == 1
    assert partial[0, 1] == quality[0, 1] == 0


def test_proximity_config_is_explicit_on_legacy_resume(tmp_path):
    import json
    from scripts.train_chart_cma import parse_args
    (tmp_path/'config.json').write_text(json.dumps({'args': {'sampling': 'independent',
        'near_miss_weight': 0., 'near_miss_scale_m': .05, 'near_miss_window_s': .08}}))
    base = ['--out', str(tmp_path), '--resume', str(tmp_path/'last.pkl')]
    assert parse_args(base).near_miss_weight == 0
    assert parse_args(base+['--near-miss-weight', '.25']).near_miss_weight == .25
    (tmp_path/'config.json').write_text(json.dumps({'args': {'sampling': 'independent'}}))
    assert parse_args(base).near_miss_weight == 0
    assert parse_args(['--out', str(tmp_path/'fresh')]).near_miss_weight == .25
    for flag, value in [('--near-miss-weight', '1'), ('--near-miss-scale-m', '0'),
                        ('--near-miss-window-s', '-1')]:
        with pytest.raises(SystemExit):
            parse_args(base+[flag, value])
