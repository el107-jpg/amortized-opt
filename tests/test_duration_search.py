from isaac_rl.duration_search import search_duration


def test_coarse_refinement_matches_full_grid_on_monotone_problem_with_fewer_repairs():
    def attempt(N, path, parent, stage):
        return N, N >= 12
    fine = search_duration(40, 40, attempt)
    coarse = search_duration(40, 40, attempt, stride=3)
    assert fine.horizon == coarse.horizon == 12
    assert len(coarse.deadlines) < len(fine.deadlines)
    assert any(t['stage'] == 'refinement' for t in coarse.deadlines)


def test_failed_coarse_deadline_does_not_prune_lower_island_and_refinement_extends():
    accepted = {28, 16, 14, 12}
    def attempt(N, path, parent, stage):
        return N, N in accepted
    fine = search_duration(28, 28, attempt)
    coarse = search_duration(28, 28, attempt, stride=3)
    assert fine.horizon == coarse.horizon == 12
    assert any(t['segments'] == 25 and not t['success'] for t in coarse.deadlines)
    assert any(t['segments'] == 16 and t['success'] for t in coarse.deadlines)
    assert any(t['segments'] == 12 and t['success'] for t in coarse.deadlines)


def test_coarse_search_exposes_its_finite_window_limit_instead_of_assuming_monotonicity():
    def attempt(N, path, parent, stage):
        return N, N == 12
    assert search_duration(28, 28, attempt).horizon == 12
    # The only shorter witness lies between coarse failures, outside the fine
    # window. A failed coarse search cannot certify a physical minimum.
    assert search_duration(28, 28, attempt, stride=3).horizon == 28
