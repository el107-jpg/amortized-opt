"""Compare duration schedules without changing trajectory repair or its gates."""
from dataclasses import dataclass


@dataclass
class DurationSearchResult:
    horizon: int
    path: object
    deadlines: list


def search_duration(initial_horizon, initial_path, attempt, *, stride=1, refine_bands=2):
    """Call attempt(N, best_path, best_N, stage) -> (path, success).

    A stride of one exhausts the shorter 20 ms grid. Larger strides first
    exhaust the coarse grid, then refine below the shortest passing witness.
    The fine window extends whenever a shorter witness is found. Failures do
    not prune the coarse sweep. This is a heuristic for nonmonotone feasibility;
    an isolated fine-grid island far below the final window can still be missed.
    """
    if initial_horizon < 1 or stride < 1 or refine_bands < 1:
        raise ValueError('Positive horizon, stride and refinement bands required')
    best_N, best_path = initial_horizon, initial_path
    tested_with, trace = {}, []
    def visit(N, stage):
        nonlocal best_N, best_path
        if N >= best_N or tested_with.get(N) == best_N:
            return
        parent_N = best_N
        path, success = attempt(N, best_path, best_N, stage)
        tested_with[N] = parent_N
        trace.append(dict(segments=N, parent_segments=parent_N, stage=stage,
                          success=bool(success)))
        if success:
            best_N, best_path = N, path
    deadlines = list(range(initial_horizon-stride, 0, -stride))
    if initial_horizon > 1 and 1 not in deadlines:
        deadlines.append(1)
    for N in deadlines:
        visit(N, 'fine' if stride == 1 else 'coarse')
    if stride > 1:
        N = best_N-1
        while N >= max(1, best_N-refine_bands*stride):
            visit(N, 'refinement')
            N -= 1
    return DurationSearchResult(best_N, best_path, trace)
