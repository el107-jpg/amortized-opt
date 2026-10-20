"""Discrete deadline search without assuming monotonic decoder feasibility."""
import numpy as np


def coarse_horizons(stride=10, maximum=100):
    """50 Hz grid: reference first, fast descent, then longer fallback anchors."""
    if stride < 1 or maximum < 50:
        raise ValueError('Positive stride and a range covering one second required')
    return list(dict.fromkeys([50, *range(50-stride, 0, -stride), 1,
                              *range(50+stride, maximum+1, stride), maximum]))


def fine_horizons(coarse, successful):
    """Scan the interval below the lowest passing coarse anchor, in full.

    Every grid point in this interval is checked, rather than relying on
    binary search over a nonmonotonic neural acceptance function. A narrow
    disconnected feasible island between failed coarse anchors can be missed.
    """
    coarse = np.asarray(coarse, int)
    successful = np.asarray(successful, bool)
    if successful.shape != coarse.shape:
        raise ValueError('One outcome per coarse anchor required')
    if not successful.any():
        return []
    high = int(coarse[successful].min())
    low = int(coarse[coarse < high].max()) if (coarse < high).any() else 0
    return list(range(low+1, high))
