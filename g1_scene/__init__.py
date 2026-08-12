"""Isaac-side scene assets: the G1-with-sticks articulation and the drum kit.

This package is the standalone replacement for the three things
``isaac_rl``/``isaac_rl_ppo`` used to import out of the main repo:
``g1_drummer.assets.g1`` (the composed articulation cfg), the drum-kit /
layout path constants, and the drum-id ordering out of ``DrumLayout``.

Like ``isaac_rl*``, it depends on Isaac Lab and is therefore NOT importable
from :mod:`amortized_opt` -- the core stays pure torch/numpy. The dependency
runs one way: ``g1_scene`` imports geometry constants from ``amortized_opt``,
never the reverse.

``g1`` is imported lazily (``from g1_scene.g1 import get_g1_cfg``) so that
``g1_scene.layout`` -- pure path/YAML work -- stays usable headless, without
Isaac Lab or ``unitree_rl_lab`` installed.
"""

from __future__ import annotations

from .layout import DRUM_KIT_USD_PATH, DRUM_LAYOUT_PATH, ordered_drum_names

__all__ = ["DRUM_KIT_USD_PATH", "DRUM_LAYOUT_PATH", "ordered_drum_names"]
