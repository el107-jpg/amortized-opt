"""Procedural one-second rhythm windows, independent of recorded performances."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from amortized_opt.rhythm import Hit, RhythmClip


@dataclass(frozen=True)
class RandomRhythmConfig:
    horizon_s: float = 1.
    max_hits: int = 10
    grid_s: float = .02
    tempo_min: float = 55.
    tempo_max: float = 220.
    pair_probability: float = .25


class RandomRhythmSampler:
    """Mix independent drums, repeating motifs and fills; no hand preferences.

    Every window is generated anew. Drum identities are exchangeable, so all
    drums receive equal expected coverage, including members of strike pairs.
    Times lie on the planner control grid. Short deadlines deliberately include
    difficult schedules; generation does not certify physical reachability.
    """

    def __init__(self, kit, config=RandomRhythmConfig()):
        if (kit.num_drums < 2 or config.max_hits < 1 or config.grid_s <= 0
                or config.horizon_s < 8*config.grid_s
                or not 0 < config.tempo_min <= config.tempo_max
                or not 0 <= config.pair_probability <= 1):
            raise ValueError('invalid procedural rhythm settings')
        self.kit, self.config = kit, config
        self.summary = dict(source='procedural, no recorded rhythm windows',
            **asdict(config), modes=['independent', 'repeating', 'fill'],
            expected_hit_shares=[1/kit.num_drums]*kit.num_drums,
            first_hit_offsets='Half 160 ms; half uniform 20..160 ms',
            hand_masks=False)

    def sample(self, rng, count):
        cfg, kit = self.config, self.kit
        clips = []
        for _ in range(count):
            mode = int(rng.choice(3, p=[.4, .3, .3]))
            tempo = float(rng.uniform(cfg.tempo_min, cfg.tempo_max))
            subdivision = int(rng.choice([1, 2, 4, 8], p=[.1, .35, .45, .1]))
            period = 60/tempo/subdivision
            swing = float(rng.uniform(.5, .7))
            first_tick = 8 if rng.random() < .5 else int(rng.integers(1, 9))
            tick = first_tick
            budget = int(rng.integers(1, cfg.max_hits+1))
            motif = rng.choice(kit.num_drums, size=int(rng.integers(2, kit.num_drums+1)), replace=False)
            previous = int(rng.integers(kit.num_drums))
            hits, event = [], 0
            while tick*cfg.grid_s <= cfg.horizon_s+1e-8 and len(hits) < budget:
                if mode == 0:
                    drum = int(rng.integers(kit.num_drums))
                elif mode == 1:
                    drum = previous if rng.random() < .65 else int(rng.integers(kit.num_drums))
                else:
                    drum = int(motif[event % len(motif)])
                pair = len(hits)+2 <= budget and rng.random() < cfg.pair_probability
                drums = [drum]
                if pair:
                    other = int(rng.integers(kit.num_drums-1))
                    drums.append(other+int(other >= drum))
                t = round(tick*cfg.grid_s, 8)
                for member, d in enumerate(drums):
                    velocity = float(rng.uniform(.2, 1.))
                    # Placeholder physical order, never a drum eligibility label.
                    hits.append(Hit(t, t, d, kit.names[d], ('left', 'right')[member], velocity))
                previous = drum
                if rng.random() < .2:
                    gap = float(np.exp(rng.uniform(np.log(.04), np.log(.4))))
                else:
                    gap = period*(2*swing if event % 2 == 0 else 2*(1-swing))
                    gap *= float(rng.choice([.5, 1., 2.], p=[.1, .8, .1]))
                gap_ticks = max(2, int(round(gap/cfg.grid_s)))
                if rng.random() < .05:
                    gap_ticks = 1
                tick += gap_ticks
                event += 1
            name = f'random_{mode}_{int(rng.integers(2**63)):016x}'
            clips.append(RhythmClip(name, hits, hits[-1].time))
        return clips
