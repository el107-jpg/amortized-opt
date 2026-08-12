"""Scene asset paths and drum-id ordering for the Isaac side of the folder.

The main repo kept these in ``g1_drummer.tasks.direct.g1_drum`` -- two path
constants in ``g1_drummer_env_cfg`` and a 105-line ``DrumLayout`` dataclass in
``rhythm``. Only two things were ever read out of ``DrumLayout`` by the env
(``names`` and ``ids``, to order the contact-sensor filters), and everything
else it parsed -- positions, radii, normals, heights, shaft/edge flags -- is
already parsed by :class:`amortized_opt.rhythm.DrumKit` off the same YAML.

So rather than carry a second parser, this module keeps the one thing DrumKit
does not do: assert that the drum ids form a contiguous ``0..D-1`` block and
hand back the id-ordered names. That invariant is load-bearing -- the env emits
one contact-sensor filter per drum in this order and then indexes the result by
drum id, so a gap in the ids would silently misattribute every hit.
"""

from __future__ import annotations

from pathlib import Path

from amortized_opt.constants import ASSETS_DIR, default_layout_path

__all__ = [
    "DRUM_KIT_USD_PATH",
    "DRUM_LAYOUT_PATH",
    "ordered_drum_names",
]

#: The 6-drum kit USD, vendored from the repo's ``assets/drums/``. It is a
#: 4 KB procedural stage (cylinders + a ground plane), not a captured mesh.
DRUM_KIT_USD_PATH: Path = ASSETS_DIR / "drums" / "simple_drum_kit_6drums.usd"

#: The drum layout YAML. Byte-identical to the repo's
#: ``configs/assets/drum_kit_layout.yaml``; :mod:`amortized_opt` reads the same
#: file through ``constants.default_layout_path()``.
DRUM_LAYOUT_PATH: Path = default_layout_path()


def ordered_drum_names(layout_path: str | Path | None = None) -> tuple[str, ...]:
    """Drum names ordered by drum id, with the ids checked to be ``0..D-1``.

    Raises ``RuntimeError`` if the ids are not a contiguous block starting at 0,
    because callers rely on ``filter index == drum id``.
    """
    import yaml

    path = Path(layout_path) if layout_path is not None else DRUM_LAYOUT_PATH
    drums = yaml.safe_load(Path(path).read_text())["drums"]
    ids = {name: int(spec["id"]) for name, spec in drums.items()}
    ordered = sorted(ids, key=lambda n: ids[n])
    if [ids[n] for n in ordered] != list(range(len(ordered))):
        raise RuntimeError(
            f"drum ids in {path} must be 0..D-1 for filter index == drum id, got "
            f"{sorted(ids.values())}"
        )
    return tuple(ordered)
