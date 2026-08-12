"""The folder must run with UNITREE_MODEL_DIR unset and no repo on sys.path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # amortized/

SCRIPT = """
import sys, torch, numpy as np
sys.path.insert(0, ".")
import amortized_opt as A

bad = [m for m in sys.modules if m.startswith("g1_drummer")]
assert not bad, f"repo package leaked into the standalone run: {bad}"

fk = A.ChainFK(dtype=torch.float64)
rnea = A.BatchedRNEA.from_urdf(dtype=torch.float64)
sdf = A.SdfPenalty(dtype=torch.float64)

q = torch.zeros(8, 17, dtype=torch.float64, requires_grad=True)
frames = fk.forward(q)
assert set(frames) == {"left_wrist", "right_wrist", "torso", "left_elbow", "right_elbow"}
loss = sdf.forward(frames=frames).sum() + rnea.rnea(q, q, q).square().sum()
loss = loss + sum(p.square().sum() for _R, p in frames.values())
loss.backward()
assert torch.isfinite(q.grad).all()

Q = np.linspace(0.0, 1.0, 5)[:, None] * np.ones((1, 17))
X, rate = A.retime_joint_path(Q, 25, A.CONTROL_DT)
assert X.shape == (25, 17) and np.isfinite(rate)

assert A.CONTROL_DT == 0.02 and abs(A.PD_DT - 1/250) < 1e-15
# PD_SUBSTEPS is an integration-accuracy knob, not the rate ratio: it was raised
# 5 -> 10 on 2026-08-12 because the explicit substep recursion in simulate_pd
# goes unstable at Kd 4.0 (see constants.py). It may only ever be raised above
# the 250 Hz / 50 Hz ratio, never dropped below it.
assert A.PD_SUBSTEPS >= round(A.CONTROL_DT / A.PD_DT), A.PD_SUBSTEPS
assert A.default_urdf_path().is_file()
print("STANDALONE_OK")
"""


def test_runs_without_unitree_model_dir_or_repo(capsys):
    env = {k: v for k, v in os.environ.items() if k != "UNITREE_MODEL_DIR"}
    env["PYTHONPATH"] = ""  # nothing from the repo on the path
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    with capsys.disabled():
        print(f"\n[standalone] rc={out.returncode} stdout={out.stdout.strip()!r}")
        if out.returncode:
            print(out.stderr[-3000:])
    assert out.returncode == 0
    assert "STANDALONE_OK" in out.stdout


def test_no_repo_imports_in_package():
    """Static check: no import statement in the core reaches the repo OR Isaac."""
    import re

    banned = re.compile(
        r"^\s*(?:from|import)\s+(g1_drummer|isaaclab|omni|trajopt|safety|control)\b", re.M
    )
    offenders = []
    for path in sorted((ROOT / "amortized_opt").rglob("*.py")):
        for m in banned.finditer(path.read_text()):
            offenders.append((path.name, m.group(0).strip()))
    assert not offenders, offenders


#: The only files under ``amortized/`` allowed to import the original repo.
#: Both are benchmarking/playback tools rather than parts of the system, and
#: both degrade with a message instead of a traceback when the repo is absent.
#: See CLAUDE.md. Adding to this list should be a deliberate decision.
REPO_ONLY_FILES = {
    "scripts/repo_adapters/trajopt_hpipm_planner.py",  # the trajopt SQP baseline
    "scripts/play_isaac.py",  # drives the repo's own G1DrummerEnv
}


def test_repo_imports_are_confined_to_the_known_exceptions():
    """Nothing outside :data:`REPO_ONLY_FILES` may import ``g1_drummer``.

    The Isaac layer (``g1_scene``, ``isaac_rl``, ``isaac_rl_ppo``) is allowed to
    import Isaac Lab -- that is its job -- but not the repo. ``tests/`` is
    excluded: several conformance tests deliberately import repo originals to
    diff against, and every one of them skips when the repo is missing.
    """
    import re

    banned = re.compile(r"^\s*(?:from|import)\s+g1_drummer\b", re.M)
    # Explicit roots rather than ROOT.rglob: runs/ is tens of GB of checkpoints
    # and walking it to find zero .py files is pure cost.
    searched = ("amortized_opt", "g1_scene", "isaac_rl", "isaac_rl_ppo", "scripts")
    offenders = []
    for top in searched:
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if "/tests/" in rel or rel in REPO_ONLY_FILES:
                continue
            for m in banned.finditer(path.read_text()):
                offenders.append((rel, m.group(0).strip()))
    assert not offenders, offenders


def test_the_known_exceptions_still_exist():
    """Guard against the list above silently going stale after a rename."""
    missing = [f for f in REPO_ONLY_FILES if not (ROOT / f).is_file()]
    assert not missing, missing
