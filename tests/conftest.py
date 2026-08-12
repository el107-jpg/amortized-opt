"""Shared fixtures. Nothing here is required for the package to run."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # amortized/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pick_cuda_device() -> str | None:
    """GPUs 0 and 2 are held by concurrent sessions on this box -- use 1 or 3.

    Override with ``AMORTIZED_TEST_DEVICE`` (e.g. ``cuda:3`` or ``cpu``).
    """
    env = os.environ.get("AMORTIZED_TEST_DEVICE")
    if env:
        return env
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    for idx in (1, 3, 0):
        if idx < n:
            return f"cuda:{idx}"
    return "cuda:0"


@pytest.fixture(scope="session")
def cuda_device() -> str:
    dev = pick_cuda_device()
    if dev is None or not dev.startswith("cuda"):
        pytest.skip("no CUDA device available")
    # MANDATORY: without this, inductor CUDA-graph capture silently no-ops off
    # cuda:0 ("The CUDA Graph is empty") and the whole speedup disappears.
    torch.cuda.set_device(torch.device(dev).index or 0)
    return dev


# --------------------------------------------------------------- repo bridge --
#: Only conformance tests use this. The package itself never imports the repo.
REPO_CONTROL = _ROOT.parent / "source" / "g1_drummer" / "g1_drummer" / "control"


def repo_available() -> bool:
    return REPO_CONTROL.is_dir()


def import_repo(module: str):
    """Import a module from the in-repo ``control`` package (tests only)."""
    if not repo_available():
        pytest.skip(f"repo originals not present at {REPO_CONTROL}")
    p = str(REPO_CONTROL)
    if p not in sys.path:
        sys.path.insert(0, p)
    import importlib

    try:
        return importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import repo module {module!r}: {exc}")
