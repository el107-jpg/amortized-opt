"""Isaac-dependent test suite.  NOT collected by ``amortized/pytest.ini``.

``pytest.ini`` sets ``testpaths = tests``, so ``python -m pytest tests -q`` --
the command `REPORT.md` §9 publishes -- never sees this directory.  That is
deliberate: every test here needs a running Isaac Sim app, which costs ~3
minutes of startup and a GPU, and cannot be a precondition for the standalone
package's suite.

Run it explicitly::

    python -m pytest isaac_rl/tests -q -m isaac --isaac-device cuda:1

A single simulation app is started once per session and shared, because Isaac
Sim cannot be started twice in one process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--isaac-device", default="cuda:1",
                     help="GPUs 0 and 2 are held by concurrent sessions on this box")
    parser.addoption("--isaac-envs", type=int, default=8)


def pytest_configure(config):
    config.addinivalue_line("markers", "isaac: needs a running Isaac Sim app")


@pytest.fixture(scope="session")
def isaac_app(request):
    """Start the sim app once.  Everything Isaac must be imported AFTER this."""
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"isaaclab not importable: {exc}")
    dev = request.config.getoption("--isaac-device")
    app = AppLauncher(headless=True, device=dev).app
    yield app
    # The carb tasking mutex throws "std::system_error: Invalid argument" at
    # shutdown.  It is Isaac's, not ours, and must not fail a green suite.
    try:
        app.close()
    except Exception:
        pass


@pytest.fixture(scope="session")
def isaac_env(isaac_app, request):
    from isaac_rl.env import make_env
    from isaac_rl.env_cfg import DrumRLEnvCfg

    dev = request.config.getoption("--isaac-device")
    n = int(request.config.getoption("--isaac-envs"))
    env = make_env(DrumRLEnvCfg(num_envs=n, device=dev, record=True,
                                episode_length_s=30.0))
    env.reset()
    return env
