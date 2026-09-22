"""Contract tests for the isolated, browser-based strike-pose editor."""

from __future__ import annotations

import hashlib
import asyncio
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("uvicorn")
import httpx

from amortized_opt.constants import G1_UPPER_BODY_JOINTS
from amortized_opt.fk import ChainFK, stick_contact_offset
from tools.strike_pose_editor.server import TOKEN_HEADER, create_app


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "tools" / "strike_pose_editor" / "static"
CANONICAL_POSES = ROOT / "assets" / "drum_goal_poses.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ASGIClient:
    """Tiny synchronous facade over HTTPX's async, in-process ASGI transport.

    Starlette's blocking ``TestClient`` can deadlock when Torch has already
    initialized its CPU worker pool.  The async transport needs no blocking
    portal thread and exercises the exact same ASGI routes.
    """

    def __init__(self, app):
        self.app = app

    def request(self, method: str, path: str, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request("DELETE", path, **kwargs)


def _client(tmp_path: Path, *, session: str = "test_session", token: str = "secret"):
    app = create_app(
        session,
        runs_root=tmp_path / "runs",
        token=token,
        static_dir=STATIC_DIR,
    )
    return app, _ASGIClient(app)


def _seed(config: dict, *, hand: str | None = None) -> dict:
    return next(row for row in config["seeds"] if hand is None or row["hand"] == hand)


def _evaluate_payload(seed: dict) -> dict:
    return {
        "q": seed["q"],
        "hand": seed["hand"],
        "drum": seed["drum"],
        "contact_sigma": seed["contact_sigma"],
    }


def _assert_finite_tree(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_tree(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        assert math.isfinite(float(value)), value


def test_clean_app_construction_is_cpu_only_and_config_is_canonical(tmp_path, monkeypatch):
    cuda_was_initialized = torch.cuda.is_initialized()

    def forbidden_cuda_init():
        raise AssertionError("strike-pose editor attempted to initialize CUDA")

    def forbidden_process(*_args, **_kwargs):
        raise AssertionError("constructing the editor must not launch a process")

    monkeypatch.setattr(torch.cuda, "init", forbidden_cuda_init)
    monkeypatch.setattr(subprocess, "Popen", forbidden_process)
    app, client = _client(tmp_path)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"ok": True, "session": "test_session"}
    cfg = client.get("/api/config").json()
    assert cfg["schema_version"] == 1
    assert cfg["session"] == "test_session"
    assert cfg["mutation_token"] == "secret"
    assert tuple(row["name"] for row in cfg["joints"]) == G1_UPPER_BODY_JOINTS
    assert len(cfg["joints"]) == 17
    assert cfg["seeds"] and cfg["drums"] and cfg["yaw_bins_rad"]
    assert set(cfg["hands"]) == {"left", "right"}
    assert cfg["sampling_mode"] == "random_il_teaching"
    assert torch.get_num_threads() == 1
    assert torch.cuda.is_initialized() == cuda_was_initialized
    assert app.state.store.root == (tmp_path / "runs" / "test_session").resolve()

    page = client.get("/")
    assert page.status_code == 200
    assert "Trajectory Teaching Studio" in page.text
    assert "Sample IL teaching task" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "application/javascript" in script.headers["content-type"]
    assert "drawBodyReference" in script.text
    assert "ROBOT_GROUND_Z = -0.80" in script.text
    assert 'id="show-left-hand"' in page.text
    assert 'id="show-right-hand"' in page.text
    assert 'id="ik-contact-x"' in page.text
    assert 'id="ik-azimuth"' in page.text
    assert 'id="ik-waist-yaw"' in page.text
    assert 'id="ik-solve"' in page.text
    assert "dashed stick + bullseye show the unsolved IK target" in page.text
    assert "/task-ik" in script.text
    assert "function drawTaskTarget" in script.text
    assert "IK target · az" in script.text
    assert "function stashHandDraft" in script.text
    assert "function restoreHandDraft" in script.text
    assert "Save selected-hand teacher" in page.text
    assert "HANDS.filter(handVisible)" in script.text
    assert "previewInFlight" in script.text
    assert "if (state.previewTimer || state.previewInFlight) return" in script.text
    assert "if (state.previewDirty) schedulePosePreview()" in script.text
    assert "torso/legs are a planted visual reference" in page.text


def test_evaluate_validates_pose_and_matches_core_fk(tmp_path):
    app, client = _client(tmp_path)
    cfg = client.get("/api/config").json()
    seed = _seed(cfg)
    payload = _evaluate_payload(seed)
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["valid"] is True
    assert result["q"] == pytest.approx(seed["q"])
    _assert_finite_tree(result)

    # Reconstruct the selected stick point directly with the core FK/helper.
    wrist = f"{seed['hand']}_wrist_yaw_joint"
    fk = ChainFK(
        joint_names=G1_UPPER_BODY_JOINTS,
        frames={wrist: wrist},
        device="cpu",
        dtype=torch.float64,
    )
    q = torch.tensor(seed["q"], dtype=torch.float64).view(1, -1)
    rotation, origin = fk.frame(q, wrist)
    local = torch.tensor(
        stick_contact_offset(seed["hand"], seed["contact_sigma"]),
        dtype=torch.float64,
    ).view(1, 3)
    expected = ChainFK.points_world(rotation, origin, local)[0, 0].numpy()
    actual = np.asarray(result["geometry"]["sticks"][seed["hand"]]["contact"])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)

    bad_length = dict(payload, q=payload["q"][:-1])
    assert client.post("/api/evaluate", json=bad_length).status_code == 422

    out_of_bounds = list(payload["q"])
    out_of_bounds[0] = cfg["joints"][0]["max"] + 0.1
    assert client.post(
        "/api/evaluate", json=dict(payload, q=out_of_bounds)
    ).status_code == 422

    # Send raw JSON because HTTPX intentionally refuses to serialize NaN.
    raw = json.dumps(payload).replace(str(payload["q"][0]), "NaN", 1)
    assert client.post(
        "/api/evaluate", content=raw, headers={"content-type": "application/json"}
    ).status_code == 422
    assert client.post(
        "/api/evaluate", json={**payload, "filesystem_path": "/tmp/not-allowed"}
    ).status_code == 422


def test_every_builtin_seed_has_finite_geometry(tmp_path):
    _app, client = _client(tmp_path)
    cfg = client.get("/api/config").json()
    assert len(cfg["seeds"]) >= 1
    for seed in cfg["seeds"]:
        response = client.post("/api/evaluate", json=_evaluate_payload(seed))
        assert response.status_code == 200, (seed["hand"], seed["drum"], response.text)
        result = response.json()
        _assert_finite_tree(result["geometry"])
        _assert_finite_tree(result["metrics"])
        _assert_finite_tree(result["diagnostics"])


def test_token_same_origin_and_append_only_save_delete_export(tmp_path):
    before_asset = _sha256(CANONICAL_POSES)
    app, client = _client(tmp_path)
    cfg = client.get("/api/config").json()
    payload = {
        **_evaluate_payload(_seed(cfg, hand="left")),
        "notes": "comfortable elbow-down hit",
    }

    assert client.post("/api/styles", json=payload).status_code == 403
    assert client.post(
        "/api/styles",
        json=payload,
        headers={TOKEN_HEADER: "wrong"},
    ).status_code == 403
    assert client.post(
        "/api/styles",
        json=payload,
        headers={TOKEN_HEADER: "secret", "origin": "https://attacker.invalid"},
    ).status_code == 403
    headers = {
        TOKEN_HEADER: "secret",
        "origin": "http://testserver",
        "sec-fetch-site": "same-origin",
    }
    saved_response = client.post("/api/styles", json=payload, headers=headers)
    assert saved_response.status_code == 201, saved_response.text
    saved = saved_response.json()["style"]
    assert saved["revision"] == 1 and saved["enabled"] is True
    assert saved["operation"] == "create"
    assert len(saved["q_ref_rad"]) == len(G1_UPPER_BODY_JOINTS)
    assert len(saved["joint_weight"]) == len(G1_UPPER_BODY_JOINTS)
    assert saved["joint_weight"][0] == 0.0  # yaw selects, rather than being imitated

    listed = client.get("/api/styles").json()
    assert [row["id"] for row in listed["active"]] == [saved["id"]]
    assert len(listed["styles"]) == 1

    exported = client.post("/api/export", headers=headers)
    assert exported.status_code == 200
    body = exported.json()
    assert body["path"] == "style_library.json"
    assert body["document"]["corrections"][0]["id"] == saved["id"]
    export_path = app.state.store.root / body["path"]
    assert export_path.resolve().is_relative_to(app.state.store.root)
    assert json.loads(export_path.read_text()) == body["document"]
    # Export is trainer input, not the append-only audit record.
    assert "diagnostics" not in body["document"]["corrections"][0]
    assert "operation" not in body["document"]["corrections"][0]

    deleted = client.delete(f"/api/styles/{saved['id']}", headers=headers)
    assert deleted.status_code == 200
    tombstone = deleted.json()["style"]
    assert tombstone["revision"] == 2
    assert tombstone["enabled"] is False
    assert tombstone["operation"] == "tombstone"
    listed = client.get("/api/styles").json()
    assert len(listed["styles"]) == 2 and listed["active"] == []
    assert client.delete(f"/api/styles/{saved['id']}", headers=headers).status_code == 404

    post_delete_export = client.post("/api/export", headers=headers).json()["document"]
    assert post_delete_export["corrections"] == []
    assert _sha256(CANONICAL_POSES) == before_asset
    assert not list(app.state.store.root.glob("*.tmp"))


def test_actor_proposal_and_direct_teacher_round_trip_are_session_local(tmp_path):
    app, client = _client(tmp_path)
    cfg = client.get("/api/config").json()
    headers = {TOKEN_HEADER: "secret"}
    sampled = client.post(
        "/api/scenarios/sample", headers=headers, json={"count": 1, "seed": 17}
    )
    assert sampled.status_code == 201, sampled.text
    scenario = sampled.json()["scenarios"][0]
    scenario_id = scenario["id"]
    assert len(scenario["trajectories"]["warmstart"]) == 50
    assert len(scenario["trajectories"]["base"]) == 50
    assert not np.allclose(
        scenario["trajectories"]["base"],
        scenario["trajectories"]["warmstart"],
    )
    assert len(scenario["geometry"]["warmstart"]["frames"]) == 50
    timing = scenario["sampling_stats"]
    assert timing["task_purpose"] == "il_trajectory_teaching"
    assert timing["timing_source"] == "random_il_pair"
    assert timing["timing_pair"] == scenario["hit_index"]
    assert timing["timing_min_hit_waypoint"] == 20
    assert min(scenario["hit_index"]) >= 20
    assert timing["timing_actor_validated"] is False
    assert "timing_ray_pairs" not in timing
    assert "timing_risk" not in timing
    assert "timing_risk_threshold" not in timing
    assert all(len(goal["contact_position"]) == 3 for goal in scenario["goals"])
    assert scenario["context"]["goal_tth"] == pytest.approx(
        [value * scenario["dt"] for value in scenario["hit_index"]]
    )

    preview_q = list(scenario["trajectories"]["base"][0])
    preview_q[0] = max(
        float(app.state.domain.q_min[0]),
        min(float(app.state.domain.q_max[0]), preview_q[0] + 0.05),
    )
    preview = client.post(
        f"/api/scenarios/{scenario_id}/preview", json={"q": preview_q}
    )
    assert preview.status_code == 200, preview.text
    preview_body = preview.json()
    assert preview_body["q"] == pytest.approx(preview_q)
    assert set(preview_body["geometry"]) == {"joints", "sticks"}
    assert set(preview_body["geometry"]["sticks"]) == {"left", "right"}

    # Slider previews cover the full URDF range even though feasibility and
    # saved teaching poses retain the tighter actor-training posture box.
    margin_index, margin_value = next(
        (i, (joint["hard_min"] + joint["min"]) / 2.0)
        for i, joint in enumerate(cfg["joints"])
        if joint["hard_min"] < joint["min"] - 1e-6
    )
    margin_q = list(scenario["trajectories"]["base"][0])
    margin_q[margin_index] = margin_value
    margin_preview = client.post(
        f"/api/scenarios/{scenario_id}/preview", json={"q": margin_q}
    )
    assert margin_preview.status_code == 200, margin_preview.text
    strict_payload = _evaluate_payload(_seed(cfg))
    strict_payload["q"][margin_index] = margin_value
    assert client.post("/api/evaluate", json=strict_payload).status_code == 422

    assert client.post(
        f"/api/scenarios/{scenario_id}/preview", json={"q": preview_q[:-1]}
    ).status_code == 422

    left_goal = next(goal for goal in scenario["goals"] if goal["hand"] == "left")
    left_row = int(left_goal["hit_index"]) - 1
    left_frame = scenario["geometry"]["base"]["frames"][left_row]
    left_q = scenario["trajectories"]["base"][left_row]
    contact_target = list(left_frame["sticks"]["left"]["contact"])
    contact_target[2] += 0.01
    direction_target = list(left_frame["sticks"]["left"]["axis"])
    direction_target[0] += 0.025
    task_ik = client.post(
        f"/api/scenarios/{scenario_id}/task-ik",
        json={
            "q_seed": left_q,
            "hand": "left",
            "contact_position": contact_target,
            "stick_direction": direction_target,
            "waist": left_q[:3],
            "joint_guides": {"left_elbow_joint": left_q[6]},
            "iterations": 30,
        },
    )
    assert task_ik.status_code == 200, task_ik.text
    task_body = task_ik.json()
    assert len(task_body["q"]) == 17
    assert task_body["metrics"]["contact_error_m"] < 0.02
    assert task_body["metrics"]["direction_error_deg"] < 5.0
    assert task_body["guided_joints"] == ["left_elbow_joint"]
    assert task_body["priority"]["soft"][:2] == [
        "stick_azimuth_elevation", "contact_position",
    ]
    assert task_body["priority"]["weights"]["stick_direction_1_minus_cos"] == 2000.0
    assert task_body["priority"]["weights"]["contact_position_m2"] == 20000.0
    assert not np.allclose(task_body["q"], left_q)
    bad_axis = client.post(
        f"/api/scenarios/{scenario_id}/task-ik",
        json={
            "q_seed": left_q,
            "hand": "left",
            "contact_position": left_frame["sticks"]["left"]["contact"],
            "stick_direction": [0.0, 0.0, 0.0],
            "waist": left_q[:3],
            "iterations": 10,
        },
    )
    assert bad_axis.status_code == 422

    opposite_arm_keyframe = client.post(
        f"/api/scenarios/{scenario_id}/keyframes",
        headers=headers,
        json={
            "waypoint": 1,
            "hand": "left",
            "joint_values": {"right_elbow_joint": 0.5},
            "mode": "soft",
            "weight": 1.0,
        },
    )
    assert opposite_arm_keyframe.status_code == 422
    assert "opposite arm" in opposite_arm_keyframe.text

    joint = scenario["joint_names"][0]
    value = scenario["trajectories"]["warmstart"][0][0]
    keyframed = client.post(
        f"/api/scenarios/{scenario_id}/keyframes",
        headers=headers,
        json={
            "waypoint": 1,
            "hand": "left",
            "joint_values": {joint: value},
            "mode": "soft",
            "weight": 1.0,
        },
    )
    assert keyframed.status_code == 200, keyframed.text
    optimized = client.post(
        f"/api/scenarios/{scenario_id}/optimize",
        headers=headers,
        json={
            "iterations": 1,
            "lr": 1.0e-4,
            "task_constraint": {
                "hand": "left",
                "contact_position": contact_target,
                "stick_direction": direction_target,
                "waist": left_q[:3],
                "terminal_q": task_body["q"],
            },
        },
    )
    assert optimized.status_code == 200, optimized.text
    result = optimized.json()["scenario"]
    assert len(result["trajectories"]["teacher"]) == 50
    assert len(result["geometry"]["teacher"]["frames"]) == 50
    assert result["rounds"][-1]["iteration"] == 1
    assert result["task_constraint"]["hand"] == "left"
    assert result["diagnostics"]["teacher"]["task_space"]["left"]

    saved = client.post(
        f"/api/scenarios/{scenario_id}/save", headers=headers, json={"hand": "left"}
    )
    teacher_dir = app.state.store.root / "teachers" / scenario_id
    accepted = result["diagnostics"]["teacher"]["feasibility"]["accepted"]
    if accepted:
        assert saved.status_code == 200, saved.text
        assert saved.json()["scenario"]["saved"] is True
        assert saved.json()["scenario"]["teaching_hand"] == "left"
        assert (teacher_dir / "teacher.pt").is_file()
        assert (teacher_dir / "metadata.json").is_file()
        assert json.loads((teacher_dir / "metadata.json").read_text())["teaching_hand"] == "left"
        assert teacher_dir.resolve().is_relative_to(app.state.store.root)
    else:
        assert saved.status_code == 409
        assert not teacher_dir.exists()


@pytest.mark.parametrize(
    "session",
    ("../escape", "a/b", "/absolute", ".", "with.dot", "x" * 65),
)
def test_session_path_traversal_is_rejected(tmp_path, session):
    with pytest.raises(ValueError):
        create_app(session, runs_root=tmp_path / "runs", token="secret")
    assert not (tmp_path / "escape").exists()


def test_symlink_session_escape_is_rejected(tmp_path):
    runs = tmp_path / "runs"
    outside = tmp_path / "outside"
    runs.mkdir()
    outside.mkdir()
    (runs / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        create_app("linked", runs_root=runs, token="secret")


def test_static_frontend_has_no_remote_resources():
    files = sorted(path for path in STATIC_DIR.rglob("*") if path.is_file())
    assert files, "the editor must ship its frontend locally"
    assert (STATIC_DIR / "index.html") in files
    remote = re.compile(
        r"(?:src|href)\s*=\s*['\"](?:https?:)?//|"
        r"@import\s+(?:url\()?\s*['\"]?https?://|"
        r"fetch\s*\(\s*['\"]https?://",
        re.IGNORECASE,
    )
    offenders = []
    for path in files:
        if path.suffix.lower() not in {".html", ".css", ".js", ".svg"}:
            continue
        match = remote.search(path.read_text(encoding="utf-8"))
        if match:
            offenders.append((path.relative_to(ROOT).as_posix(), match.group(0)))
    assert not offenders, offenders


def _unused_local_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError:
        pytest.skip("sandbox forbids opening even a loopback socket")


def test_server_process_starts_on_loopback_and_terminates_cleanly(tmp_path):
    port = _unused_local_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.strike_pose_editor.server",
            "--session",
            "smoke",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--runs-root",
            str(tmp_path / "runs"),
            "--static-dir",
            str(STATIC_DIR),
            "--token",
            "smoke-token",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15.0
        health = None
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5) as response:
                    health = json.load(response)
                break
            except OSError:
                time.sleep(0.1)
        assert health == {"ok": True, "session": "smoke"}, (
            process.poll(), process.stdout.read() if process.poll() is not None else ""
        )
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    # ``Popen.terminate`` is SIGTERM on Linux. Uvicorn may consume it and
    # return zero or the interpreter may report the requested signal itself;
    # either outcome confirms the editor stopped without needing SIGKILL.
    assert process.returncode in (0, -signal.SIGTERM)


def test_launcher_is_local_cpu_only_and_rejects_unsafe_sessions():
    launcher = ROOT / "scripts" / "run_strike_pose_editor.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "tools.strike_pose_editor.server" in text
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env={**os.environ, "SESSION": "../escape", "PORT": "6010"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "SESSION must" in result.stderr
