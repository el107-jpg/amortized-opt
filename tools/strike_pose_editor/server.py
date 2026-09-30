"""FastAPI server for the isolated localhost strike-pose editor."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from .domain import DomainValidationError, EditorDomain
from .il_train import collate_training_records, load_backbone, save_teacher_bundle
from .hpipm_backend import (
    HpipmUnavailable,
    hpipm_available,
    solve_hpipm_trajectory,
)
from .trajectory_opt import (
    DirectTrajectoryOptimizer,
    TaskSpaceConstraint,
    TrajectoryScenario,
    WaypointKeyframe,
    quintic_terminal_initialization,
    sample_trajectory_scenario,
    warp_terminal_approach,
)

__all__ = ["SessionStore", "create_app", "main"]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "strike_pose_editor"
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "rhythm_timing_direct_admission_gpu1_20k_20260918"
    / "step_020000.pt"
)
SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
TOKEN_HEADER = "X-Strike-Editor-Token"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Any) -> None:
    """Write JSON beside its destination, fsync it, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class SessionStore:
    """Append-only style revisions confined to one validated session root."""

    def __init__(self, runs_root: str | Path, session: str, joint_names: tuple[str, ...]):
        if not SESSION_RE.fullmatch(session):
            raise ValueError(
                "session must be 1-64 characters: letters, digits, '_' or '-', "
                "and must begin with a letter or digit"
            )
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        candidate = self.runs_root / session
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.runs_root):
            raise ValueError("session resolves outside the strike-pose editor runs root")
        if candidate.is_symlink():
            raise ValueError("session directory may not be a symbolic link")
        candidate.mkdir(parents=False, exist_ok=True)
        if candidate.resolve() != resolved:
            raise ValueError("session directory changed while it was being opened")
        self.session = session
        self.root = resolved
        self.joint_names = tuple(joint_names)
        self.revisions_path = self.root / "revisions.json"
        self.export_path = self.root / "style_library.json"
        self._lock = threading.RLock()
        if not self.revisions_path.exists():
            _atomic_json(
                self.revisions_path,
                {
                    "schema_version": 1,
                    "session": self.session,
                    "joint_names": list(self.joint_names),
                    "styles": [],
                },
            )
        self._read_document()  # validate an existing session before serving it

    def _read_document(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.revisions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid session revision log {self.revisions_path}") from exc
        if raw.get("schema_version") != 1:
            raise RuntimeError("unsupported strike-pose revision schema")
        if raw.get("session") != self.session:
            raise RuntimeError("revision log belongs to a different session")
        if tuple(raw.get("joint_names", ())) != self.joint_names:
            raise RuntimeError("revision log joint ordering does not match this robot")
        if not isinstance(raw.get("styles"), list):
            raise RuntimeError("revision log styles must be a list")
        return raw

    def list_revisions(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._read_document()["styles"])

    @staticmethod
    def _active(revisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in revisions:
            style_id = str(row["id"])
            if style_id not in latest:
                order.append(style_id)
            latest[style_id] = row
        return [latest[k] for k in order if bool(latest[k].get("enabled"))]

    def active(self) -> list[dict[str, Any]]:
        return self._active(self.list_revisions())

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            doc = self._read_document()
            doc["styles"].append(row)
            _atomic_json(self.revisions_path, doc)
        return row

    def tombstone(self, style_id: str) -> dict[str, Any]:
        try:
            style_id = str(uuid.UUID(style_id))
        except ValueError as exc:
            raise KeyError(style_id) from exc
        with self._lock:
            doc = self._read_document()
            matches = [r for r in doc["styles"] if r.get("id") == style_id]
            if not matches or not bool(matches[-1].get("enabled")):
                raise KeyError(style_id)
            previous = matches[-1]
            row = dict(previous)
            row.update(
                {
                    "revision": int(previous["revision"]) + 1,
                    "enabled": False,
                    "operation": "tombstone",
                    "created_at": _utc_now(),
                }
            )
            doc["styles"].append(row)
            _atomic_json(self.revisions_path, doc)
            return row

    def export(self) -> dict[str, Any]:
        with self._lock:
            revisions = self._read_document()["styles"]
            active = self._active(revisions)
            corrections = []
            for row in active:
                corrections.append(
                    {
                        key: row[key]
                        for key in (
                            "id",
                            "revision",
                            "enabled",
                            "hand",
                            "drum",
                            "yaw_center_rad",
                            "yaw_half_width_rad",
                            "q_ref_rad",
                            "joint_weight",
                            "contact_sigma_m",
                            "notes",
                            "created_at",
                        )
                    }
                )
            output = {
                "schema_version": 1,
                "joint_names": list(self.joint_names),
                "corrections": corrections,
            }
            _atomic_json(self.export_path, output)
            return output


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: list[float]
    hand: str
    drum: str
    contact_sigma: float | None = None


class StyleCreateRequest(EvaluateRequest):
    yaw_center_rad: float | None = None
    yaw_half_width_rad: float | None = None
    joint_weight: list[float] | None = None
    notes: str = Field(default="", max_length=2000)


class ScenarioSampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=1, ge=1, le=4)
    seed: int | None = None


class KeyframeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    waypoint: int = Field(ge=1, le=50)
    hand: str | None = Field(default=None, pattern="^(left|right)$")
    joint_values: list[float] | dict[str, float]
    mode: str = Field(default="soft", pattern="^(soft|hard)$")
    weight: float = Field(default=1.0, ge=0.0, le=1.0e4)
    notes: str = Field(default="", max_length=500)


class PosePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: list[float]


class TaskIKRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q_seed: list[float]
    hand: str = Field(pattern="^(left|right)$")
    contact_position: list[float]
    stick_direction: list[float]
    waist: list[float]
    joint_guides: dict[str, float] = Field(default_factory=dict)
    guide_weight: float = Field(default=25.0, ge=0.0, le=1.0e4)
    iterations: int = Field(default=140, ge=10, le=400)


class TaskConstraintRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hand: str = Field(pattern="^(left|right)$")
    contact_position: list[float] = Field(min_length=3, max_length=3)
    stick_direction: list[float] = Field(min_length=3, max_length=3)
    waist: list[float] = Field(min_length=3, max_length=3)
    terminal_q: list[float] = Field(min_length=17, max_length=17)


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: int = Field(default=60, ge=1, le=500)
    lr: float = Field(default=1.0e-3, gt=0.0, le=0.1)
    backend: str = Field(default="auto", pattern="^(auto|hpipm|adam)$")
    sqp_initialization: str = Field(
        default="current", pattern="^(current|quintic)$"
    )
    task_constraint: TaskConstraintRequest | None = None


class SaveTeacherRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hand: str = Field(pattern="^(left|right)$")


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_ids: list[str] = Field(min_length=1, max_length=128)
    steps: int = Field(default=100, ge=1, le=10_000)
    device: str = "cpu"


@dataclasses.dataclass
class ScenarioRecord:
    scenario: TrajectoryScenario
    optimizer: DirectTrajectoryOptimizer
    keyframes: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    rounds: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    saved: bool = False
    teaching_hand: str | None = None
    task_constraint: TaskSpaceConstraint | None = None
    base: list[list[float]] | None = None
    adapted: list[list[float]] | None = None
    geometry: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    base_diagnostics: dict[str, Any] | None = None
    optimizer_backend: str | None = None
    backend_diagnostics: dict[str, Any] | None = None


@dataclasses.dataclass
class TrainJob:
    job_id: str
    root: Path
    process: subprocess.Popen[Any]
    scenario_ids: tuple[str, ...]
    attached: bool = False


def _teacher_bundle(records: list[ScenarioRecord]):
    if any(row.teaching_hand not in ("left", "right") for row in records):
        raise ValueError("every saved teacher must identify exactly one teaching hand")
    return collate_training_records(
        [
            row.scenario.training_bundle(
                row.optimizer.trajectory.detach(),
                teaching_hand=row.teaching_hand,
            )
            for row in records
        ]
    )


def _domain_error(exc: DomainValidationError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise HTTPException(status_code=422, detail=f"{name} must be finite")
    return value


def create_app(
    session: str,
    *,
    runs_root: str | Path | None = None,
    token: str | None = None,
    static_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    train_device: str = "cpu",
) -> FastAPI:
    """Create one isolated editor app.

    All persistent writes and child-job outputs remain below the validated
    session root. ``runs_root`` and ``checkpoint`` are startup-only settings;
    HTTP clients can never choose filesystem paths.
    """
    domain = EditorDomain()
    store = SessionStore(runs_root or DEFAULT_RUNS_ROOT, session, domain.joint_names)
    mutation_token = token or secrets.token_urlsafe(24)
    if not mutation_token:
        raise ValueError("token must not be empty")

    app = FastAPI(title="Strike Pose Editor", version="1")
    app.state.domain = domain
    app.state.store = store
    app.state.mutation_token = mutation_token
    app.state.checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT).resolve(strict=False)
    app.state.train_device = str(train_device)
    app.state.scenarios: dict[str, ScenarioRecord] = {}
    app.state.jobs: dict[str, TrainJob] = {}
    app.state.actor = None
    app.state.trajectory_lock = threading.RLock()

    @app.middleware("http")
    async def local_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    async def require_mutation_access(
        request: Request,
        supplied: str | None = Header(default=None, alias=TOKEN_HEADER),
    ) -> None:
        if supplied is None or not secrets.compare_digest(supplied, mutation_token):
            raise HTTPException(status_code=403, detail="invalid mutation token")
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            host = request.headers.get("host", "")
            if parsed.scheme not in ("http", "https") or parsed.netloc != host:
                raise HTTPException(status_code=403, detail="cross-origin mutation rejected")
        fetch_site = request.headers.get("sec-fetch-site")
        if fetch_site and fetch_site not in ("same-origin", "none"):
            raise HTTPException(status_code=403, detail="cross-origin mutation rejected")

    def get_scenario(scenario_id: str) -> ScenarioRecord:
        with app.state.trajectory_lock:
            record = app.state.scenarios.get(scenario_id)
        if record is None:
            raise HTTPException(status_code=404, detail="scenario not found")
        return record

    def ensure_actor_model() -> None:
        """Lazily restore the actor used to propose an editable trajectory."""
        if app.state.actor is None:
            if not app.state.checkpoint.is_file():
                raise HTTPException(
                    status_code=409, detail="configured actor checkpoint is unavailable"
                )
            model, _raw, _digest = load_backbone(app.state.checkpoint, device="cpu")
            app.state.actor = model

    def actor_proposal(scenario: TrajectoryScenario) -> torch.Tensor:
        """Run the current amortized actor once; the quintic is input, not output."""
        ensure_actor_model()
        model = app.state.actor
        if scenario.context.num_drums != model.cfg.num_drums \
                or scenario.context.num_future != model.cfg.num_future:
            raise HTTPException(
                status_code=500,
                detail="sampled scenario context does not match the actor checkpoint",
            )
        state = torch.cat([scenario.q0, scenario.qd0], dim=-1)
        with torch.no_grad():
            encoded = model.encode_context(scenario.context)
            trajectory = model(
                scenario.init * scenario.masks.elem_mask,
                state,
                encoded,
                scenario.masks,
            ).traj
        if tuple(trajectory.shape) != (1, 50, 17) or not torch.isfinite(trajectory).all():
            raise HTTPException(status_code=500, detail="actor produced an invalid trajectory")
        return trajectory.detach().clone()

    def geometry_for(record: ScenarioRecord, name: str, trajectory: Any) -> dict[str, Any]:
        cached = record.geometry.get(name)
        if cached is not None:
            return cached
        value = torch.as_tensor(trajectory, dtype=torch.float32)
        if tuple(value.shape) == (50, 17):
            value = value.unsqueeze(0)
        if tuple(value.shape) != (1, 50, 17):
            raise ValueError(f"trajectory {name!r} must have shape (50, 17)")
        helper = DirectTrajectoryOptimizer(
            record.scenario, initial_trajectory=value
        )
        geometry = helper.geometry_json()
        record.geometry[name] = geometry
        return geometry

    def record_payload(record: ScenarioRecord) -> dict[str, Any]:
        data = record.scenario.to_json()
        trajectories: dict[str, Any] = {"warmstart": data["init"]}
        geometries: dict[str, Any] = {
            "warmstart": geometry_for(record, "warmstart", record.scenario.init)
        }
        diagnostics: dict[str, Any] = {}
        if record.rounds:
            latest = record.optimizer.result_json(include_geometry=True)
            if record.backend_diagnostics is not None:
                latest["diagnostics"]["optimizer_backend"] = record.backend_diagnostics
            trajectories["teacher"] = latest["trajectory"]
            geometries["teacher"] = latest["geometry"]
            diagnostics["teacher"] = latest["diagnostics"]
        if record.base is not None:
            trajectories["base"] = record.base
            geometries["base"] = geometry_for(record, "base", record.base)
            if record.base_diagnostics is not None:
                diagnostics["base"] = record.base_diagnostics
        if record.adapted is not None:
            trajectories["adapted"] = record.adapted
            geometries["adapted"] = geometry_for(record, "adapted", record.adapted)
        data.update(
            {
                "trajectories": trajectories,
                "geometry": geometries,
                "diagnostics": diagnostics,
                "keyframes": record.optimizer.keyframes_json(),
                "rounds": record.rounds,
                "saved": record.saved,
                "teaching_hand": record.teaching_hand,
                "task_constraint": (
                    record.task_constraint.to_json()
                    if record.task_constraint is not None else None
                ),
                "optimizer_backend": record.optimizer_backend,
                "optimizer_backend_diagnostics": record.backend_diagnostics,
            }
        )
        return data

    def attach_job_predictions(job: TrainJob) -> None:
        if job.attached:
            return
        predictions_path = job.root / "predictions.pt"
        if not predictions_path.is_file():
            return
        raw = torch.load(predictions_path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict) or raw.get("format") != "strike_pose_il_predictions_v1":
            raise RuntimeError("invalid IL prediction artifact")
        with app.state.trajectory_lock:
            for scenario_id in job.scenario_ids:
                record = app.state.scenarios.get(scenario_id)
                if record is None:
                    continue
                for name in ("base", "adapted"):
                    value = raw.get(name, {}).get(scenario_id)
                    if not torch.is_tensor(value) or tuple(value.shape) != (50, 17):
                        raise RuntimeError(
                            f"IL predictions missing {name} for {scenario_id}"
                        )
                    setattr(record, name, value.tolist())
                    record.geometry.pop(name, None)
            job.attached = True

    def job_payload(job: TrainJob) -> dict[str, Any]:
        path = job.root / "status.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {"status": "running"}
        else:
            code = job.process.poll()
            payload = {
                "status": "queued" if code is None else "failed",
                "error": None if code is None else f"trainer exited with status {code}",
            }
        payload["job_id"] = job.job_id
        steps = max(int(payload.get("steps", 0)), 1)
        payload["progress"] = min(max(float(payload.get("step", 0)) / steps, 0.0), 1.0)
        if payload.get("status") == "complete":
            attach_job_predictions(job)
            payload["scenario_ids"] = list(job.scenario_ids)
        return payload

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "session": session}

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        has_hpipm, hpipm_error = hpipm_available()
        out = domain.config()
        out.update(
            {
                "session": session,
                "mutation_token": mutation_token,
                "train_device": app.state.train_device,
                "checkpoint_available": app.state.checkpoint.is_file(),
                "actor_loaded": app.state.actor is not None,
                "sampling_mode": "random_il_teaching",
                "optimizer_backends": [
                    {
                        "id": "hpipm",
                        "label": "HPIPM SQP",
                        "available": has_hpipm,
                        "unavailable_reason": hpipm_error,
                    },
                    {
                        "id": "adam",
                        "label": "Adam fallback",
                        "available": True,
                        "unavailable_reason": None,
                    },
                ],
                "default_optimizer_backend": "hpipm" if has_hpipm else "adam",
            }
        )
        return out

    @app.post(
        "/api/scenarios/sample",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_mutation_access)],
    )
    async def sample_scenarios(body: ScenarioSampleRequest) -> dict[str, Any]:
        seed0 = body.seed if body.seed is not None else secrets.randbelow(2**31 - 1)
        sampled: list[dict[str, Any]] = []
        with app.state.trajectory_lock:
            ensure_actor_model()
            for index in range(body.count):
                scenario = sample_trajectory_scenario(seed0 + index * 1_000_003)
                proposal = actor_proposal(scenario)
                optimizer = DirectTrajectoryOptimizer(
                    scenario, initial_trajectory=proposal
                )
                record = ScenarioRecord(
                    scenario=scenario,
                    optimizer=optimizer,
                    base=proposal[0].tolist(),
                    base_diagnostics=optimizer.result_json(
                        include_geometry=False
                    )["diagnostics"],
                )
                record.geometry["base"] = optimizer.geometry_json()
                record.geometry["warmstart"] = geometry_for(
                    record, "warmstart", scenario.init
                )
                app.state.scenarios[scenario.scenario_id] = record
                sampled.append(record_payload(record))
        return {"scenarios": sampled}

    @app.get("/api/scenarios/{scenario_id}")
    async def read_scenario(scenario_id: str) -> dict[str, Any]:
        with app.state.trajectory_lock:
            return {"scenario": record_payload(get_scenario(scenario_id))}

    @app.post("/api/scenarios/{scenario_id}/preview")
    async def preview_pose(
        scenario_id: str, body: PosePreviewRequest
    ) -> dict[str, Any]:
        """Fast, read-only FK used while an editor slider is moving."""
        with app.state.trajectory_lock:
            record = get_scenario(scenario_id)
            sigma = {
                hand: float(record.scenario.strike.contact_sigma[0, hi].item())
                for hi, hand in enumerate(("left", "right"))
            }
        try:
            # Actor trajectories and the editor sliders may use the full URDF
            # range, while saved/evaluated poses retain the tighter training
            # posture box.  Read-only FK must therefore accept the hard range
            # or a single marginal actor joint freezes every slider preview.
            geometry = domain.pose_geometry(
                body.q, sigma, allow_posture_margin=True
            )
        except DomainValidationError as exc:
            raise _domain_error(exc) from exc
        return {"q": [float(value) for value in body.q], "geometry": geometry}

    @app.post("/api/scenarios/{scenario_id}/task-ik")
    async def solve_task_ik(
        scenario_id: str, body: TaskIKRequest
    ) -> dict[str, Any]:
        """Solve an elbow-down terminal pose from stick task coordinates."""
        with app.state.trajectory_lock:
            record = get_scenario(scenario_id)
            hi = 0 if body.hand == "left" else 1
            if not bool(record.scenario.active[0, hi].item()):
                raise HTTPException(status_code=422, detail=f"{body.hand} hand is idle")
            sigma = float(record.scenario.strike.contact_sigma[0, hi].item())
            target_drum = int(record.scenario.target_drum[0, hi].item())
            valid = record.scenario.context.drum_valid[0].detach().cpu().bool()
            heads = record.scenario.context.drum_pos[:, valid].detach().cpu().to(torch.float64)
            radii = record.scenario.context.drum_radius[:, valid].detach().cpu().to(torch.float64)
        try:
            return domain.solve_task_pose(
                body.q_seed,
                hand=body.hand,
                contact_position=body.contact_position,
                stick_direction=body.stick_direction,
                waist=body.waist,
                contact_sigma=sigma,
                joint_guides=body.joint_guides,
                guide_weight=body.guide_weight,
                iterations=body.iterations,
                drum_head_centers=heads,
                drum_radii=radii,
                target_drum=target_drum,
            )
        except DomainValidationError as exc:
            raise _domain_error(exc) from exc

    @app.post(
        "/api/scenarios/{scenario_id}/keyframes",
        dependencies=[Depends(require_mutation_access)],
    )
    async def set_keyframe(scenario_id: str, body: KeyframeRequest) -> dict[str, Any]:
        with app.state.trajectory_lock:
            record = get_scenario(scenario_id)
            if isinstance(body.joint_values, list):
                if len(body.joint_values) != len(record.scenario.joint_names):
                    raise HTTPException(status_code=422, detail="joint_values must have 17 values")
                values = dict(zip(record.scenario.joint_names, body.joint_values))
            else:
                values = dict(body.joint_values)
            if not values or not all(math.isfinite(float(value)) for value in values.values()):
                raise HTTPException(
                    status_code=422,
                    detail="joint_values must be nonempty and finite",
                )
            if body.hand is not None:
                disallowed = [
                    name for name in values
                    if not name.startswith("waist_")
                    and not name.startswith(f"{body.hand}_")
                ]
                if disallowed:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"{body.hand}-hand keyframe cannot adjust the opposite arm: "
                            + ", ".join(disallowed)
                        ),
                    )
            replacement = {
                "waypoint": body.waypoint,
                "hand": body.hand,
                "values": values,
                "mode": body.mode,
                "weight": body.weight,
                "notes": body.notes,
            }
            existing = next(
                (
                    row for row in record.keyframes
                    if row["waypoint"] == body.waypoint
                    and row.get("hand") == body.hand
                ),
                None,
            )
            if existing is not None:
                record.keyframes.remove(existing)
                merged = dict(existing["values"])
                merged.update(values)
                replacement["values"] = merged
            record.keyframes.append(replacement)
            record.keyframes.sort(key=lambda row: int(row["waypoint"]))
            previous = record.optimizer.trajectory.detach().clone()
            try:
                optimizer = DirectTrajectoryOptimizer(
                    record.scenario,
                    initial_trajectory=previous,
                    keyframes=[
                        WaypointKeyframe(
                            waypoint=int(row["waypoint"]),
                            values=row["values"],
                            mode=str(row["mode"]),
                            weight=float(row["weight"]),
                        )
                        for row in record.keyframes
                    ],
                    task_constraints=(
                        [record.task_constraint]
                        if record.task_constraint is not None else []
                    ),
                    learning_rate=record.optimizer.learning_rate,
                )
            except (TypeError, ValueError) as exc:
                record.keyframes.remove(replacement)
                if existing is not None:
                    record.keyframes.append(existing)
                    record.keyframes.sort(key=lambda row: int(row["waypoint"]))
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            record.optimizer = optimizer
            record.saved = False
            record.geometry.pop("teacher", None)
            return {"scenario": record_payload(record)}

    @app.post(
        "/api/scenarios/{scenario_id}/optimize",
        dependencies=[Depends(require_mutation_access)],
    )
    async def optimize_scenario(
        scenario_id: str, body: OptimizeRequest
    ) -> dict[str, Any]:
        with app.state.trajectory_lock:
            record = get_scenario(scenario_id)
            has_hpipm, hpipm_error = hpipm_available()
            backend = body.backend
            if backend == "auto":
                backend = "hpipm" if has_hpipm else "adam"
            if backend == "hpipm" and not has_hpipm:
                raise HTTPException(status_code=409, detail=hpipm_error)
            if backend != "hpipm" and body.sqp_initialization != "current":
                raise HTTPException(
                    status_code=422,
                    detail="quintic SQP initialization requires the HPIPM backend",
                )
            if body.task_constraint is not None:
                requested = body.task_constraint
                terminal_q = domain.validate_pose(
                    requested.terminal_q, allow_posture_margin=True
                )
                constraint = TaskSpaceConstraint(
                    hand=requested.hand,
                    contact_position=requested.contact_position,
                    stick_direction=requested.stick_direction,
                    waist=requested.waist,
                    terminal_q=requested.terminal_q,
                )
                hit_col = 0 if requested.hand == "left" else 1
                hit_row = int(record.scenario.hit_index[0, hit_col].item()) - 1
                selected_cols = [
                    index for index, name in enumerate(record.scenario.joint_names)
                    if name.startswith("waist_")
                    or name.startswith(f"{requested.hand}_")
                ]
                if backend == "hpipm" and body.sqp_initialization == "quintic":
                    previous = quintic_terminal_initialization(
                        record.scenario, terminal_q, hand=requested.hand
                    )
                else:
                    previous = warp_terminal_approach(
                        record.optimizer.trajectory.detach().clone(),
                        terminal_q,
                        hit_waypoint=hit_row + 1,
                        joint_indices=selected_cols,
                    )
                active_keyframes = [
                    row for row in record.keyframes
                    if row.get("hand") in (None, requested.hand)
                ]
                try:
                    record.optimizer = DirectTrajectoryOptimizer(
                        record.scenario,
                        initial_trajectory=previous,
                        keyframes=[
                            WaypointKeyframe(
                                waypoint=int(row["waypoint"]),
                                values=row["values"],
                                mode=str(row["mode"]),
                                weight=float(row["weight"]),
                            )
                            for row in active_keyframes
                        ],
                        task_constraints=[constraint],
                        learning_rate=body.lr,
                    )
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                record.task_constraint = constraint
                record.saved = False
                record.geometry.pop("teacher", None)
            backend_diagnostics: dict[str, Any]
            if backend == "hpipm":
                if record.task_constraint is None:
                    raise HTTPException(
                        status_code=422,
                        detail="HPIPM SQP requires a solved task-space IK target",
                    )
                before = record.optimizer
                try:
                    hpipm_result = solve_hpipm_trajectory(
                        record.scenario,
                        before.trajectory.detach(),
                        task_constraint=record.task_constraint,
                        keyframes=before.keyframes,
                        max_iterations=body.iterations,
                    )
                    candidate = DirectTrajectoryOptimizer(
                        record.scenario,
                        initial_trajectory=hpipm_result.trajectory,
                        keyframes=before.keyframes,
                        task_constraints=[record.task_constraint],
                        learning_rate=body.lr,
                    )
                except HpipmUnavailable as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422, detail=f"HPIPM SQP failed: {exc}"
                    ) from exc

                def score(optimizer: DirectTrajectoryOptimizer) -> tuple[float, ...]:
                    payload = optimizer.result_json(include_geometry=False)
                    feasibility = payload["diagnostics"]["feasibility"]
                    return (
                        0.0 if bool(feasibility["feasible"]) else 1.0,
                        0.0 if bool(feasibility["feasible"]) else float(
                            feasibility["normalized_violation_score"]
                        ),
                        float(payload["diagnostics"]["objective"]),
                    )

                candidate_selected = score(candidate) < score(before)
                record.optimizer = candidate if candidate_selected else before
                record.optimizer.iteration = int(hpipm_result.iterations)
                record.optimizer._converged = bool(hpipm_result.converged)
                record.optimizer._stop_reason = (
                    "hpipm_sqp" if candidate_selected
                    else "hpipm_rescore_kept_warmstart"
                )
                backend_diagnostics = {
                    **hpipm_result.diagnostics,
                    "backend": "hpipm",
                    "requested_backend": body.backend,
                    "sqp_initialization": body.sqp_initialization,
                    "sqp_initialization_source": (
                        "fresh_velocity_matched_terminal_quintic"
                        if body.sqp_initialization == "quintic"
                        else "current_trajectory_terminal_warp"
                    ),
                    "candidate_selected_by_fused_cost": candidate_selected,
                }
                result = record.optimizer.result_json()
            else:
                if not math.isclose(record.optimizer.learning_rate, body.lr):
                    for group in record.optimizer.optimizer.param_groups:
                        group["lr"] = body.lr
                    record.optimizer.learning_rate = body.lr
                result = record.optimizer.solve(
                    max_iterations=body.iterations,
                    min_iterations=min(40, body.iterations),
                    patience=min(30, max(body.iterations // 4, 1)),
                )
                backend_diagnostics = {
                    "backend": "adam",
                    "requested_backend": body.backend,
                    "iterations": int(result["iteration"]),
                    "converged": bool(
                        result["diagnostics"]["convergence"]["converged"]
                    ),
                    "message": result["diagnostics"]["convergence"]["reason"],
                }
            record.optimizer_backend = backend
            record.backend_diagnostics = backend_diagnostics
            result["diagnostics"]["optimizer_backend"] = backend_diagnostics
            record.geometry["teacher"] = result["geometry"]
            record.rounds.append(
                {
                    "iteration": result["iteration"],
                    "trajectory": result["trajectory"],
                    "geometry": result["geometry"],
                    "diagnostics": result["diagnostics"],
                    "keyframes": result["keyframes"],
                    "status": "complete",
                    "backend": backend,
                    "backend_diagnostics": backend_diagnostics,
                    "hand": (
                        record.task_constraint.hand
                        if record.task_constraint is not None else None
                    ),
                }
            )
            record.saved = False
            return {"scenario": record_payload(record)}

    @app.post(
        "/api/scenarios/{scenario_id}/save",
        dependencies=[Depends(require_mutation_access)],
    )
    async def save_teacher(
        scenario_id: str, body: SaveTeacherRequest
    ) -> dict[str, Any]:
        with app.state.trajectory_lock:
            record = get_scenario(scenario_id)
            if not record.rounds:
                raise HTTPException(status_code=409, detail="optimize the teacher before saving")
            if record.task_constraint is None or record.task_constraint.hand != body.hand:
                raise HTTPException(
                    status_code=409,
                    detail="optimize this hand's persistent task-space target before saving",
                )
            feasibility = record.optimizer.status_json()["feasibility"]
            if not feasibility.get("accepted", False):
                failed = [
                    name for name, passed in feasibility.get("checks", {}).items()
                    if not passed
                ]
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "teacher has not passed feasibility checks: "
                        + ", ".join(failed)
                    ),
                )
            teacher_root = store.root / "teachers" / scenario_id
            teacher_root.mkdir(parents=True, exist_ok=True)
            record.teaching_hand = body.hand
            bundle = _teacher_bundle([record])
            save_teacher_bundle(teacher_root / "teacher.pt", bundle)
            _atomic_json(
                teacher_root / "metadata.json",
                {
                    "scenario_id": scenario_id,
                    "saved_at": _utc_now(),
                    "iterations": record.optimizer.iteration,
                    "optimizer_backend": record.optimizer_backend,
                    "optimizer_backend_diagnostics": record.backend_diagnostics,
                    "teaching_hand": body.hand,
                    "task_constraint": record.task_constraint.to_json(),
                    "keyframes": record.optimizer.keyframes_json(),
                },
            )
            record.saved = True
            return {"scenario": record_payload(record)}

    @app.post("/api/train", dependencies=[Depends(require_mutation_access)])
    async def start_training(body: TrainRequest) -> dict[str, Any]:
        if body.device != app.state.train_device:
            raise HTTPException(
                status_code=422,
                detail=f"this app session is configured for {app.state.train_device!r}",
            )
        if body.device != "cpu" and not re.fullmatch(r"cuda:\d+", body.device):
            raise HTTPException(status_code=422, detail="device must be 'cpu' or 'cuda:N'")
        if not app.state.checkpoint.is_file():
            raise HTTPException(status_code=409, detail="configured actor checkpoint is unavailable")
        if len(set(body.scenario_ids)) != len(body.scenario_ids):
            raise HTTPException(status_code=422, detail="scenario_ids must be unique")
        with app.state.trajectory_lock:
            records = [get_scenario(scenario_id) for scenario_id in body.scenario_ids]
            if not all(record.saved for record in records):
                raise HTTPException(status_code=409, detail="all teachers must be saved first")
            bundle = _teacher_bundle(records)
        job_id = "il-" + uuid.uuid4().hex[:12]
        job_root = store.root / "jobs" / job_id
        job_root.mkdir(parents=True, exist_ok=False)
        bundle_path = save_teacher_bundle(job_root / "teachers.pt", bundle)
        command = [
            sys.executable,
            "-m",
            "tools.strike_pose_editor.il_train",
            "--bundle",
            str(bundle_path),
            "--checkpoint",
            str(app.state.checkpoint),
            "--output-dir",
            str(job_root),
            "--steps",
            str(body.steps),
            "--device",
            body.device,
        ]
        env = os.environ.copy()
        env["TMPDIR"] = str(store.root / ".tmp")
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        log = (job_root / "trainer.log").open("xb")
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
        job = TrainJob(job_id, job_root, process, tuple(body.scenario_ids))
        app.state.jobs[job_id] = job
        return {"job_id": job_id, "status": "queued", "progress": 0.0}

    @app.get("/api/jobs/{job_id}")
    async def read_job(job_id: str) -> dict[str, Any]:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            return job_payload(job)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def stop_child_jobs() -> None:
        for job in tuple(app.state.jobs.values()):
            if job.process.poll() is not None:
                continue
            job.process.terminate()
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait(timeout=5)

    app.add_event_handler("shutdown", stop_child_jobs)

    @app.post("/api/evaluate")
    async def evaluate(body: EvaluateRequest) -> dict[str, Any]:
        try:
            return domain.evaluate(body.q, body.hand, body.drum, body.contact_sigma)
        except DomainValidationError as exc:
            raise _domain_error(exc) from exc

    @app.get("/api/styles")
    async def list_styles() -> dict[str, Any]:
        revisions = store.list_revisions()
        return {"styles": revisions, "active": SessionStore._active(revisions)}

    @app.post(
        "/api/styles",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_mutation_access)],
    )
    async def create_style(body: StyleCreateRequest) -> dict[str, Any]:
        try:
            evaluated = domain.evaluate(body.q, body.hand, body.drum, body.contact_sigma)
        except DomainValidationError as exc:
            raise _domain_error(exc) from exc

        q = evaluated["q"]
        yaw_center = q[domain._yaw_i] if body.yaw_center_rad is None else _finite(
            body.yaw_center_rad, "yaw_center_rad"
        )
        if not (domain.q_min[domain._yaw_i] <= yaw_center <= domain.q_max[domain._yaw_i]):
            raise HTTPException(status_code=422, detail="yaw_center_rad is outside waist-yaw limits")
        if body.yaw_half_width_rad is None:
            bins = sorted(set(float(v) for v in domain.library.yaw_grid_rad))
            gaps = [b - a for a, b in zip(bins, bins[1:]) if b > a]
            yaw_half_width = (min(gaps) / 2.0) if gaps else math.radians(12.5)
        else:
            yaw_half_width = _finite(body.yaw_half_width_rad, "yaw_half_width_rad")
        if not (0.0 < yaw_half_width <= math.pi):
            raise HTTPException(status_code=422, detail="yaw_half_width_rad must be in (0, pi]")

        if body.joint_weight is None:
            weights = domain.default_joint_weight(body.hand)
        else:
            weights = [float(v) for v in body.joint_weight]
            if len(weights) != len(domain.joint_names):
                raise HTTPException(status_code=422, detail="joint_weight must have 17 values")
            if not all(math.isfinite(v) and v >= 0.0 for v in weights):
                raise HTTPException(
                    status_code=422, detail="joint_weight values must be finite and non-negative"
                )
            if not any(v > 0.0 for v in weights):
                raise HTTPException(status_code=422, detail="joint_weight may not be all zero")

        row = {
            "id": str(uuid.uuid4()),
            "revision": 1,
            "enabled": True,
            "operation": "create",
            "hand": body.hand,
            "drum": body.drum,
            "yaw_center_rad": float(yaw_center),
            "yaw_half_width_rad": float(yaw_half_width),
            "q_ref_rad": q,
            "joint_weight": weights,
            "contact_sigma_m": float(evaluated["selection"]["contact_sigma"]),
            "notes": body.notes,
            "created_at": _utc_now(),
            # Derived values are audit data only.  A trainer must recompute them.
            "diagnostics": {
                "geometry": evaluated["geometry"],
                "target": evaluated["target"],
                "metrics": evaluated["metrics"],
                "collision": evaluated["diagnostics"]["collision"],
                "warnings": evaluated["warnings"],
            },
        }
        return {"style": store.append(row)}

    @app.delete(
        "/api/styles/{style_id}",
        dependencies=[Depends(require_mutation_access)],
    )
    async def delete_style(style_id: str) -> dict[str, Any]:
        try:
            return {"style": store.tombstone(style_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="active style not found") from exc

    @app.post("/api/export", dependencies=[Depends(require_mutation_access)])
    async def export_styles() -> dict[str, Any]:
        document = store.export()
        return {"path": store.export_path.name, "document": document}

    chosen_static = DEFAULT_STATIC_DIR if static_dir is None else Path(static_dir)
    if chosen_static.is_dir():
        chosen_static = chosen_static.resolve()

        # Keep the three small frontend assets in memory.  Besides avoiding
        # filesystem work on every request, this avoids Starlette's sync
        # threadpool path: in Torch-initialised processes that path can
        # deadlock TestClient/AnyIO worker startup on some libgomp builds.
        index_bytes = (chosen_static / "index.html").read_bytes()
        script_bytes = (chosen_static / "app.js").read_bytes()
        style_bytes = (chosen_static / "style.css").read_bytes()

        @app.get("/", include_in_schema=False)
        async def index_page():
            return Response(index_bytes, media_type="text/html")

        @app.get("/static/app.js", include_in_schema=False)
        async def app_script():
            return Response(script_bytes, media_type="application/javascript")

        @app.get("/static/style.css", include_in_schema=False)
        async def app_styles():
            return Response(style_bytes, media_type="text/css")

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        default=os.environ.get("STRIKE_POSE_EDITOR_SESSION"),
        required=os.environ.get("STRIKE_POSE_EDITOR_SESSION") is None,
    )
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("STRIKE_POSE_EDITOR_PORT", "6010"))
    )
    parser.add_argument(
        "--runs-root",
        default=os.environ.get("STRIKE_POSE_EDITOR_RUNS_ROOT", str(DEFAULT_RUNS_ROOT)),
    )
    parser.add_argument("--static-dir", default=str(DEFAULT_STATIC_DIR))
    parser.add_argument("--token", default=os.environ.get("STRIKE_POSE_EDITOR_TOKEN"))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("STRIKE_POSE_EDITOR_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
    )
    parser.add_argument(
        "--train-device",
        default=os.environ.get("STRIKE_POSE_EDITOR_TRAIN_DEVICE", "cpu"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not (1 <= args.port <= 65535):
        raise SystemExit("--port must be between 1 and 65535")
    if args.train_device != "cpu":
        raise SystemExit("the isolated localhost launcher currently supports --train-device cpu")
    # The server process is intentionally blind to GPUs even when invoked from a
    # shell whose CUDA_VISIBLE_DEVICES points at an active training GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    app = create_app(
        args.session,
        runs_root=args.runs_root,
        token=args.token,
        static_dir=args.static_dir,
        checkpoint=args.checkpoint,
        train_device=args.train_device,
    )
    print(f"Strike Pose Editor: http://{args.host}:{args.port}/", flush=True)
    print(f"Mutation token: {app.state.mutation_token}", flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
