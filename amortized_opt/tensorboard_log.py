"""TensorBoard scalar logging for phase-2 ``metrics.jsonl`` records."""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

_LOSS_KEYS = frozenset({"loss", "cost", "aux", "tc", "bptt"})

_SKIP_KEYS = frozenset({
    "event", "cfg", "model_cfg", "scenario_cfg", "pool", "pool_stats",
    "world_size", "replay", "post_hit", "replay_short", "batch_short",
    "roller", "frontier", "evals", "last_probe",
    "ls_seed_provenance", "torque_physics", "static_torque_tighten",
    "torque_static_only", "torque_al_off", "torque_static_al",
    "w_torque_static", "eta_torque", "stage_a_open", "error",
    "elapsed_s", "steps_per_s", "lr", "grad_norm",
})

_SCHEDULE_KEYS = frozenset({
    "alpha", "alpha_tau_peak", "alpha_vel_peak", "alpha_coll_peak",
    "alpha_jbox_peak", "alpha_waist", "alpha_torque_full",
})

_EVAL_RATE_KEYS = frozenset({
    "ok_hit", "ok_speed", "ok_collision", "ok_torque", "ok_joint_limits",
    "acceptance", "elbow_down", "stage_a",
})

_EVAL_COUNT_KEYS = frozenset({
    "accepted", "n", "short_accepted", "short_n",
})


def _is_scalar(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    return isinstance(v, numbers.Real) and not isinstance(v, str)


def _diag_section(name: str) -> str:
    if name in {
        "d_region", "d_pos_err", "d_speed_err",
        "tcf_l", "tcm_l", "tcn_l", "tce_l", "tcx_l",
        "tcf_r", "tcm_r", "tcn_r", "tce_r", "tcx_r",
    }:
        return "diagnostics/hit"
    if name.startswith("d_c_") or name in {
        "d_min_sdf", "d_stick_sdf", "d_tip_lat", "d_tip_cross",
    }:
        return "diagnostics/collision"
    if name.startswith("d_tau") or name == "d_vel_excess":
        return "diagnostics/torque"
    if name.startswith("d_waist") or name in {"d_box_excess", "d_elbow"}:
        return "diagnostics/waist"
    if name == "d_jerk":
        return "diagnostics/smooth"
    return "diagnostics/other"


def _train_scalar_tag(key: str) -> str:
    if key in _LOSS_KEYS:
        return f"loss/{key}"
    if key.startswith("c_"):
        return f"terms/{key[2:]}"
    if key.startswith("v_"):
        return f"violations/{key[2:]}"
    if key in _SCHEDULE_KEYS:
        return f"schedule/{key}"
    if key.startswith("ls_"):
        return f"line_search/{key}"
    if key.startswith("grad_"):
        return f"grad/{key[5:]}"
    if key.startswith("gb_"):
        return f"grad_balance/{key[3:]}"
    if key.startswith("phase_"):
        return f"performance/{key.removeprefix('phase_')}"
    if key.startswith("d_") or key.startswith(("tcf_", "tcm_", "tcn_", "tce_", "tcx_")):
        return f"{_diag_section(key)}/{key}"
    if key.startswith("waist_") or key.startswith("pr_distill"):
        return f"waist/{key}"
    if key in {"sfx_cost", "sfx_tie", "mpc_init_tie", "mpc_init_frac"}:
        return f"aux/{key}"
    return f"other/{key}"


def _eval_scalar_tag(key: str) -> str:
    if key in _EVAL_RATE_KEYS:
        return f"eval/rates/{key}"
    if key in _EVAL_COUNT_KEYS:
        return f"eval/counts/{key}"
    if key.endswith("_p50") or key.endswith("_p90") or "err" in key or key in {
        "tau_excess_p90", "jerk_p50", "pen_max_mm_p90",
        "contact_pos_err_mm_p50", "speed_err_p50",
    }:
        return f"eval/errors/{key}"
    return f"eval/other/{key}"


def _probe_scalar_tag(key: str) -> str:
    return f"probe/{key}"


def scalars_from_record(rec: dict) -> list[tuple[str, float, int]]:
    """Flatten one metrics record into TensorBoard ``(tag, value, step)``."""
    event = rec.get("event")
    step = int(rec.get("step", 0))
    if step <= 0 or event not in ("train", "eval", "probe"):
        return []

    out: list[tuple[str, float]] = []

    def add(tag: str, val: Any) -> None:
        if _is_scalar(val):
            out.append((tag, float(val)))

    if event == "train":
        add("optimizer/lr", rec.get("lr"))
        add("optimizer/grad_norm", rec.get("grad_norm"))
        add("optimizer/steps_per_s", rec.get("steps_per_s"))
        for g, val in (rec.get("lam_i") or {}).items():
            add(f"al/lambda/{g}", val)
        for g, val in (rec.get("pen") or {}).items():
            add(f"al/pen/{g}", val)
        for g, val in (rec.get("lam") or {}).items():
            add(f"al/weight/{g}", val)
        for key, val in (rec.get("timing_critic") or {}).items():
            add(f"timing_critic/{key}", val)
        for key, val in (rec.get("roller") or {}).items():
            if key.startswith("timing_"):
                add(f"timing_curriculum/{key.removeprefix('timing_')}", val)
            elif key.startswith("dls_terminal_"):
                add(f"terminal_dls/{key.removeprefix('dls_terminal_')}", val)
            elif key.startswith("curvature_teacher_"):
                add(f"terminal_curvature/{key.removeprefix('curvature_teacher_')}", val)
            elif key.startswith("curvature_direct_"):
                add(f"curvature_direct/{key.removeprefix('curvature_direct_')}", val)
        for key, val in rec.items():
            if key in _SKIP_KEYS or key in ("event", "step", "lam", "lam_i", "pen"):
                continue
            if isinstance(val, dict):
                continue
            add(_train_scalar_tag(key), val)
    elif event == "eval":
        for key, val in rec.items():
            if key in _SKIP_KEYS or key in ("event", "step"):
                continue
            if isinstance(val, dict):
                continue
            add(_eval_scalar_tag(key), val)
    else:  # probe
        for key, val in rec.items():
            if key in _SKIP_KEYS or key in ("event", "step"):
                continue
            if isinstance(val, dict):
                continue
            add(_probe_scalar_tag(key), val)

    return [(tag, val, step) for tag, val in out]


class MetricsTensorboard:
    """Optional TensorBoard writer for :func:`train_phase2` metrics."""

    def __init__(self, log_dir: Path, *, enabled: bool = True) -> None:
        self._writer = None
        self._enabled = bool(enabled)
        self.log_dir = Path(log_dir)
        if not self._enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            import warnings

            warnings.warn(
                "tensorboard not installed; disabling TensorBoard logging. "
                "pip install 'tensorboard>=2.14' or pip install -e '.[monitoring]'",
                stacklevel=2,
            )
            self._enabled = False
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_record(self, rec: dict) -> None:
        if self._writer is None:
            return
        for tag, val, step in scalars_from_record(rec):
            self._writer.add_scalar(tag, val, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
