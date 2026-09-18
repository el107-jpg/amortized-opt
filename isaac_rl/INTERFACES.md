# `isaac_rl` public interfaces

Contract for downstream consumers (currently `amortized/isaac_rl_ppo/`, which
imports these modules **read-only** and adds only its own update rule).

**GPU allocation:** this session (RL-1 / RL-1b / **RL-2**) uses **`cuda:1`**
exclusively. `cuda:3` is the PPO session / waist fine-tune. `cuda:0` and
`cuda:2` are held by other sessions and must not be used.

### The device trap — use `CUDA_VISIBLE_DEVICES`, not `--device` alone

`SimulationCfg.device` / `--device cuda:N` steers **PhysX**. It does *not* steer
**Warp** or several `isaaclab` extensions, which initialise on **`cuda:0`**
regardless. On a box where `cuda:0` is full, that shows up as a Warp OOM
followed by a misleading `Failed to get DOF velocities` out of the PhysX tensor
API (reported by the PPO session). The robust recipe, adopted for every run in
this package:

```bash
CUDA_VISIBLE_DEVICES=1 python isaac_rl/scripts/<entry>.py --device cuda:0 ...
```

i.e. **mask the GPU at the driver level and always ask for `cuda:0` inside**.
`torch.cuda.set_device` still runs in every entry point; that is necessary and
not sufficient.

### PSN-vs-PPO A/B protocol — executed-outcome metrics only

The PSN arm (this package's `train_psn.py`) and the PPO arm
(`amortized/isaac_rl_ppo/`) have deliberately different reward internals — the
PSN return mixes analytic `FusedCost` terms and a tracking term that the PPO
arm shapes differently, so `reward/total` is **not comparable across arms**.
Compare only the executed-outcome metrics, on identical held-out clips and
seeds:

| metric | source |
| --- | --- |
| miss rate | `measure.latency_report()["overall"]["miss_rate"]` |
| \|latency\| p50 / p90 | same, `lat_abs_p50_ms` / `lat_abs_p90_ms` |
| speed-in-band | same, `speed_in_band` (band `[1.0, 3.5] m/s`) |
| collisions | `RewardTerms.raw["plan.collision"]` + executed SDF penetration |
| open-loop grid | `scripts/eval_phase2.py --grid-seed 7 --skip-closed-loop` |

Both arms must publish the clip list and the seed list they used.

**Stability legend**

| mark | meaning |
| --- | --- |
| **STABLE** | will not change without a changelog entry + migration note |
| **PROVISIONAL** | may still change before the RL-1 smoke test is signed off |

Status as of the changelog head (RL-1b): the Isaac smoke test has now run to
completion three times (ff OFF / ff ON x analytic lead / ff ON x Isaac lead), so
`env.DrumRLEnv.plant` and `safety` are **STABLE**. The reward *keys* are STABLE;
the reward **weights** stay PROVISIONAL until the training stage tunes them.
51 Isaac-free pytest conformance tests are green.

---

## Rate convention — `isaac_rl.env_cfg` — **STABLE**

```python
SIM_DT: float = 1.0 / 250.0      # physics step
DECIMATION: int = 5              # physics substeps per control tick
CONTROL_DT: float = 0.02         # = SIM_DT * DECIMATION, a true 50 Hz control period
CONTROL_HZ: float = 50.0
assert_rate_consistency() -> None
```

Resolved decision 1 (2026-08-10). These are **owned here**, deliberately NOT
imported from `g1_drummer.control.control_rate` (which is 1/120 & decimation 2,
and is also read by the repo's offline AMP/teacher/BC generators and by
`qp_filter.SafetyFilter`'s default `dt` — mutating it would reshard every
dataset in the main repo). `assert_rate_consistency()` runs at import and pins
`SIM_DT * DECIMATION == amortized_opt.CONTROL_DT`, i.e. **one waypoint = one
control tick**. Do not break that identity: every horizon in the stack
(`hit_index`, element masks, Fourier time codes, the reference lead expressed
in waypoint fractions) assumes it, and a mismatch raises nowhere.

```python
@dataclass
class DrumRLEnvCfg:                       # plain dataclass, Isaac-free
    num_envs: int = 256
    env_spacing: float = 4.0
    pin_pelvis: bool = True
    freeze_legs: bool = True
    use_ideal_pd: bool = True
    pd_stiffness_scale: float = 1.0
    pd_damping_scale: float = 4.0         # G1_PD_DAMPING_SCALE; arms Kd 4, waist-yaw Kd 20
    episode_length_s: float = 12.0
    rhythm_split: str = "eval"            # "eval" | "train"
    rhythm_dir: str | None = None
    random_clip_offset: bool = False
    horizon: int = 50
    iterations: int = 1
    cmd_lead_s: str | float | tuple = "fit"
    hit_lead_s: float = 0.0
    freeze_wp: int = 0
    hit_mode: str = "geometric"           # "geometric" | "contact"  (DECISION 1)
    contact_force_threshold: float = 1.0
    contact_sensors: bool = True
    gravity_feedforward: bool = True
    foh_ramp: bool = True
    randomize_kit: bool = False
    lag_substeps: tuple[int, int] = (0, 0)
    restitution_range: tuple[float, float] | None = None
    friction_range: tuple[float, float] | None = None
    device: str = "cuda:1"
    seed: int = 0
    record: bool = False
    # properties: sim_dt, decimation, control_dt
    def knob_table(self) -> list[tuple[str, str, str]]
```

`build_env_cfg(**kw) -> DrumRLEnvCfg`.

**Checkpoint paths are NOT in the env cfg.** The model checkpoint is a script
argument (`--ckpt`, default `runs/phase2b_main/best.pt`) and is loaded by the
caller, which then constructs `IsaacMPC` with the live `nn.Module`. Nothing in
`isaac_rl` reads a checkpoint path from a config object.

---

## `isaac_rl.env` — **STABLE**

```python
build_isaac_cfg(cfg: DrumRLEnvCfg) -> DirectRLEnvCfg   # requires a running Isaac app
make_env(cfg: DrumRLEnvCfg) -> DrumRLEnv               # ditto
```

`DrumRLEnv` subclasses `isaaclab.envs.DirectRLEnv` but is **not** driven through
`gym`. Its `_get_observations` / `_get_rewards` / `_get_dones` are deliberately
dead stubs (the RL here is episodic parameter-space search, not PPO). Consumers
use:

```python
env.read_state() -> (q, qd)                    # (B, 17) each, G1_UPPER_BODY_JOINTS order
env.plant(prev_cmd, cmd, q, qd) -> (q, qd)     # ONE control tick = DECIMATION substeps
env.reset_to(env_ids, q, qd=None) -> None
env.contact_forces() -> (B, 2, D, 3) | None    # hand axis 0=left; filter index == drum id
env.joint_ids: list[int]                       # articulation indices, preserve_order
env.contact_drum_names: tuple[str, ...]
```

`plant`'s signature is deliberately identical to
`amortized_opt.pd_sim.BatchedPDSim.step`, so the same `DrumMPCLoop` runs against
either plant. **A PPO variant that wants a per-step `gym` surface must wrap
`plant`, not override it.**

---

## `isaac_rl.schedule` — **STABLE** (15 conformance tests green)

```python
@dataclass
class ScheduleConfig:
    dt: float = CONTROL_DT; horizon: int = 50
    num_drums: int = 8; num_future: int = 3
    horizon_cap_wp: int | None = None
    hit_radius_frac: float = 0.6; speed_min: float = 1.0; speed_max: float = 3.0
    hit_lead_s: float = 0.0
    randomize_kit: bool = False; kit_randomization: KitRandomization | None = None
    max_hits: int = 256
    cap_wp: int                                 # property

class BatchedSchedule:
    def __init__(self, n_envs, cfg, *, device, dtype=torch.float32)
    def reset(self, env_ids, clips: list[RhythmClip], kits: list[DrumKit], *,
              t_start=0.0, rng=None, clip_ids=None) -> None
    def context(self) -> ContextInputs           # amortized_opt.tokenizer
    def hit_index(self) -> LongTensor            # (B, 2), -1 = idle
    def strike_context(self, hit_index=None) -> StrikeContext   # amortized_opt.cost
    def advance(self) -> BoolTensor              # (B, 2) fired mask; steps the clock
    def scheduled_tick(self) -> LongTensor       # (B, 2)
    def done(self) -> BoolTensor                 # (B,)
    t_now: FloatTensor                           # (B,) float64 PROPERTY, = t0 + tick*dt
    tick, t0, cursor, hit_t, hit_drum, hit_pos, hit_vel, hit_valid
    drum_pos, drum_axis, drum_radius, drum_height, drum_valid, geom
    start_drum, start_ticks, departure_exempt_wp
```

`clip_hit_table(clip, kit, rng, *, cfg) -> dict[str, np.ndarray]`.

Invariants pinned by tests, do not "simplify" them away:
- the clock is **float64 derived from an integer tick count**, never accumulated
  (float32 accumulation of 0.02 flipped a boundary comparison at t = 1.70 s and
  left the cursor one hit behind for the rest of an episode);
- the clip is queried **strictly forward** of `t_now` (`SNAP_EPS = 1e-6`), so a
  hit at exactly `t_start` never fires;
- `hit_lead_s` shifts the **query**, not `time_to_hit`, so every hit fires
  exactly once at any lead;
- an out-of-range goal is **demoted to `future[0]`**, not dropped.

---

## `isaac_rl.mpc` — **STABLE** except `TickResult.aux` contents

```python
@dataclass
class MPCConfig:
    horizon: int = 50; dt: float = CONTROL_DT; iterations: int = 1
    cmd_lead_s: str | float | tuple | None = "fit"
    cmd_slew_eta: float | None = 1.0
    clamp_cmd_to_box: bool = True
    freeze_wp: int = 0; freeze_waist: bool = True
    encode_every_tick: bool = False             # see DECISION 2 (resolved)
    enc_yaw_tol: float = 0.05                   # rad; re-encode on a yaw-BC move
    amp_dtype: torch.dtype | None = torch.bfloat16
    joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS
    yaw_bins: int = 129

@dataclass
class TickResult:
    cmd: Tensor          # (B, 17) executed command
    traj: Tensor         # (B, H, 17) plan, with cmd written into row 0
    elem_mask: Tensor    # (B, H, 17)
    hit_index: LongTensor    # (B, 2)
    active: BoolTensor       # (B, 2)
    aux: dict            # PROVISIONAL keys: vel, acc, torque, fk_pos, fk_vel,
                         #                   point_mask, token_mask
    frozen: BoolTensor   # (B,)
    encoded: bool

class IsaacMPC:
    def __init__(self, model, schedule: BatchedSchedule, cfg=None, *, device,
                 dtype=torch.float32, library=None, kit_names=None)
    def reset(self, env_ids, q0) -> None
    def masks(self, hit_index) -> WaypointMasks
    def plan(self, q, qd) -> TickResult          # does NOT step the clock
    def commit(self, cmd, q, fired) -> None      # call AFTER plant + schedule.advance
    last_ref0: Tensor    # (B, 17) the plan's UN-LED first waypoint (tracking reference)
    prev_cmd: Tensor; init: Tensor
    encode_fraction: float
```

Call order per tick is load bearing: `plan` → (safety) → `plant` →
`schedule.advance` → reward → `commit`. Computing the reward after `commit`
prices the *next* strike's geometry against the pose reached for the previous one.

---

## `isaac_rl.reward` — keys **STABLE**, weights **PROVISIONAL**

```python
@dataclass
class RewardConfig:
    w_hit=10.0, w_timing=5.0, timing_tau_s=0.02, w_speed=2.0,
    speed_lo=1.0, speed_hi=3.5, w_miss=5.0, window_ticks=5,
    w_extra_hit=5.0, w_early_contact=2.0,
    w_plan_cost=0.05, plan_terms=(...) | None,
    w_track=1.0, track_scale_rad=0.05, w_filter=0.0, w_alive=0.0,
    joint_names=G1_UPPER_BODY_JOINTS

@dataclass
class RewardTerms:
    total: Tensor                 # (B,)
    terms: dict[str, Tensor]      # keys below -- STABLE set, values PROVISIONAL
    raw: dict[str, Tensor]
    counts: dict[str, Tensor]
    def log(self) -> dict[str, float]   # "reward/total", "terms/*", "raw/*", "counts/*"
```

`terms` keys (**STABLE set**): `hit`, `timing`, `speed`, `miss`, `extra_hit`,
`early_contact`, `plan_cost`, `track`, `filter`, `alive`.
`counts` keys: `scheduled_hits`, `fired`, `missed`, `extra_hits`,
`early_contacts`, `filter_interventions`.
`raw` keys: `hit_latency_s`, `contact_speed`, `track_rms_rad`,
`plan_cost_total`, `plan.<FusedCost term>`, `viol.<group>`.

```python
@dataclass
class ContactScore:
    crossed, in_disk: BoolTensor       # (B, 2)
    frac, speed, axial_m, inplane_m: Tensor   # (B, 2)
    drum: LongTensor                   # (B, 2)

class ExecutedStrikeScorer:
    def __init__(self, *, fk: ChainFK, dt=CONTROL_DT, device=None, dtype=torch.float32)
    def contact_points(self, q, sigma) -> (pts (B,2,3), dirs (B,2,3))
    def head_frame(self, pts, ctx: StrikeContext) -> (axial (B,2), inplane (B,2))
    def score(self, q0, q1, ctx) -> ContactScore
    def any_drum_contact(self, q0, q1, heads, axes, radii, valid, sigma) -> (B,2,D) bool

class ExecutedReward:
    def __init__(self, cfg, *, scorer, cost: FusedCost | None = None,
                 device=None, dtype=torch.float32)
    def step(self, *, q_before, q_after, qd_after, ctx, hit_index, fired, ref0,
             traj=None, elem_mask=None, q_plan0=None, v_plan0=None,
             drum_heads=None, drum_axes=None, drum_radii=None, drum_valid=None,
             filter_active=None) -> (RewardTerms, ContactScore)
```

Conventions: `axial > 0` = short of the head; `inplane > 0` = outside the legal
disk edge — identical to `amortized_opt.probe.head_frame_error`, so the numbers
are directly comparable to `REPORT.md` §4 and README §7c. `contact_sigma` is the
arc coordinate from the stick's **centre** (`STICK_LENGTH/2` = the tip), matching
`FusedCost._terminal` and `fk.stick_contact_offset`.

---

## `isaac_rl.perturb` — **STABLE** (16 tests green)

```python
@dataclass
class NoiseConfig:
    mode: str = "film"        # "film" | "lora"
    rank: int = 8; sigma: float = 0.0; antithetic: bool = True
    blocks: tuple[int, ...] | None = None
    lora_sites: tuple[str, ...] = ("attn", "cross", "ffn")
    seed: int = 0

@dataclass
class AdaptiveSigmaConfig:
    target_mm=3.0, adapt_rate=1.01, scale_min=1e-3, scale_max=1e3, ema=0.9

class AdaptiveSigma:
    scale: float; ema_mm: float | None; n_obs: int
    def observe(self, distance_mm: float) -> float
    def state_dict(self) / load_state_dict(d)

@dataclass
class ParameterNoise:
    @classmethod build(cls, model, n_envs, cfg, *, device=None, dtype=torch.float32)
    def attach(self) / detach(self)
    def redraw(self, envs: Tensor | None) -> None     # None = all
    def epsilon(self) -> Tensor                       # (n_envs, dim) NES coordinates
    def set_sigma(self, sigma, envs=None) -> None     # 0 = clean policy on those envs
    def observe_deviation(self, distance_mm: float) -> float
    n_envs: int; dim: int

build_perturbation(model, n_envs, cfg, *, device, dtype) -> FiLMPerturbation | LoRAPerturbation
antithetic_normal(n, shape, *, generator, device, dtype, antithetic) -> Tensor
deviation_report(traj_clean, traj_pert, *, elem_mask=None, hit_index=None,
                 joint_names=None, tip_clean=None, tip_pert=None) -> dict
```

Attaches via `nn.Module` forward hooks on the **decoder only** — the checkpoint
is untouched and the encoder is never perturbed. `sigma = 0` is bit-identical to
the unperturbed model (tested).

---

## `isaac_rl.safety` — **STABLE** (implemented and measured)

```python
@dataclass
class SafetyConfig:
    mode: str = "box"            # "box" | "qp" | "off"
    margin: float = 0.005
    d_activate: float = 0.10     # QP row culling only
    eta_vel: float = 1.0         # slew = eta_vel * isaac_velocity_limits() * dt
    dt: float = CONTROL_DT
    sdf_steps: int = 2           # projected-gradient clearance steps; 0 disables
    sdf_threshold: float = 1e-4
    sdf_max_step: float = 0.05
    sdf_relax: float = 1.0
    sdf_include_self: bool = True
    sdf_compile: str = "reduce-overhead"     # | "default" | "off"
    limit_margin: float = 0.0; posture_limits: bool = True; tol: float = 1e-6
    qp_all: bool = False; qp_k_pool: int = 16; qp_use_braking: bool = True
    qp_eta_brake: float = 0.5; qp_include_self: bool = True
    qp_osqp_settings: dict | None = None

class BatchedSafetyFilter:
    def __init__(self, cfg, *, joint_names=G1_UPPER_BODY_JOINTS, device,
                 dtype=torch.float32, layout_path=None)
    def reset(self, env_ids, q) -> None      # slew reference + per-env QP warm start
    def project(self, cmd, q, qd, *, target_drum=None) -> (safe (B,17), engaged (B,) bool)
    stats: dict    # mode, calls, env_steps, engaged, engaged_frac, box, slew, sdf,
                   # qp_solves, qp_active, qp_fallback, sdf_compiled
    def reset_stats(self) -> None
    mode: str      # the mode ACTUALLY running ("qp" demotes to "box" on import failure)
```

`target_drum` is `(B, 2)` drum ids `[left, right]`, `-1` = none (a `(B,)` tensor is
broadcast to both hands). Box order: joint box -> slew vs `prev_cmd` -> projected
SDF steps re-clamped into box n slew. Branch-free, no host sync; counters
accumulate on-device and only `stats` syncs.

Measured at B=256 on cuda:1: **8.9-9.4 ms/call** with defaults (box + slew + 2
compiled clearance steps), **0.59 ms** with `sdf_steps=0`. Eager clearance is
~54 ms/step and flat from B=64 to B=2048 (kernel-launch bound, not FLOPs), which
is why `sdf_compile="reduce-overhead"` is the default. `mode="qp"` is LIVE on
this box (pinocchio + hppfcl present) at ~76 ms for 19 engaged envs of 32 --
eval-only rates.

**Callers must call `filt.reset(env_ids, q)` on every env reset**, or the first
post-reset tick is spuriously slew-limited against a stale `prev_cmd`.

---

## `isaac_rl.rnea_cache` — **STABLE** (11 tests green)

Pinocchio cannot expose `model.parents` inside an Isaac Sim process on this box,
so `BatchedRNEA.from_urdf` — and therefore the gravity/Coriolis feedforward,
`FusedCost`, and `TrajectoryScorer` — could not be built there. The kinematic
tree is a build-time constant of the URDF, so it is frozen to an `.npz` in a
normal process and reconstructed inside Isaac with no Pinocchio at all.

```python
CACHE_PATH: Path                      # isaac_rl/rnea_cache.npz (committed, 5 KiB)
cache_fingerprint(joint_names=..., urdf_path=None, with_armature=True) -> str
build_cache(path=None, *, joint_names=..., urdf_path=None, with_armature=True) -> Path
load_cached(path=None, *, device="cpu", dtype=torch.float32, joint_names=...,
            urdf_path=None, with_armature=True, check=True) -> BatchedRNEA
build_or_load(*, device="cpu", dtype=torch.float32, path=None, joint_names=...,
              urdf_path=None, with_armature=True, verbose=True) -> BatchedRNEA | None
@contextmanager
rnea_from_cache_if_needed(*, verbose=True)   # patches BatchedRNEA.from_urdf
```

* **Verified bit-exact**: max `|tau_live - tau_cached|` = **0.0e+00** over 512
  random states, and 0.0e+00 for the gravity-only term the feedforward uses.
* `build_or_load` tries the **live** build first, so a non-Isaac process never
  depends on the cache being fresh; the cache is the Isaac-only fallback. It
  returns `None` only if both fail — callers must log that loudly, because a
  silently missing feedforward changes the plant.
* The cache carries a URDF sha256 + joint-set + `BUILD_ID` fingerprint and a
  **stale cache raises** rather than being used.
* `rnea_from_cache_if_needed()` exists for consumers that build their own RNEA
  internally and expose no injection point — `TrajectoryScorer` is the one that
  matters, since `probe.hit_latency` runs on it.

Rebuild with `python isaac_rl/scripts/build_rnea_cache.py` (outside Isaac);
it re-verifies against the live build and prints `CACHE_OK`.

**Gotcha for consumers:** do **not** pass a shared `ChainFK` into `FusedCost`.
Its posture term reads `frames["left_shoulder"]`, which the default `ChainFK`
frame set does not carry, and the `KeyError` only fires when a cost is actually
evaluated. Construct it as `A.FusedCost(cfg, rnea=rnea, ...)` and let it build
its own FK with `FusedCost.FRAMES`.

---

## The ISAAC-fitted reference lead — **PUBLISHED**, use this one

`amortized_opt.lead.LEAD_PRESETS["fit"]` was identified against
`amortized_opt.pd_sim` (an analytic FOH servo model), **not** against PhysX.
RL priority 3 requires the executed lead to be the one fitted against the plant
that is actually running. The Isaac fit now exists:

| | |
| --- | --- |
| **file** | `amortized/runs/rl1b/lead_isaac.json` |
| **key** | `best_strike_s` (17 floats, seconds, `G1_UPPER_BODY_JOINTS` order) |
| also in file | `best_all_s`, `rms_at_0_strike_mrad`, `rms_at_best_strike_mrad`, `analytic_fit_s` |
| produced by | `python isaac_rl/scripts/fit_lead_isaac.py --headless --device cuda:1` |
| conditions | 162-case `standard` grid, champion `runs/phase2b_main/best.pt`, sim_dt 0.004 / dec 5, gravity ff ON, sweep 0–50 ms @ 2.5 ms, strike window 6 knots |

```python
ISAAC_LEAD_S = (0.005, 0.045, 0.045, 0.045, 0.0125, 0.045, 0.0, 0.0325, 0.02, 0.045, 0.02, 0.0, 0.005, 0.015, 0.015, 0.05, 0.05)
```

Consume it either way:

```python
# from the JSON (preferred -- one source of truth)
import json
lead = tuple(json.load(open("runs/rl1b/lead_isaac.json"))["best_strike_s"])
mpc = IsaacMPC(model, sched, MPCConfig(horizon=H, cmd_lead_s=lead), ...)

# or via the smoke script's loader
python isaac_rl/scripts/smoke_rl.py --lead runs/rl1b/lead_isaac.json ...
```

`MPCConfig.cmd_lead_s` accepts a preset name, a float, or this 17-vector;
`smoke_rl.py --lead <path.json>` reads `--lead-key` (default `best_strike_s`)
out of it and records `lead_source` / `lead_s` in its report.

**It is NOT a small correction to the analytic vector** — see the group table in
`isaac_rl/README.md` §8. The wrists move from 0.0 ms (analytic: "flat, a lead
cannot touch them") to a 38.8 ms median with 20–32 % tracking gains, which
inverts §7c's per-group reading on this plant.

---

## `isaac_rl.loop` — **STABLE**

```python
class DrumMPCLoop:
    def __init__(self, *, schedule, mpc, reward, plant, noise=None, safety=None,
                 record=False)
    def tick(self, q, qd) -> (q, qd, RewardTerms, TickResult, ContactScore)
    def rollout(self, q, qd, steps) -> (q, qd, dict[str, float])
    def q_hist(self) -> np.ndarray            # (T, B, 17)
    def fired_events(self) -> list[(tick, env, hand, drum)]
    stats: LoopStats;  history: list[TickRecord]

@dataclass
class LoopStats:
    ticks, t_plan, t_filter, t_plant, t_reward, t_total, filter_interventions
    def summary(self, n_envs, dt=CONTROL_DT) -> dict
      # env_fps, realtime_factor, share_plan/filter/plant/reward, ms_per_tick_*
```

`plant` is any callable `(prev_cmd, cmd, q, qd) -> (q, qd)`.

---

## `isaac_rl.measure` — **STABLE**

```python
@dataclass
class EpisodeView: q_hist: np.ndarray; hit_steps: list
per_env_latency(loop, kit, *, scorer=None, window_ticks=5, dt=CONTROL_DT) -> list[dict]
latency_report(loop, kit, *, device="cpu", window_ticks=5, dt=CONTROL_DT) -> dict
    # {"overall": summarize_latency(...), "n_strikes": int,
    #  "by_hand": {...}, "by_drum": {...}}
```

Wraps `amortized_opt.probe.hit_latency` **verbatim** — the same function that
produced `REPORT.md` §4's 63 % miss rate — so any number out of here is directly
comparable to it. Do not reimplement the crossing search downstream.

---

## Scripts (entry points, not a library API)

| script | purpose |
| --- | --- |
| `isaac_rl/scripts/smoke_rl.py` | the RL-1 smoke test; `--ckpt --device --envs --steps --sigma --safety --out` |
| `isaac_rl/scripts/ref_loop.py` | the same loop with the ANALYTIC plant — the control experiment for the Isaac numbers |
| `isaac_rl/scripts/fit_lead_isaac.py` | re-fit the per-joint reference lead against Isaac's actuator |

---

## Changelog

| date | change | migration |
| --- | --- | --- |
| 2026-08-11 | Initial contract. `schedule`, `perturb`, `loop`, `measure`, `mpc` STABLE (31 pytest conformance tests green); `env`, `reward` weights, `safety` PROVISIONAL pending the Isaac smoke test. Session GPU: **cuda:1**. | n/a |
| 2026-08-11 | `BatchedSchedule.t_now` changed from a stored float32 buffer to a **float64 property** `t0 + tick * dt`; new fields `tick`, `t0`; `hit_t` and `duration` are now float64. | Read `sched.t_now` (property, unchanged name); do not assign to it. Advance the clock only via `advance()`. |
| 2026-08-11 | **RL-1b.** New module `isaac_rl.rnea_cache` + committed `rnea_cache.npz` (bit-exact, 0.0e+00). `env`/`safety` promoted to STABLE. **ISAAC-fitted lead published** at `runs/rl1b/lead_isaac.json[best_strike_s]` — use it instead of `LEAD_PRESETS["fit"]`. `smoke_rl.py` gained `--lead <path.json>` / `--lead-key`, and now exits via `os._exit` (Isaac's `app.close()` hangs in the carb mutex and was serialising queued runs behind a 28-min timeout). | Consumers building `FusedCost` must NOT pass a shared `ChainFK` (needs shoulder frames) — pass `rnea=` only. Anything that hard-codes `LEAD_PRESETS["fit"]` for Isaac should switch to the JSON. |
| 2026-08-12 | **RL-2 bug-fix batch** (all four reported by the PPO session). (a) **`DrumMPCLoop.tick` now unpacks `out[0], out[1]`** from the plant instead of `q, qd = plant(...)`, so `BatchedPDSim.step`'s new 3-tuple `(q, qd, integral)` can be handed to the loop directly; regression test `amortized/tests/test_isaac_rl_loop.py` (3 tests, main suite, Isaac-free) drives the loop with the raw `sim.step`. (b) `smoke_rl.py` built a stray unused `SdfPenalty` next to the `FusedCost` construction — **removed**; `FusedCost` takes no `sdf=` kwarg and never received one (the reported call site was `ref_loop.py`'s `StartSampler(sdf=...)`, which is correct — `StartSampler` *does* take `sdf`). (c) `rnea_cache.build_or_load` is confirmed the **only** RNEA entry inside `isaac_rl` (`env.try_build_rnea` delegates to it; `measure` wraps the non-injectable `TrajectoryScorer` in `rnea_from_cache_if_needed`), and it gained **`AMORTIZED_RNEA_CACHE_ONLY=1`** to skip the live Pinocchio attempt entirely. (d) new **device-trap** and **PSN-vs-PPO A/B protocol** sections at the top. | Callers passing an adapter lambda solely to drop the integral may now pass `sim.step` directly (the lambda still works). Set `AMORTIZED_RNEA_CACHE_ONLY=1` in Isaac entry points if the live build ever crashes rather than raising. Launch with `CUDA_VISIBLE_DEVICES=<gpu> ... --device cuda:0`. |
| 2026-08-11 | **DECISION 5 (time-invariant perturbation subspace) — WITHDRAWN, deferred to RL-2 as "revisit only if σ tuning stalls".** The RL-1 finding (deviation uniform along the horizon) was an artifact of probing from a gravity-sagged state with the feedforward off. With ff ON the FiLM deviation is 65–66 % of its energy in 11 % of the grid (~6× contact concentration) at both σ and both mechanisms. No API change; `NoiseConfig` is unchanged. | none |
