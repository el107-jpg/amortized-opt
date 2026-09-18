# `isaac_rl_ppo` — executed-prefix PPO for the amortized optimizer

The **PPO variant** of the RL fine-tune. The sibling `amortized/isaac_rl/` is
the primary design (episodic parameter-space noise); this package is the
documented fallback, built so the two can be raced on the same plant, the same
reward and the same checkpoint.

It imports `isaac_rl` **read-only** through one adapter (`bridge.py`) and adds
only its own actor and update rule.

```
bridge.py       the ONLY module that touches isaac_rl; the compatibility contract
cfg.py          one dataclass tree + the checkpoint-resolution rules
start.py        play_isaac start pose + first-hit pad (used by train_ppo)
actor.py        LoRA + Gaussian head + AR(1) + the MLP critic (rsl_rl-shaped)
ppo_loop.py     storage, GAE, the fused update, and the rollout driver
train_ppo.py    the entry point (JSONL metrics, checkpoints, resume)
tests/          50 tests, none of which need Isaac
{env,mpc,...}.py   a PINNED SNAPSHOT of isaac_rl, for diffing and as a fallback
```

---

## The design in one page

**The action is the executed first waypoint — 17 dims — not the 50×17
trajectory.** Everything else follows from that.

One control tick:

1. The pretrained transformer runs its MPC pass as a **deterministic feature
   extractor**: warm start (previous plan advanced one tick, arm columns
   re-seeded when a hit completes) + measured state + context → one refinement →
   a 50×17 plan. This is `IsaacMPC.plan`, unchanged.
2. A Gaussian head over the plan's first waypoint: `mean = traj[:, 0]` plus a
   zero-initialised linear correction read off the decoder's waypoint-0 hidden
   state; `std` is a learned state-independent per-dim parameter (init 0.45,
   clamped to `[0.02, 1.0]` on the forward path). Log-prob and entropy are stock
   `Normal` — **the PPO math is exact and untouched.**
3. At update time the analytic `FusedCost` on the full plan and the PPO
   surrogate are combined in **trajectory space** at waypoint 0: always-project
   the PPO gradient onto the cost nullspace, **rescale cost per sample to the
   PPO gradient's magnitude**, then add the leftover at PPO's native scale.
   `--orth-leak α` re-injects fraction `α` of the *parallel* PPO component at
   waypoint 0 only (default 0 = hard orth). Unscaled cost (aux ~4000) railed
   KL and NaNed the first Isaac run.
   Waypoints `1..H-1` are cost-only — they are the next tick's warm start.
   `--no-orth` restores the previous scalar sum.
4. Exploration noise is AR(1)-correlated in time (per-joint-group τ: waist
   0.30 s, shoulder 0.25 s, elbow 0.15 s, wrist 0.10 s), variance-preserving so
   the marginal stays exactly `N(mean, std)`.
5. The sample becomes a command through the deployed path — the reference lead
   (**OFF in Isaac**, see below), the slew bound, the joint-box clamp — with the
   noise added as an additive joint-space offset *after* the lead.

### The lead is regime-dependent, and each regime has its own fitted vector

`EnvConfig.plant_regime` is an explicit field because the two plants need
**different** vectors and using the wrong one is silent:

| regime | plant | resolved `cmd_lead_s` |
| --- | --- | --- |
| `analytic` | `amortized_opt.pd_sim` | `"fit"` — identified against exactly this servo |
| `isaac` (**default**) | PhysX | `runs/rl1b/lead_isaac.json[best_strike_s]` — RL-1b's Isaac fit |

`PpoRunConfig.resolve_lead()` loads the Isaac vector **from the published JSON
rather than embedding the numbers**, so a re-fit propagates by replacing a file.
`best_strike_s` (not `best_all_s`) is the default key: it is fitted on the
6-knot strike window, which is the interval the reward actually scores.

**This is not a small correction to the analytic vector.** The wrists go from
0.0 ms on the analytic plant — where §7c read them as "flat, a lead cannot touch
them" — to a **38.8 ms median with 20–32 % tracking gains** on PhysX. The
per-group reading inverts. Any intuition carried over from the analytic fit is
wrong here; use the file. `validate()` refuses `cmd_lead_s="fit"` in the Isaac
regime outright (`test_analytic_preset_is_refused_in_isaac`), and
`test_isaac_regime_loads_the_published_fit` asserts the loaded wrist entries are
non-zero, which is the fit's signature.

> **A retracted finding, recorded so it is not re-derived.** An earlier version
> of this README justified `lead OFF` in Isaac with "158/158 strikes crossed the
> head early, so the analytic lead over-compensates". **That was wrong.** The
> sibling traced it to a gravity-feedforward-OFF artifact — the arms were
> sagging, not arriving early — and it has been corrected. Gravity feedforward
> is ON in the standard config, and the lead is ON with the Isaac-fitted vector.

Trainable: LoRA rank-8 deltas on the decoder's attention out-projections and FFN
matrices, the Gaussian head, the mean correction, and the critic. The 19.32 M
backbone is frozen and the freeze is asserted after the first backward.

### Why not the full-trajectory action

The approved analysis (`~/.claude/plans/wait-what-would-be-tender-narwhal.md`)
rejects the ~850-dim formulation on five mechanism-specific grounds, and the
most decisive is arithmetic rather than aesthetic: **`desired_kl = 0.015` is a
summed-over-17-dims budget** (≈8.8e-4/dim, calibrated in `rsl_rl_ppo_cfg.py`
:280-317). At 850 dims the same number is 1.8e-5/dim, and the adaptive
controller rails the learning rate at rsl_rl's 1e-5 floor — the exact documented
failure that already forced `epochs 8→3` and `desired_kl 0.006→0.015` on the MLP
runs. The other four: ~96 % of such an action never touches the environment
(only waypoint 0 executes before replanning), so PPO would spend its ratio
budget re-noising waypoints whose only effect is through the next tick's warm
start — a channel GAE cannot see; the refinement operator contracts ~7×/iteration
and **denoises the tail before it ever executes**; the AR(1) machinery is
per-control-tick and becomes meaningless across a chunk spanning several beats;
and the rollout buffer plus Adam state blow up. Executed-prefix keeps the env's
17-dim command interface, the AR(1) sampler, the KL budget and the whole reward
stack unchanged.

---

## Route taken for the loop

**A compact self-contained PPO**, whose update mirrors rsl_rl 3.1.2's
`PPO.update` term for term — same clip, the same adaptive-KL controller with the
same thresholds and 1.5× step, `surrogate + c_v·value − c_e·entropy`, clipped
value loss, global grad-norm clipping, advantage normalisation — rather than a
subclass of `rsl_rl.algorithms.PPO`. Three blockers, in order of weight:

1. **The auxiliary analytic cost has to share the forward pass.** rsl_rl's
   `update` is a monolithic loop with no per-loss hook — which is exactly why
   `bc_aux.DrumBcAuxPPO` does not fuse its BC term at all but takes *separate*
   gradient steps after `super().update()` returns, sharing only the Adam
   optimizer. That is right for BC (an MLP forward is free) and wrong here: the
   aux cost is evaluated on the same emitted trajectory the policy loss already
   produced, so fusing it is free, whereas the bc_aux pattern would mean a
   second 19 M-parameter refinement pass per minibatch — doubling the single
   most expensive part of the update to avoid writing ~150 lines.
2. **The policy's input is not an observation vector.** It is a stateful warm
   start plus a structured `ContextInputs`. rsl_rl's `RolloutStorage` stores a
   flat `TensorDict` per transition and reshuffles it; reconstructing a warm
   start from that is either impossible or wrong in a way nothing would catch.
3. **The sibling env is deliberately not a gym surface** — `isaac_rl.env` says
   so outright, and `INTERFACES.md` adds: *"A PPO variant that wants a per-step
   gym surface must wrap `plant`, not override it."* That is what
   `RolloutDriver` does.

A secondary hazard settles it: rsl_rl calls `policy.act` *inside* the minibatch
loop, and the AR(1) sampler mutates per-env noise state on every `act` — under
rsl_rl that state would be advanced 12 extra times per iteration, at minibatch
rather than env shape.

`AmortizedPpoActor` still implements the full rsl_rl `ActorCritic` surface
(`act` / `act_inference` / `get_actions_log_prob` / `evaluate` / `entropy` /
`action_mean` / `action_std` / `distribution` / `reset`) and ships
`register_with_rsl_rl()`, so it is a drop-in the day someone builds that gym
wrapper.

---

## Interfaces consumed from `isaac_rl` (the compatibility contract)

Everything below is resolved in `bridge.py` and nowhere else; each name was
checked against `amortized/isaac_rl/INTERFACES.md`. A break surfaces as an
`ImportError` naming the symbol, not as a wrong number hours into a run.

| module | names consumed | their stability mark |
| --- | --- | --- |
| `env_cfg` | `SIM_DT`, `DECIMATION`, `CONTROL_DT`, `DrumRLEnvCfg`, `assert_rate_consistency`, `build_env_cfg` | STABLE |
| `schedule` | `BatchedSchedule`, `ScheduleConfig` — `.reset/.context/.hit_index/.strike_context/.advance/.done/.goal`, attrs `geom`, `drum_radius`, `drum_valid`, `kit_names` | STABLE |
| `mpc` | `IsaacMPC`, `MPCConfig`, `TickResult` — `.reset/.masks/.plan/.commit`, attrs `last_ref0`, `prev_cmd`, `init`, `step_max`, `q_lo`, `q_hi`, `arm_cols`, `waist_cols` | STABLE (`TickResult.aux` PROVISIONAL — unused here) |
| `reward` | `ExecutedReward`, `ExecutedStrikeScorer`, `RewardConfig`, `RewardTerms`, `ContactScore` | key sets STABLE, weights PROVISIONAL |
| `measure` | `EpisodeView`, `latency_report`, `per_env_latency` | — |
| `env` (optional) | `make_env`, `DrumRLEnvCfg` — `.read_state/.plant/.reset_to/.step_dt` | PROVISIONAL |
| `safety` (optional) | `BatchedSafetyFilter`, `SafetyConfig` | PROVISIONAL, OFF by default |

**Two sources.** `AMORTIZED_PPO_ISAAC_RL_SOURCE=sibling` (default) uses the live
package; `=pinned` uses the snapshot in this directory. The default is
`sibling`, and the reason is concrete: the snapshot's `schedule.py` predates the
sibling's **float32 clock-drift fix** (`t_now += 0.02` accumulated in float32
flipped a hit-boundary comparison at t = 1.70 s and left the cursor one hit
behind for the rest of the episode). Since every clip time is quantized to the
control grid, *every* hit sits on that knife edge, so the pin would freeze a
known bug in exactly the quantity this project exists to improve. The snapshot
is kept on disk for diffing and as a fallback; falling back warns loudly.

Both configurations are tested: **50/50 pass against the live sibling and 50/50
against the pin.**

---

## How to run

```bash
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate /data2/eric/conda_envs/g1drum
source /data2/tiger/beyond_mimic_workspace/software/setup_conda_env.sh
export UNITREE_MODEL_DIR=/data2/eric/workspace/unitree_model TMPDIR=/data2/eric/tmp
cd amortized

python -m pytest isaac_rl_ppo/tests -q                 # 50 tests, no Isaac needed
AMORTIZED_PPO_ISAAC_RL_SOURCE=pinned python -m pytest isaac_rl_ppo/tests -q

python isaac_rl_ppo/train_ppo.py --dry-run             # resolve cfg + ckpt, no GPU
python isaac_rl_ppo/train_ppo.py --device cuda:3 --envs 64 --iterations 20 \
    --steps-per-env 24 --name smoke --headless
python isaac_rl_ppo/train_ppo.py --device cuda:3 --resume runs/isaac_rl_ppo_smoke/last.pt
```

**GPU:** `cuda:1` is reserved by the sibling session per their `INTERFACES.md`;
`cuda:0` / `cuda:2` are held by others. Use **`cuda:3`**. Every entry point calls
`torch.cuda.set_device` and `torch.set_float32_matmul_precision("high")`.

Artifacts land in `amortized/runs/isaac_rl_ppo_<name>/`: `config.json`,
`metrics.jsonl` (one object per iteration), `best.pt`, `last.pt`.

### Checkpoint resolution

`phase2d_waist` **only if its own report chapter or `summary*.json` explicitly
marks it ACCEPTED** → else `phase2b_main` (the standing champion). `phase2c_main`
is **blocklisted** and cannot be selected even explicitly without an error: it is
known-regressed (README 7c — overfit the seed-0 validation grid, +7 seed-0 / −8
seed-7, closed loop 25/55 → 8/55). File existence alone never selects
`phase2d_waist`, because two legs write into that directory.

---

## Two environment traps this package works around

Both cost a full Isaac startup to rediscover, so they are encoded in code with
comments rather than left as folklore.

**1. Pinocchio vs Isaac's Boost.Python registry** — *now solved upstream; the
workaround here is kept as a fallback.* RL-1b landed `isaac_rl.rnea_cache`
(committed, bit-exact), so `BatchedRNEA`, the gravity/Coriolis feedforward,
`FusedCost` and the in-process strike scorer all work inside Isaac. The
description below is why the cache exists and why this package builds the RNEA
pre-launch anyway (it costs nothing and covers the case where the cache is
absent or stale).

`BatchedRNEA.from_urdf` reads
`model.parents` (a `std::vector<Index>`) through Boost.Python. Inside a running
Isaac process that conversion raises

```
TypeError: No Python class registered for C++ class
           std::vector<unsigned long, std::allocator<unsigned long> >
```

because Isaac's runtime brings its own Boost.Python/eigenpy and shadows the
converter registration. It is **not** a broken Pinocchio build — the identical
call succeeds in the same conda env in a process that has not launched Isaac
(verified both ways). `train_ppo.py` therefore builds the RNEA **before**
`AppLauncher` and moves it to the GPU afterwards; the constructed object is
plain torch tensors plus Python ints, so nothing it does later touches Boost.
Measured working in-Isaac: an earlier smoke ran 18 iterations with the analytic
aux cost ON, its value falling 1541 → 597 as the LoRA trained.

This once blocked `isaac_rl/env.py`, which builds its RNEA for the gravity
feedforward *inside* `__init__` (after the app is up), making `make_env` raise
with `gravity_feedforward=True`. **Resolved by `rnea_cache`.** Gravity
feedforward ON is now the standard config and the default here; `--no-gravity-ff`
remains as a diagnostic and is explicitly *not* how runs should be configured —
ff OFF makes the arms sag, which is what produced the retracted "158/158 early"
reading above.

**2. The PhysX device trap.** `SimulationCfg.device` defaults to `cuda:0`. If
the app runs on a different GPU, PhysX blocks *forever* inside
`get_dof_velocities` — no exception, no timeout, no log line. On this box
`cuda:0` is held by another session, so the default is exactly the trap.
`isaac_rl.env.make_env` composes build-and-instantiate in one call and does not
expose the sim cfg in between, so `train_ppo.py` calls the two halves directly
(`build_isaac_cfg` → pin `sim.device` → instantiate) and prints the pinned
device. A public "build, override device, instantiate" entry point in
`isaac_rl.env` would remove the need to reach for a private helper.

**Pinning `sim.device` is necessary but NOT sufficient.** Isaac's Warp layer and
several `isaaclab_*` extensions still initialise on `cuda:0` during app startup,
whatever `--device` says. When `cuda:0` filled to 31.8/32.6 GB the same failure
came back as

```
Warp CUDA error 2: out of memory (in function cuda_context_is_primary)
Exception: Failed to get DOF velocities from backend
```

i.e. the trap can surface as a *raise* rather than a hang, from a GPU you never
asked for. The robust fix is to hide the other GPUs from the process entirely:

```bash
CUDA_VISIBLE_DEVICES=3 python isaac_rl_ppo/train_ppo.py --device cuda:0   # cuda:0 == physical 3
```

Note the renumbering — under `CUDA_VISIBLE_DEVICES=3` the only visible device is
`cuda:0`. This is the recommended invocation on this box. **Check the
`INTERFACES.md` changelog for GPU occupancy before launching**: the sibling's
RL-2 training claims `cuda:1`, and `cuda:0` / `cuda:2` are held by others, so
this package stays on physical GPU 3.

**3. `app.close()` hangs.** Isaac's shutdown blocks in the carb tasking mutex —
it serialised the sibling's queued runs behind a 28-minute timeout.
`train_ppo.py` flushes its artifacts and then calls `os._exit(0)`, which is what
their scripts do. Everything worth keeping is on disk before that point.

---

## The safety filter is a CONTROL, and it defaults OFF

`--safety off` (the default) is a deliberate experimental control, not an
oversight. Two independent reasons:

1. **It breaks the PPO contract.** The filter rewrites the executed command, so
   the stored log-prob no longer describes the action that actually ran. Handling
   that correctly (store the projected action, or treat projection as env
   dynamics) is an open design question — DECISION 7.
2. **It confounds the return.** In the sibling's ff-ON baseline the filter
   engages on **26–34 % of ticks**. Any reading of a PPO return from a filtered
   run is confounded by that intervention rate, so the off-control has to exist
   first. Their RL-2 is isolating the same variable.

**The honest Isaac baseline is an 87 % miss rate** (ff ON, Isaac lead) against
63 % in the analytic sim — with the filter-engagement confound above still
unresolved. That 87 % is the number this fine-tune has to move, and it is the
number to compare against; the analytic 63 % is a different plant.

---

## DECISIONS NEEDING REVIEW

1. **Noise is added after the lead, not to `traj[:, 0]`.** The lead interpolates
   waypoints 0 and 1 with a *per-joint* fraction, so perturbing waypoint 0 by
   `d` moves the command by `(1 − f_j)·d` — a different effective σ on every
   joint, changing whenever the fit is re-run. The log-prob would still be exact,
   but the σ the KL controller reasons about would not be the σ the plant sees.
   Adding the offset to the finished command keeps "one unit of action" = "one
   radian at the joint" for all 17 dims, which is what the shared 17-dim KL
   budget assumes. Re-clamping after the offset means the executed command can
   differ from the sampled action near a limit; that is ordinary action clipping
   and PPO's ratio is unaffected. `test_lead_is_not_applied_twice` pins it.
2. **The tracking reference is `"sampled"` — RULED, 2026-08-12.** The tracking
   term scores the plant against what was **actually commanded** (the sampled
   action, with whatever lead is active), not against the un-led mean plan.
   Rationale: scoring against the mean charges the policy for its own
   exploration noise, so the reward is not action-consistent and the PPO
   gradient is corrupted — the reward must be a function of the action that was
   taken.

   This **deliberately diverges from the sibling**, whose `last_ref0` is the
   un-led mean first waypoint (`mpc.py`, `INTERFACES.md`). The divergence is
   small in their setting and large in ours: their PSN policy has no per-tick
   sampling noise, so "the mean" and "what was commanded" nearly coincide, while
   here they differ by exactly the exploration offset. It is also **moot in
   Isaac today**, since the lead is OFF — led and un-led references are the same
   vector until RL-1b lands. `track_reference="mean"` reproduces the sibling
   exactly. Their session has been informed.

   Note this does *not* reintroduce the failure their comment warns about
   ("pricing the lead itself as error"): the reference is the sampled action in
   joint space, not the led command, so an active lead is still not charged as
   tracking error.
3. **`reward_plan_cost_weight = 0.0`.** The sibling prices `FusedCost` as a
   non-differentiable *reward* term (`w_plan_cost = 0.05`). This package prices
   the same quantity as a differentiable *auxiliary loss* (weight 0.1), which is
   strictly better signal for the same thing; leaving both on double-counts it.
   Set >0 to restore their behaviour.
4. **The aux cost drops the terminal groups by default**
   (`collision, torque, velocity, joint_box, smooth, posture`). In Isaac the
   executed contact *is* the terminal objective, and keeping the kinematic
   terminal too re-creates the conflict README 7c identified (a plan that
   *executes* on the head is kinematically ~140 mm short of it). `None` uses
   `CostTerms.total`.
5. **Aux weight 0.1 is a starting guess, not a measurement.** On an untrained
   tiny model the aux gradient reached ~2.6e4 before clipping; `max_grad_norm`
   absorbs it, but the weight should be re-tuned against the real backbone once
   the first curves exist.
6. **fp32, not bf16.** The sibling's `MPCConfig.amp_dtype` defaults to bfloat16;
   this package forces fp32, because PPO re-evaluates the mean at update time and
   the ratio is only 1.0 at the first epoch if collection and update agree
   numerically. bf16 carries ~3 decimal digits and would bias log(ratio) before a
   single gradient step. This costs throughput — the main lever if collection is
   too slow. `test_ratio_is_one_before_any_gradient_step` measures it (currently
   `|ratio − 1| < 1e-4`).
7. **The QP safety filter is OFF.** A filter that rewrites the executed command
   changes the action the stored log-prob was computed for. Correct handling
   (store the *projected* action, or treat projection as env dynamics) is a real
   design question, not a flag flip.
8. **Episode start matches `play_isaac`; clip-end keeps leftover `q` with a
   1 s gap.** After Isaac `env.reset`, the USD `init_state` is overwritten with
   `amortized_opt.ready_joint_pos`. Clips are padded so the first hand hit is at
   `--first-hit-s` (default 1.0 s). `BatchedSchedule.done` is patched to wait
   until `t_now >= duration` so the RCC `post_buffer` (last hit + `--clip-gap-s`,
   default 1.0 s) actually plays — leftover pose, no next clip yet. Then
   `on_done` teleports to that leftover `q` at rest and redraws the clip.
   `--isaac-ready` keeps Unitree standing; `--first-hit-s 0` / `--clip-gap-s 0`
   restore native first-hit times and chain-on-last-hit.
9. **Time-limit bootstrapping.** Episodes end on clip exhaustion — a time limit,
   not a failure — but GAE cuts the bootstrap at `done`. The standard fix is to
   bootstrap through time-limit terminations. Left as-is to match the sibling's
   episode semantics; worth revisiting.
10. **Beat-scheduled ρ is not ported.** The MLP version reads an upcoming beat
    time from fixed MLP observation indices that do not exist here. τ is
    carried across correctly (ρ is recomputed at 50 Hz, *not* copied from the
    60 Hz stack — `test_ar1_rho_from_tau_at_50hz` pins this); the beat schedule
    is a future knob.
11. **`amortized_opt.FusedCost` takes no `sdf=` kwarg**, but the sibling's
    `isaac_rl/scripts/smoke_rl.py` passes one — that call would raise. Reported,
    not fixed: their tree is read-only to this package.
12. **`BatchedPDSim.step` now returns three values**, not two. Another session
    added a PID integral term, so the analytic plant returns
    `(q, qd, integral)` while Isaac's `env.plant` still returns `(q, qd)`.
    `RolloutDriver._step_plant` tolerates both and threads the integral back
    only when the plant accepts it (otherwise a `ki > 0` run would silently
    reset its accumulator every tick, making the I term a no-op). **The
    sibling's `DrumMPCLoop` still unpacks two and therefore no longer runs
    against `pd_sim`** — reported, not patched.
13. **`ModelConfig.dec_layers` default moved 6 → 3** mid-build. The run is
    immune (the model is built from `ck["model_cfg"]`, so the champion's 6
    layers / 19.32 M are used), but anything reading library defaults now gets a
    10.8 M model. `tests/test_lora_and_determinism.py::CHAMPION_ARCH` pins the
    shipped architecture explicitly for this reason.
14. **The aux gate default is `"auto"`, resolving ON — RULED, 2026-08-12.**
    With the aux cost off, nothing prices waypoints 1..H-1, which are the next
    tick's warm start; that defeats the design. `"auto"` resolves ON because the
    pre-launch RNEA build succeeds (evidence: an 18-iteration in-Isaac run with
    it ON, aux falling 1541 → 597 as the LoRA trained). The manual off switch
    (`--aux-gate off`) is kept, and is what the reported smoke used.

    Caveat carried from decision 5: at weight 0.1 the aux term *dominates* the
    loss (aux ~1500 against surrogate ~0.1), so the weight — not the gate — is
    the thing to tune first on the real backbone.
15. **Always-orth combine at waypoint 0, ON by default.** Scalar
    `surrogate + aux.weight * cost` lets aux's magnitude swallow PPO. The
    update always-projects the waypoint-0 PPO gradient onto the analytic
    cost's nullspace. **Cost is then rescaled per sample** so
    `||g_cost||_F = cost_to_rl_norm · ||g_rl0||` (default 1) and the leftover
    RL is added at PPO's native scale. The first Isaac run skipped that match
    (`g_cost` at aux ~4000 vs surrogate ~0.08): KL sat at 0.07 (target 0.015),
    the LR railed, and the plan NaNed on iter 5. `--cost-to-rl` / `--orth-scale`
    / `--orth-leak` are the knobs (`--orth-leak` is the parallel-component
    leak at waypoint 0; default 0). `kl_early_stop` (ON) aborts the rest of
    an iteration once a minibatch exceeds `2 · desired_kl`. Nonfinite
    advantages skip the update.
    Log `train/orth_cos` and `train/orth_cost_scale`.

---

## Throughput note

Collection dominates, as it does in the MLP stack (98.6 % of wall clock at 0.9 M
params). A 19 M actor with a decoder pass per env-tick multiplies rollout
inference substantially; the levers, in order, are `num_envs`,
`num_steps_per_env`, re-enabling bf16 for collection only (at the cost of
decision 6), and `micro_batch_size` for update memory. `micro_batch_size` is a
pure memory knob — chunks are reweighted exactly by size, and
`test_micro_batching_matches_a_single_pass` asserts the accumulated gradient is
identical to a single pass to within 1e-5 relative.
