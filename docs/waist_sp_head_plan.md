# Learned waist-yaw set-point (encoder head) — plan

> **Status: implemented** (2026-08-13).  Tests in
> ``tests/test_waist_sp_head.py``.  Two consistency fixes landed beyond the
> plan: the MPC roller invalidates a slot's stored ``aim_yaw`` on every
> tensorized regoal and on a post-hit carry respawn (the aim was solved for
> goals/state the slot no longer has), resetting ``yaw_now`` to the measured
> yaw so the head's fallback -- "hold where you are" -- matches what the
> roller's own waist re-seed does with the trajectory.
>
> **Addendum (same day): pitch/roll ease-to-upright.**  The old seeds held
> pitch/roll at the MEASURED start pose for the whole horizon; with starts
> sampled in a posture box, every phase-1 target therefore taught "keep your
> lean" -- the prior the phase-2 upright well fights.
> ``warmstart_trajectory(waist_upright_ease_wp=...)`` (None = old behaviour
> bit for bit) replaces the hold with a quintic smoothstep to upright (or to
> ``waist_setpoint`` when given, e.g. the MPC loop's planned command),
> velocity-stretched at extreme leans.  Threaded through both scenario
> configs, the apex builder, ``MPCGenConfig`` / ``quintic_planner``, and
> exposed as ``--waist-upright-ease-wp`` on both training scripts (try 5).
> Tests: ``tests/test_waist_upright_ease.py``.

## Goal

Give the model one learned, per-goal-segment waist-yaw set-point and price the
trajectory against it, so the torso (a) stops drifting under the rate-only
pricing of section 9b, (b) holds still through contact, and (c) can learn a
*better* aiming law than `waist_aim`'s hand-set constants where the cost pays
for it — without a precomputed library and without two anchors fighting.

## Design (the short version)

1. **Head = analytic law + zero-init residual.**  The context encoder gains a
   small readout: `sp = clamp(base + MLP([pooled_tokens, yaw_now, base]), ±1.745)`
   where `base = aim_yaw(targets, yaw_current=yaw_now)` computed at data time,
   and the MLP's final linear is **zero-initialized** (the same trick
   `output_mode="residual"` uses).  The untrained head therefore *is* the
   analytic law exactly — no distillation loss, no cold start, no phase-1
   changes to the loss at all.
2. **The set-point conditions the decoder** as one extra cross-attention
   memory token (`proj([sp, yaw_now, sp−yaw_now]) + own bias param`) so
   "follow your own announced plan" is easy credit assignment.  No `type_emb`
   widening; the token gets its own embedding parameter like the decoder's
   `state_emb`.
3. **Phase 2 anchor reuses the existing `w_waist_aim` term.**  The cost already
   has a yaw-position IL term pricing `(q_yaw − waist_yaw_ref)²` with its own
   ramp (`weights["waist_aim"]`) and metric slot (`c_waist_aim`) — it just has
   nothing learned to point at.  When the head is on, the trainer fills
   `StrikeContext.waist_yaw_ref` with `sp` broadcast over the horizon instead
   of the analytic ramp; gradient flows through the term into the head
   (encode runs outside `torch.compile` but inside autograd).  Two inert-by-
   default `CostConfig` knobs make the term fit the job: a **deadband** on the
   yaw error (free within ±d, quadratic outside) and a **yaw-only** switch
   (pitch/roll are already priced by the upright well).
4. **Tether = residual penalty.**  `w_tether · mean(res²)` at trainer level —
   the residual *is* `sp − aim_yaw` by construction, so the tether needs no
   extra target plumbing.  Unramped (it is exactly zero at init).  Mean |res|
   is the "how far has the learned law moved from the analytic one" diagnostic.

Both-ways gradient is deliberate: the anchor is a within-segment variance
term + amortized consistency, not a hard pin; the pair (trajectory, head)
walks together to whatever yaw the cost actually prefers, bounded by the
tether and the init basin.

## Changes by file

### `amortized_opt/tokenizer.py`
- `ContextInputs` += `yaw_now (B,)`, `aim_yaw (B,)`, `aim_valid (B,) bool`,
  defaulted in `__post_init__` (zeros / False) like the idle block, so every
  pre-existing call site and every cached pool keeps working.
  `CONTEXT_FIELDS` is derived from the dataclass fields, so the data pipeline
  records/batches them automatically.

### `amortized_opt/rhythm.py`
- `build_context_inputs(..., yaw_now=None, waist_cfg=None)`: when `yaw_now`
  is given, compute `aim_yaw` per row from the snapshot's ACTIVE current
  goals in the pelvis frame (`g.position − PELVIS_IN_ENV`), with
  `yaw_current = yaw_now[b]` — same function + same default `WaistAimConfig`
  the warm-start ramp uses, so `base` equals the ramp's terminal by
  construction.

### `amortized_opt/model.py`
- `ModelConfig.waist_sp_head: bool = False` — OFF is bit-identical (strict
  checkpoint load).  `ARCH_FROM_WEIGHTS` entry so old checkpoints reconcile.
- `ContextEncoder`: masked mean-pool of the final tokens → head MLP
  (zero-init final linear) → residual → `sp`; append the conditioning token
  to `memory` (mask True).
- `ContextEncoding` += optional `waist_yaw_sp`, `waist_yaw_res`; update
  `to()` / `expand()` and the positional construction at the TC slice in
  `train_cost.py`.

### call sites of `build_context_inputs` (pass the state's waist yaw)
`mpc_gen.py`, `warmstart_data.py`, `warmstart_apex.py`, `phase2_data.py`,
`probe.py`, `planner.py`, `scripts/mpc_smoke_apex.py`.

The two batched schedules construct `ContextInputs` directly rather than via
`build_context_inputs`, so they fill the aim fields themselves:
`isaac_rl_ppo/schedule.py` and `isaac_rl/schedule.py`
(`BatchedSchedule.context(yaw_now=...)` computes `aim_yaw` host-side from the
active goals in the pelvis frame), with `isaac_rl_ppo/mpc.py` /
`isaac_rl/mpc.py` (`IsaacMPC._encoding`) threading the measured waist yaw in.
Callers that omit `yaw_now` (the large-B RL training loops) keep the zero
fallback, i.e. the head conditions on "hold 0 rad".

### `amortized_opt/cost.py`
- `CostConfig.waist_aim_deadband: float = 0.0`,
  `CostConfig.waist_aim_yaw_only: bool = False` — both inert at defaults;
  applied inside the existing `waist_aim` raw-term block.

### `amortized_opt/train_cost.py`
- `Phase2Config` += `waist_sp_from_head: bool = False`,
  `w_waist_sp_tether: float = 0.0`.
- Loop: when the head drives the anchor,
  `strike = replace(strike, waist_yaw_ref=enc.waist_yaw_sp[:, None].expand(-1, bucket))`
  and `total += w_tether * mean(res²)`; report mean |res| with the eval line.

### `scripts/train_phase1_apex.py`, `scripts/train_phase2.py`
- `--waist-sp-head` (both), `--waist-aim-deadband`, `--w-waist-sp-tether`
  (phase 2; `--w-waist-aim` becomes the anchor weight).  Phase 2 accepts a
  warm start without the head via the idle-token partial-load precedent.

### `tests/test_waist_sp_head.py`
- head OFF: state-dict keys and encoder output unchanged;
- head ON, fresh init: `sp == aim_yaw` exactly (zero residual);
- gradient reaches head params through the anchor term;
- `build_context_inputs` aim fields agree with a direct `aim_yaw()` call;
- old-context path (fields absent) still batches and encodes.

## Known caveats

- **Enabling the head over a pre-head warm start is NOT bit-identical for the
  trajectory.**  The conditioning token is built from the *unnormalized*
  `[sp, yaw_now, sp - yaw_now]` triple and joins the cross-attention memory,
  so it perturbs the decoder's output from refinement step 0 even with a
  zero-initialized residual.  Only the SET-POINT is exactly the analytic law
  at init (`sp == aim_yaw`, and `waist_yaw_res == 0` wherever the aim sits
  inside the ±1.745 rad clamp); the decoded trajectory shifts slightly the
  moment the head is switched on.

## Notes / verified

- The CLAUDE.md `simplified_cost`/`validate()` weight-zeroing gotcha lives in
  the repo trajopt's `TrajOptConfig`, not this package's `CostConfig`;
  `w_waist_aim` is a real, wired weight with a `TERM_KEYS` slot.
- `aim_yaw` handles randomized drum positions (raw target inputs) at
  ~95 µs/call — negligible at data-generation time; never called in the
  training step.
- Phase 1 needs **no loss change**: the imitation targets' yaw ramp and the
  head's init are the same law, and the head trains through the conditioning
  token.
