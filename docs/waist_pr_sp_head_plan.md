# Learned waist pitch/roll set-point (encoder head) — plan

> **Status: implementing.**  Sibling of `docs/waist_sp_head_plan.md` (yaw).
> Tests in `tests/test_waist_pr_sp_head.py`.  Does **not** reshape the yaw
> head: a `waist_sp_head` checkpoint loads strictly; this flag is append-only.

## Goal

Lean is a **posture**, not momentum (`CostConfig.w_waist_pr_vel`, the 2026-08-13
reach measurement, the seed's hold/`waist_upright_ease_wp`).  The decoder
currently treats it as a per-tick map from measured `q0`: closed-loop probe
on `rhythm_bellman_nf1_rope{,_plant}` showed plant |pitch|/|roll| p90 on the
URDF rail (±0.52), `∂wp1/∂q0` p90 > 1, and 89–92 % of ticks asking for *more*
lean than measured.  The seed is ignored (`∂wp1/∂init ≈ 0`).

Give the model one learned, per-goal-segment pitch/roll set-point — the same
job the yaw head already does — so the torso (a) can lean when reach pays for
it, (b) holds that lean through the segment instead of hunting at 50 Hz, and
(c) stays near upright unless the cost actually wants otherwise.

## Design

1. **Head = residual on 0, zero-init.**  There is no analytic lean law
   (`waist_aim.py` is yaw-only on purpose).  `sp_pr = clamp(MLP(pooled tokens),
   ±URDF)`.  Untrained = upright.  **Pooled tokens only** — no measured
   pitch/roll, no `yaw_now`.  Putting plant lean in the encoder is the
   2026-08-14 `yaw_now` runaway.

2. **Own conditioning token, appended BEFORE the yaw token** so
   `substitute_waist_sp`'s `memory[:, :-1]` (yaw is last) stays valid.
   Payload is `[sp_pitch, sp_roll]` only, not `[sp, q_pr, sp−q_pr]`.  The
   decoder already sees measured lean on the state token; the encoder token
   must not tick, or the Isaac cache is a lie.

3. **Phase-2 anchor reuses `w_waist_upright`.**  When
   `StrikeContext.waist_pr_ref` is set, the upright well prices
   `‖q_pr − ref‖²` instead of `‖q_pr‖²`.  Ref `0` is bit-identical to today.
   Sample 0 (measured `q0`) is masked out of the reduction under a ref, same
   as `waist_aim`, so a trajectory sitting on the set-point reads zero.
   Do **not** also price P/R through `waist_aim` (two anchors fighting).
   Head runs keep `waist_aim_yaw_only=True`.

4. **Tether = `w_waist_pr_sp_tether · mean(sp_pr²)`.**  Residual *is* `sp`
   (base is 0).  This is “help reach, don’t sit on the rail.”  Unramped.
   Default **5.0** when the train flag is on (yaw left this at 0 and
   co-drifted ~1 rad).  Log `waist_pr_sp_abs`.

4b. **Hold encode-time pose = `w_waist_pr_sp_hold · mean((sp_pr − q0_pr)²)`.**
    The announcement stays near the waist at encode so a new drum is an arm
    problem, not a torso yank.  `q0` is **stopgrad** and is **not** an encoder
    input (that is the `yaw_now` runaway).  Default **5.0** when the train
    flag is on.  Log `waist_pr_sp_hold` (|sp − q0|).  Yaw has a separate
    `w_waist_sp_hold` (default 0 — yaw is supposed to walk toward `aim_yaw`).

5. **No P/R search distillation in v1.**  Yaw search stays yaw-only.  The
   tether is the bound; a 2-D grid is a later add if `sp_abs` saturates
   against it at the rail.

6. **Seed eases to `sp_pr`, like yaw to `aim_yaw`.**  After every
   `encode_context` that precedes a decode, `seed_waist_pr_to_sp` rewrites the
   pitch/roll columns of `init` to a velocity-matched quintic from `q0` to
   the announced set-point (detached).  Untrained (`sp == 0`) is the existing
   ease-to-upright.  The numpy pool still eases to 0 -- there is no network at
   pool-build time; the rewrite is the analog of `aim_yaw` being available
   without the head.  Roller regoal encodes the switched rows so a new hit
   entering the 1 s window ramps lean toward the new SP in the same rebuild
   that ramps yaw toward the new aim.
   Quintic terminals then get a 1-step full-arm DLS (`adjust_arm_terminal`)
   after that actual strike-knot waist is known -- kit jitter, yaw-bin, and
   lean vs the upright library all fold into one FK error.  Apex teacher
   stays at 3 steps.

## What this is not

- Not `frozen_joints` (that is tether → ∞, no lean for far cymbals).
- Not putting `q_pitch`/`q_roll` on `ContextInputs`.
- Not widening `waist_sp_out` 1→3 (would break every yaw-head checkpoint).

## Changes by file

### `amortized_opt/model.py`
- `ModelConfig.waist_pr_sp_head: bool = False`.  `ARCH_FROM_WEIGHTS` proof
  `encoder.waist_pr_sp_out.weight`.  `n_context_tokens += 1`.
- `ContextEncoder`: `waist_pr_sp_hidden` (`d → d/2`), `waist_pr_sp_out`
  (`d/2 → 2`, zero-init), `waist_pr_sp_proj` (`2 → d`), `waist_pr_sp_emb`.
  Append P/R token before the yaw token.  Clamp to URDF pitch/roll.
- `ContextEncoding` += `waist_pr_sp (B, 2)`, `waist_pr_res (B, 2)`; `to` /
  `expand`.

### `amortized_opt/cost.py`
- `StrikeContext.waist_pr_ref: (B, H, 2) | None` — knot convention identical
  to `waist_yaw_ref` (row `k` is knot `k+1`), columns `[pitch, roll]`.
- Upright well: if ref set, `‖q_pr − ref‖²` and drop sample 0.

### `amortized_opt/train_cost.py`
- `Phase2Config.waist_pr_sp_from_head`, `w_waist_pr_sp_tether`.
- Main loss: `waist_pr_ref = sp[:, None, :].expand(-1, H, -1)`; tether on
  `mean(res²)`; hold on `mean((sp − q0_pr)²)`.  After encode,
  `seed_waist_pr_to_sp` on `batch["init"]`.
  Refuse `from_head` without the model flag.
- Roller tick + `_rebuild_waist_from_aim` ease P/R to the encoder SP.
- `substitute_waist_sp` forwards the P/R fields; yaw token stays last.

### `amortized_opt/warmstart.py`
- `seed_waist_pr_to_sp`: velocity-matched quintic on pitch/roll to `sp_pr`.

### `amortized_opt/planner.py`, `probe.py`, `isaac_rl*/mpc.py`
- After encode, rewrite init P/R to `enc.waist_pr_sp` (skip `init="hold"`).
- Quintic inits then get a 1-step full-arm DLS after that waist is known.

### `scripts/train_phase2.py`, `scripts/train_phase1_apex.py`
- `--waist-pr-sp-head`, `--w-waist-pr-sp-tether`, `--w-waist-pr-sp-hold`.
  Phase 2 sets `from_head`, defaults tether and hold 5.0, warns if
  `--w-waist-upright` is 0.

### `tests/test_waist_pr_sp_head.py`
OFF bit-identical; untrained `sp == 0`; invariant to `yaw_now`; gradient
through the upright term and the tether; ref-hold pays 0; checkpoint
reconciliation; both heads keep yaw-search substitution valid.
