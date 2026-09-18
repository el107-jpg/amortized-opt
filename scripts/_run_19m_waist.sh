#!/usr/bin/env bash
# Full 19 M run on the merged tree (collision_peak + waist-ffwd + the phase-1
# idle arm): phase-1 warm start -> phase-2 cost descent with
#
#   * the torque PEAK term      (w_torque_peak = 2.0, the shipped default)
#   * the collision PEAK term   (w_collision_peak = 500, README's suggested
#                                start; this is its FIRST run in a loss)
#   * section 9b's "direct" waist stabilisation: pitch/roll priced by an
#     upright well and boxed, yaw RATE priced.  Yaw POSITION stays free --
#     the torso has to turn to reach the outer drums.
#
# Phase 1 is re-run rather than reusing runs/phase1_main/best.pt (2026-08-10):
# every stored 19 M warm start predates the idle-arm work, so its IDLE context
# token is 7 424 random parameters and its targets teach the idle arm to freeze
# where it is -- the opposite of what phase 2 prices.  That matters more than
# usual here: ok_collision scores the STRIKING hand only, but 159 of 243 cases
# have a hand inside a drum once the idle arm is counted, so a large share of
# the collision gradient is spent on violations acceptance never reads.
set -euo pipefail

cd "$(dirname "$0")/.."
DEV="${DEV:-cuda:1}"
# New tag: the head + ease + velocity-matched seed generation of 2026-08-13.
# w_waist_aim 500 is the shipped weight (scripts/ablation_waist.py: the 9b
# arms ran 5x/10x of it) -- NOTE the 9b aim arms carried the waist_yaw_ref
# off-by-one fixed 2026-08-13, so their poor showing does not indict the term.
# The deadband (0.15 rad) and tether (50) follow the flag help / plan doc;
# the tether has no principled default -- watch ``waist_sp_res`` in the train
# log and treat saturation as the tether binding.
TAG="${TAG:-19m_waist_sp}"
P1="runs/phase1_${TAG}"
# Suffix the PHASE-2 dir only, so a phase-2 re-tune keeps the phase-1 warm
# start AND the pool cache (both keyed on TAG) instead of paying 20 min of
# phase 1 + 31 min of scenario generation to change a cost weight.
P2="runs/phase2_${TAG}${P2_SUFFIX:-}"
# SKIP_P1=1 reuses "$P1/best.pt" as-is.  Do NOT do that across a change of
# AIM_TAU: phase 1 distils the waist set-point head onto the analytic aim, so a
# checkpoint trained against one aim law is a warm start for the other only in
# the sense that the residual has to unlearn it.
SKIP_P1="${SKIP_P1:-0}"
# A RUN-SPECIFIC pool cache: safe to reuse on a resume, impossible to collide
# with a stale one from another configuration (the npz carries no config key,
# which is how the section-8 runs silently trained on 30 841 scenarios).
#
# TILT_DEG and RADIUS_FRAC are BAKED into the pool, so they have to key the
# cache name too -- else turning one on and reusing "$POOL" trains on the old
# flat-normal / nominal-radius scenarios and reports a clean run.  Deriving the
# suffix here rather than asking the caller to remember is the whole point: the
# failure above was silent, and the schema marker cannot catch either knob (it
# checks CONTEXT_FIELDS *names*, and both knobs only change those fields'
# VALUES).
TILT_DEG="${TILT_DEG:-0}"
RADIUS_FRAC="${RADIUS_FRAC:-0}"
# ...and so is AIM_TAU: the pool's ``aim_yaw`` column and its seeds' yaw ramps
# are both solved with WaistAimConfig.future_tau, and the schema marker is just
# as blind to it as to the two above (names, not values).  1.0 s is the shipped
# future-aware law; 0 is the pre-2026-08-14 current-goals-only one.
AIM_TAU="${AIM_TAU:-1.0}"
# ...and so are THE CLIPS: every scenario's goals come from a random instant of
# a random assets/rcc/train clip, and the 2026-08-14 expansion (10 -> 687
# clips; floor_tom / cymbal coverage went from ~zero to thousands of hits)
# changes the pool completely while the schema marker sees nothing.  Key the
# cache on the clip-set size so the old npz can never be silently reused.
RCC_N=$(ls assets/rcc/train/*.json 2>/dev/null | wc -l)
# Rhythm augmentation + rare-drum targeting (2026-08-14), all BAKED INTO THE
# POOL like the knobs above.  JIT_T: uniform +- seconds on each hit's time,
# shared across a unison group (0.03 s = 1.5 waypoints).  JIT_V: uniform +- on
# the 0-1 loudness before the 1-3 m/s speed map.  RARE_BOOST: fraction of
# plan-native draws targeted at an inverse-frequency-weighted drum -- the
# floor_tom/cymbal coverage lever; the corpus is 36 % snare / 4.3 % floor_tom,
# and uniform instants inherit that skew no matter how many clips there are.
JIT_T="${JIT_T:-0.03}"
JIT_V="${JIT_V:-0.2}"
RARE_BOOST="${RARE_BOOST:-0.25}"
# PAIR_FRAC: share of pool scenarios from the uniform two-handed PAIR family
# (both hands, drum combination uniform over the 26 legal ones) -- the real
# grooves still put most simultaneous work on hihat+snare.  BAKED INTO THE
# POOL, so it keys the cache below.  PLATEAU_EVALS: rolling-eval ticks without
# improvement before the run stops; every 19m_waist_sp predecessor died on
# this counter (5), never the step cap, so 10 is the "let it live" arm.
PAIR_FRAC="${PAIR_FRAC:-0}"
PLATEAU_EVALS="${PLATEAU_EVALS:-5}"
# TRACKING-TORQUE SYSTEM (2026-08-15).  W_TRACK_TORQUE / W_TRACK_TORQUE_PEAK
# price the DEMANDED PD torque (RNEA + kernel feedback estimate) against the
# velocity-derated rating; SAT_EFFORT=1 clamps the analytic plant's motors at
# the rating like Isaac does.  Enabling the weights also flips the eval's
# ok_torque to the same estimator (one meaning of accepted; NOT comparable
# with runs before 2026-08-15).  Validated: wrist peak ratio est/sim
# 1.04-1.18, lag 0; RNEA alone sees ZERO of the wrist saturation.
W_TRACK_TORQUE="${W_TRACK_TORQUE:-0}"
W_TRACK_TORQUE_PEAK="${W_TRACK_TORQUE_PEAK:-0}"
SAT_EFFORT="${SAT_EFFORT:-0}"
# POSTURE BOX (2026-08-16).  WIDE_POSTURE=1 empties G1_DRUM_POSTURE_LIMITS_RAD,
# so shoulder yaw and wrist roll get their full URDF range instead of ~30 % of
# it.  Motivation is TORQUE: wrist roll is rated 25 N*m and wrist pitch/yaw
# only 5 N*m, and the 5 N*m pair is what saturates -- the tied-down joint is
# the one with headroom.  BAKED INTO THE POOL (start sampler + seed clip), so
# it keys the cache.  Isaac never had this preference, so widening moves us
# TOWARD the sim, not past it.
WIDE_POSTURE="${WIDE_POSTURE:-0}"
if [[ "$WIDE_POSTURE" == "1" ]]; then export AMORTIZED_POSTURE_LIMITS=wide; fi
POOL_TAG="${TAG}_rcc${RCC_N}"
if [[ "$TILT_DEG" != "0" ]]; then POOL_TAG="${POOL_TAG}_tilt${TILT_DEG}"; fi
if [[ "$RADIUS_FRAC" != "0" ]]; then POOL_TAG="${POOL_TAG}_rad${RADIUS_FRAC}"; fi
if [[ "$JIT_T" != "0" || "$JIT_V" != "0" ]]; then POOL_TAG="${POOL_TAG}_jit${JIT_T}x${JIT_V}"; fi
if [[ "$RARE_BOOST" != "0" ]]; then POOL_TAG="${POOL_TAG}_rb${RARE_BOOST}"; fi
if [[ "$PAIR_FRAC" != "0" ]]; then POOL_TAG="${POOL_TAG}_pf${PAIR_FRAC}"; fi
if [[ "$WIDE_POSTURE" == "1" ]]; then POOL_TAG="${POOL_TAG}_wide"; fi
# UNCONDITIONAL, unlike the two above: the pools already sitting in $TMPDIR
# were baked before the aim law read the futures at all and carry no marker of
# it, so the new default has to claim a name none of them can already have.
POOL_TAG="${POOL_TAG}_tau${AIM_TAU}"
POOL="${TMPDIR:-/tmp}/p2pool_${POOL_TAG}.npz"

# Generate the phase-2 pool CONCURRENTLY with phase 1.  It is CPU-bound and
# independent of the phase-1 checkpoint (the on-policy half of phase-2's data
# is the ROLLER replay, built at train time), so its ~25 min hides entirely
# inside phase-1's GPU training.  32 workers, leaving phase-1's own generator
# its 48 of the 128 logical cores.  The flags MUST mirror the phase-2
# invocation's scenario-shaping set below: the cache's schema marker checks
# CONTEXT_FIELDS names only, so a config drift here would produce a silently
# different pool, not an error.
POOL_GEN_PID=""
if [[ ! -f "$POOL" ]]; then
  echo "=== [0/2] phase-2 pool gen (background, 32 workers) ==="
  python scripts/train_phase2.py --gen-pool-only --pool-cache "$POOL" \
      --scenarios 250000 --workers 32 \
      --horizon 50 --buckets 25 50 \
      --waist-upright-ease-wp 5 \
      --drum-tilt-deg "$TILT_DEG" \
      --drum-radius-frac "$RADIUS_FRAC" \
      --waist-aim-future-tau "$AIM_TAU" \
      --hit-time-jitter-s "$JIT_T" \
      --hit-loudness-jitter "$JIT_V" \
      --rare-drum-boost "$RARE_BOOST" \
      --pair-frac "$PAIR_FRAC" \
      > "runs/poolgen_${POOL_TAG}.log" 2>&1 &
  POOL_GEN_PID=$!
fi

if [[ "$SKIP_P1" == "1" ]]; then
  echo "=== [1/2] phase 1 -- SKIPPED, reusing $P1/best.pt ==="
  test -f "$P1/best.pt" || { echo "no $P1/best.pt to reuse" >&2; exit 1; }
else
echo "=== [1/2] phase 1 -- 19 M APEX warm start, idle arm ON  ($(date +%H:%M)) ==="
# The APEX builder, not the straight-line one.  The imitation TARGET is the
# apex three-point spline; the model's INPUT is a 50/50 per-sample coin flip
# between the noised apex and the noised straight-line quintic to the same
# terminal (``--init-mode mix``).  Pure straight->apex would make the target a
# deterministic function of the context alone and the model would learn to
# ignore its init, destroying the refinement-operator property an MPC warm
# start depends on.  Recorded A/B: mix best_val 0.06224 vs straight 0.08117.
#
# The idle arm is NOT apex: apex_warmstart_trajectory calls the base builder
# with both hands idle -- which, with a home table, parks both arms on a
# straight-line quintic ease-to-home -- and the apex spline then overwrites
# only the columns of a hand that HAS a goal.  Straight line to park, apex to
# the drum.
# 2026-08-13 additions:
#   * --waist-sp-head: the learned waist-yaw set-point (zero-init residual on
#     the analytic aim law; phase 1 trains it through the conditioning token
#     only, no loss change -- docs/waist_sp_head_plan.md)
#   * --waist-upright-ease-wp 5: seed AND target pitch/roll ease from the
#     sampled lean to upright (velocity-matched quintic), replacing the
#     "stay leaned" prior the phase-2 upright well used to fight
python scripts/train_phase1_apex.py \
    --device "$DEV" \
    --dec-layers 6 \
    --steps 20000 \
    --batch 1024 \
    --horizon 50 --buckets 25 50 \
    --train-scenarios 500000 --val-scenarios 40000 \
    --workers 48 \
    --init-mode mix --init-mix 0.5 \
    --waist-sp-head \
    --waist-upright-ease-wp 5 \
    --waist-aim-future-tau "$AIM_TAU" \
    --hit-time-jitter-s "$JIT_T" \
    --hit-loudness-jitter "$JIT_V" \
    --out "$P1"
fi

if [[ -n "$POOL_GEN_PID" ]]; then
  echo "=== waiting for background pool gen (pid $POOL_GEN_PID) ==="
  # A failed background gen is NOT fatal: phase 2's own load_or_make_pool
  # regenerates inline when the cache is absent -- slower, never wrong.
  wait "$POOL_GEN_PID" || echo "pool gen FAILED (see runs/poolgen_${POOL_TAG}.log); phase 2 will regenerate inline"
fi

echo
echo "=== [2/2] phase 2 -- cost descent  ($(date +%H:%M)) ==="
# The collision MEAN is deliberately kept alongside the peak, mirroring the
# torque group (w_torque 20 mean + w_torque_peak 2 smooth-max).  The peak
# reduces the raw PENETRATION and is identically zero on a clear element, so
# it is silent on ~2 000 of the ~2 100 (knot, point, drum) pairs per sample --
# the mean is the only dense collision gradient the approach phase gets.
# "--w-collision 0" is the peak-only variant if that A/B is wanted.
#
# 2026-08-13, after runs/phase2_19m_waist_sp diverged (loss +70 % off a
# step-2 500 minimum, rolling acceptance 8/81 -> 2/81 by step 6 000):
#
#   * --waist-pr-box 0.25 (was 0.15).  0.15 rad was tighter than the pool's
#     OWN p90 peak |waist pitch/roll| -- 0.148 rad on the 25-knot bucket,
#     0.178 on the 50 -- so 10.8 % / 26.3 % of rows were outside the box at
#     knot 0, the MEASURED q0 the optimizer cannot move.  A constraint the
#     data violates on arrival leaves a residual no amount of descent clears,
#     and the augmented Lagrangian integrates that residual forever.  0.25 is
#     ~p99 of the 50-bucket (0.253), i.e. a box the seeds actually satisfy.
#     NOTE this buys the torso ~14 deg of lean: if that is more than the
#     hardware should do, tighten it back and instead pay for the reach
#     somewhere else (--w-waist-yaw-vel is the term that made the arms, not
#     the torso, buy strike speed -- arm torque went 24 -> 156, 37 % of cost).
#   * --ramp-start 10000 --ramp-end 30000 (was 2000 / 20000).  The limit
#     hinges began ramping at step 2 000, which is exactly where the loss
#     bottomed and where d_pos_err was still FLAT at 0.10 m -- the model had
#     not learned to reach before the constraints started closing.  Not
#     --stage-a-gate: stage_a_open never fired once in 7 000 steps of the
#     diverged run, so that gate would freeze the curriculum outright, which
#     the WeightSchedule docstring calls the worse failure on a short run.
#
# Paired with rho["joint_box"] 100 -> 15 in phase2_al_config (same date, same
# cause).  Judge this run by v_joint_box / v_terminal / acceptance, NOT by L:
# under an augmented Lagrangian L is not comparable across steps.
#
# 2026-08-13 round 2, off the _b run's step-26 000 breakdown.  _b fixed what it
# set out to fix -- ok_joint_limits 0.148 -> 0.975, ok_torque 0.198 -> 0.963
# with tau_excess_p90 exactly 0, acceptance 8/81 -> 41/81 -- which left only
# two gates binding: ok_hit 0.716 (contact_pos_err_mm_p50 = 60.8 mm) and
# ok_collision 0.741 (pen_max_mm_p90 = 29.8 mm).  Their product, 0.53, is the
# whole shortfall.  Meanwhile 58 % of the objective was going to constraints
# that already PASS (c_torque 36.6 %, c_posture 14.9 %, c_joint_box 6.9 %)
# against 22 % for the terminal term that was failing.  So:
#
#   * --w-waist-upright 100 (was 1000).  14.9 % of the budget for a term the
#     paired ablation ranks BELOW doing nothing: arm "direct" (upright + box +
#     yaw-rate) scores 0.346 where "base" (yaw-rate only) scores 0.457, and
#     "freeze" (pitch/roll removed outright) collapses to 0.123.  The lean is
#     how the robot reaches; pricing it away costs acceptance.  Kept non-zero
#     rather than dropped because the torso still has to be stabilised for
#     hardware -- that is a balance question the acceptance grid cannot see.
#   * --w-collision-peak 2000 (was 500).  The MEAN collision term is divided by
#     ~2 000 point-drum pairs and is already at an effective 1.27e6 without
#     closing a 29.8 mm p90 penetration; the PEAK is the undiluted reduction
#     and is the one that actually prices penetration depth.  Raise the half
#     that can move the number.
#   * lam_max["torque"] = 3.0 in phase2_al_config -- see the note there.
#
# Three changes at once, so a regression here is not attributable without a
# follow-up A/B.  The paired baseline is runs/phase2_19m_waist_sp_b.
#
# 2026-08-13 round 3: TILT_DEG=3.  Per-drum head-normal tilt, 3 deg of cone
# half-angle (realised: p50 2.1, max 3.0).  This is the only KitRandomization
# knob that changes the model's INPUT distribution rather than just the
# geometry -- ``drum_features`` has always carried the normal (3) and the
# scalar tilt angle (1), 4 of DRUM_FEAT_DIM = 9, but every run before today
# held them constant, so the encoder had no reason to read them.  NOTE the
# phase-1 warm start being reused here was itself trained flat-normal; phase 2
# trains the encoder too, so it can learn to read those 4 features, but the
# warm start does not already know them.
#
# 2026-08-13 round 4 -- the waist, settled by MEASUREMENT rather than tuning.
# Arm-only IK (``correct_tip`` excludes the waist) to the randomized tilted
# goals, waist PINNED, each row scored by its best lean direction:
#
#     lean 0.00 rad -> tip error p90 11.6-18.1 cm, 27-39 % miss by >1 cm
#     lean 0.25 rad -> p90  1.5- 3.5 cm, 11-17 %
#     lean 0.52 rad -> p90  0.3- 1.7 cm,  6-12 %   <- the URDF limit
#     lean 0.59 rad -> p90  0.2- 1.4 cm,  6-12 %   <- what _d actually did
#     lean 0.69 rad -> p90  0.2- 1.6 cm,  5-12 %   <- _d's peak
#
# So the lean IS reach -- upright roughly triples the miss rate -- and the
# benefit SATURATES at the URDF limit: 0.52 -> 0.69 moves nothing outside noise.
# Every run so far was therefore fighting the wrong battle, pricing lean below
# 0.52 (a box of 0.15, then 0.25) while NOTHING enforced the physical limit.
# _d ran lam_joint_box to its 100 ceiling, spent 18.3 % of the objective, and
# still emitted a peak lean past 0.520 rad on 94 % of logged steps.
#
#   * --waist-pr-box 0.52: stop penalising lean the task genuinely needs; the
#     box now states the physical truth and nothing tighter.
#   * --clamp-traj-to-box: enforce that truth STRUCTURALLY, straight-through,
#     because the soft hinge has now demonstrably lost at full multiplier.
#   * --w-waist-pr-vel 20: the missing half of the waist story.  Position is
#     bounded, rate was not, so a plan could whip pitch/roll across its range
#     and pay only for where it ended up.  Since reach saturates at the limit, a
#     lean is worth paying for as a POSTURE and worth nothing as MOMENTUM.
#     Started at the yaw term's 20 so ``c_waist_pr_vel`` and ``c_waist_yaw_vel``
#     are directly comparable in metrics.jsonl.
#   * --w-waist-upright 100 unchanged: the gentle posture preference, which the
#     ablation says to keep small.
#
# 2026-08-14: --lr-horizon 40000 against --steps 60000.  The cosine denominator
# was the STEP CAP, but no run has ever reached it -- every one of the five
# 19m_waist_sp runs ended on the PLATEAU counter (8 800 killed, then 26 000,
# 48 000, 40 000, 30 000), never on --max-hours.  Decaying on a horizon you
# never reach freezes the model at a mid-schedule LR (_b ended at 6.3e-5, 63 %
# of peak), which is the worst place to leave an augmented Lagrangian: the
# multipliers integrate a violation the primal is still too hot to clear.
# 40 000 sits just above the median observed stop, so the long runs now end at
# the min_lr_frac floor the way the schedule intends.  It does NOT fix the
# short ones -- a 26 000-step plateau still ends near 35 % of peak.  The
# plateau counter, not the LR schedule, is what decides when these runs stop.
#
# 2026-08-14, BPTT pinned EXPLICITLY at step 40 000 (--bptt-start-frac 0.6667):
# the default (0.67) put onset at 40 200 as a side effect of the step cap, so
# any change to --steps would silently move it.  With the grooves corpus the
# runs are expected to go longer, so 40 000 should actually be reached --
# BPTT has fired exactly once in this family (_c, last 7 350 steps; loss
# jumped 550 -> 622 at onset, acceptance had peaked BEFORE it, 81-case noise
# means that indicts nothing).  KNOWN TENSION, accepted for now: with
# --lr-horizon 30000 the LR at 40 000 is already the 5 %% floor (5e-6), so
# BPTT trains at floor LR.  If it should bite harder, raise --lr-horizon or
# min_lr_frac; if it turns out useless, --bptt-max-frac 0 it.
# RESUME=<phase-2 ckpt> fine-tunes that checkpoint instead of starting cost
# descent over from the phase-1 warm start.  --reset-al always rides along: the
# multipliers are integrators calibrated against the constraint set they ran
# under, so carrying them across a run that CHANGES a constraint (the waist box
# went 0.25 -> 0.52 here) prices the new constraint with the old constraint's
# lambda.  They cannot fade on their own either -- at decay 0.999 every 100
# steps, _d's lam_joint_box of 98.7 needs ~3e5 steps to reach 5.
# Only ``joint_box`` is reset: its box moved 0.25 -> 0.52 so its multiplier is
# calibrated against a constraint that no longer exists.  Every other group's
# constraint is UNCHANGED and its lambda is hard-won -- resetting them all took
# lam_terminal 38.7 -> 1.0 on the first attempt and dropped acceptance
# 56/81 -> 0/81 by step 2 000.  Override with RESET_AL="a b c" (empty = all).
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME" --reset-al ${RESET_AL-joint_box})
fi
python scripts/train_phase2.py \
    --device "$DEV" \
    --init "$P1/best.pt" \
    "${RESUME_ARGS[@]}" \
    --steps 60000 \
    --batch 512 \
    --lr 1e-4 \
    --lr-horizon 30000 \
    --horizon 50 --buckets 25 50 \
    --scenarios 250000 --workers 48 \
    --pool-cache "$POOL" \
    --drum-tilt-deg "$TILT_DEG" \
    --drum-radius-frac "$RADIUS_FRAC" \
    --waist-aim-future-tau "$AIM_TAU" \
    --hit-time-jitter-s "$JIT_T" \
    --hit-loudness-jitter "$JIT_V" \
    --rare-drum-boost "$RARE_BOOST" \
    --pair-frac "$PAIR_FRAC" \
    --plateau-evals "$PLATEAU_EVALS" \
    $( [[ "$SAT_EFFORT" == "1" ]] && echo --plant-saturate-effort ) \
    --no-compile \
    --w-torque-peak 2.0 --torque-peak-ramp 1000 \
    --w-track-torque "$W_TRACK_TORQUE" \
    --w-track-torque-peak "$W_TRACK_TORQUE_PEAK" \
    --w-collision-peak 2000 --collision-peak-ramp 1000 \
    --w-stick-stick "${W_STICK_STICK:-100}" \
    --w-waist-upright 100 --waist-pr-box 0.52 --w-waist-yaw-vel 20 \
    --w-waist-pr-vel 20 --clamp-traj-to-box \
    --waist-ramp 300 \
    --ramp-start 10000 --ramp-end 30000 \
    --bptt-start-frac 0.6667 \
    --waist-sp-head --w-waist-aim 500 --waist-aim-deadband 0.15 \
    --w-waist-sp-tether "${TETHER:-50}" \
    --waist-sp-search-frac "${SP_SEARCH_FRAC:-0}" \
    --w-waist-sp-search "${W_SP_SEARCH:-0}" \
    --waist-upright-ease-wp 5 \
    --cmd-lead fit_kd4 \
    --eval-every 2000 --eval-cases 81 --ckpt-every 2000 \
    --probe-every 2000 --probe-cmd-lead fit_kd4 \
    --target-acceptance 0.95 \
    --max-hours 14 \
    --out "$P2"

echo
echo "=== done  ($(date +%H:%M)) ==="
echo "phase 1: $P1"
echo "phase 2: $P2"
