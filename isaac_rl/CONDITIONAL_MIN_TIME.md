# Conditional minimum-time planning overlay

The optional `ConditionalMinimumTimeChart` in `conditional_min_time.py` stores
three planning-time anchors per measured or explicitly initialized eligible cell:
left, center and right at proxy values -1, 0 and +1. The quadratic is

```
T(z) = center + (right-left)/2*z + ((left+right)/2-center)*z*z
```

The proxy is the other hand's activity, **not measured waist yaw**. Its stored
per-drum coordinates are the kit's lateral Y coordinates normalized to [-1, 1].
For each scored hand transition, other-hand target intervals run from that
hand's previous scheduled event to its next scheduled hit. Intersect them with
the scored transition, retain the final two intervals with positive overlap,
and average their target coordinates using their retained overlap durations.
A hit after the scored deadline still contributes if movement toward it
already overlaps the scored interval. After the last known target, a parked
interval uses that drum's coordinate; an unknown parked target uses neutral 0.
Parking extends the last target interval; it does not consume another last-two-hit slot. Repeated actual hits remain distinct intervals.

Each candidate's complete target timelines are constructed before any edge is
scored. Simultaneous hits therefore do not depend on member iteration order.
The receding planner supplies full committed hit history: an own-hand transition
may start before several already committed other-hand hits. Direct callers
supplying only last drums/times approximate missing earlier history as movement
toward the last known target from `origin_time` (default 0). Use
`committed_history=[left_events, right_events]`, each containing chronological
`(time, drum_index)` pairs, when that approximation is insufficient. Last history
events must agree with the supplied last state. Context outside the candidate
window is unavailable; the last known other-hand goal is treated as parked.

The original NPZ chart, support counts, unknown/ineligible masks, penalties and
graph tie preferences are preserved. Ready-pose first arrivals remain a separate
scalar measured calibration; they are never relabeled as departures from a drum.
The overlay is a planning correction, **not a feasibility certificate** or a
replacement for the chart's measured minimum times.

Create a flat initial overlay and check exact saved-panel parity:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/init_conditional_min_time_chart.py \
  --table runs/transition_decoder_torque_integral_20261005_2240/collision_5mm/transition_table.npz \
  --out runs/conditional_min_time_chart_20261006/conditional_chart.npz \
  --parity-panel runs/min_time_chart_vs_graph_4_6_groups_20261006/problems.pt
```

To opt in to existing 4/6-group probe planning, add
`--conditional-chart runs/conditional_min_time_chart_20261006/conditional_chart.npz`
to `scripts/probe_min_time_chart_choices.py`, retaining `--table` and any matching
`--ready-initial` calibration. Default callers continue to use the scalar chart.
The probe records the overlay path, base fingerprint, proxy mapping and rule.
Isaac comparison playback consumes those selected hands as before; it does not
need a separate curve evaluator when playing an already assigned schedule.

For CMA-ES or other parameter search:

```python
base = MinimumTimeChart.load(table_path)
chart = ConditionalMinimumTimeChart.load(overlay_path, base)
initial = chart.get_parameters()
labels = chart.parameter_mapping()
chart.set_parameters(candidate)  # ValueError for invalid proposals; unchanged on failure
```

The vector contains left/center/right values in C-order hand/source/destination
order. The table has 85 eligible cells and 63 measured finite cells. An overlay
without initialization has 189 parameters. Initializing the 22 unknown cells
expands this to **255 parameters** without altering the measured base table,
support, trials or hand eligibility. Unknown ready-pose arrivals remain separate.

Create an expanded overlay, optionally retaining existing learned curves:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/init_conditional_min_time_chart.py \
  --table runs/transition_decoder_torque_integral_20261005_2240/collision_5mm/transition_table.npz \
  --from-overlay runs/chart_cma_trial_20261006/best_chart.npz \
  --initialize-unknown --unknown-center-s 0.3 --unknown-half-width-s 0.05 \
  --unknown-seed 20261006 --out runs/conditional_chart_expanded/conditional_chart.npz
```

Each unknown cell receives one independent uniform draw between 250 and 350 ms,
shared by its three initial anchors. CMA can then tune all three. The explicit
`initialized_unknown` mask marks planning guesses; successful-trial counts stay
unchanged, so scoring retains the base support-risk penalty. Heatmaps mark these
entries with `*`. `with_initialized_unknown()` returns a new overlay and preserves
all existing curves. Reapplying it does not rerandomize previously initialized cells.

Format 2 saves this mask and loads legacy format 1 overlays as well. Drum order
and a fingerprint of the base times, support, trials and eligibility must match.
The original running 189-parameter trial keeps its old dimensions. Expanded
overlays require a fresh CMA optimizer, optionally starting from learned curves.

For measured cells, the initial three anchors equal each base time. Without
unknown-cell initialization this preserves scores and decisions exactly.
All learned curves must remain at least the positive control-grid floor
(default 0.02 seconds) throughout [-1, 1]: validation checks endpoints and an
interior minimum, not merely the three anchors. Extrapolation is rejected.
Parameter proposals are validated before mutation; the optimizer should treat
invalid proposals as rejected candidates. This implements the scoring/search
interface. CMA-ES optimization and batched Isaac evaluation are now provided by
`scripts/train_chart_cma.py`; see [CHART_CMA.md](CHART_CMA.md) for launch,
checkpoint/resume, fitness and environment-capacity details.
