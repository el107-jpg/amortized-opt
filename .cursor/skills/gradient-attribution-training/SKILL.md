---
name: gradient-attribution-training
description: >-
  Measure per-loss-term decoder gradient norms during PyTorch training without
  common attribution mistakes. Use when balancing AL/constraint vs objective
  terms, debugging torque/limit gradient dominance, adding grad probes to
  train_cost, or interpreting metrics.jsonl grad_* fields in amortized-opt.
---

# Gradient attribution during training

## What to measure (and what not to)

**Goal:** For each scalar loss piece `L_i`, estimate how much it pushes the
trainable parameters:

\[
\| \nabla_\theta (w_i \cdot L_i) \|_2
\]

**Do not substitute:**

| Wrong proxy | Why it fails |
|---|---|
| `c_torque / cost` (term **value** share) | Pricing mixes warning, `λ×ReLU`, `λ_peak×w×sum`, PHR in one `terms["torque"]` bucket |
| `v_*` violations or `lam_*` | Dual state ≠ gradient magnitude; sparse max vs dense smooth-max ReLU differ |
| Global `grad_norm` alone | Only shows clip saturation; says nothing about term balance |
| `loss.backward()` then read `.grad` | Contaminated sum of **all** terms |
| `raw["torque_peak"]` vs `terms["velocity"]` | Different groups, weights, and pricing paths |

**Orthogonal-decomposition caveat:** Per-term `autograd.grad` sums correctly only
if you treat terms as independent scalars. Shared graph nodes mean fractions are
**indicative**, not exact Shapley values. Good enough for training balance.

---

## Correct procedure

### 1. Fresh forward on the same batch context

Use the **same** `init, state, enc, masks, q0, v0, strike, lengths, weights`
the training step would use (after quintic correction, SP injection, etc.).

Do **not** reuse a graph consumed by `total.backward()`.

### 2. One `autograd.grad` per term

```python
core = module.module if hasattr(module, "module") else module
params = [p for p in core.parameters() if p.requires_grad]

dec = module(init, state, enc, masks)
ct = cost.forward(q0, dec.traj, v0, strike, lengths=lengths, weights=weights)

for k in TERM_KEYS:
    t = ct.terms[k].mean()
    if not t.requires_grad:
        continue
    grads = torch.autograd.grad(
        w_cost * t, params, retain_graph=True, allow_unused=True,
    )
    gn = sum(g.detach().float().pow(2).sum() for g in grads if g is not None).sqrt()
```

Match training weights: multiply by `cfg.w_cost` (and `cfg.w_aux` if probing aux).

### 3. Report norms and fractions

```python
norms = {f"grad_{k}": float(gn)}
total = math.sqrt(sum(v * v for v in norms.values()))
norms["grad_torque_frac"] = norms["grad_torque"] / total  # example
```

Log to `metrics.jsonl` on a **sparse** schedule (`grad_term_every`, e.g. 50),
`rank == 0` only. Cost per probe: **1 extra forward + len(TERM_KEYS) grad calls**.

### 4. Keep training step separate

```text
grad_term_probe = measure_...(…)   # diagnostic forward
loss, metrics, ... = step_fn(...) # normal forward
total.backward()                  # includes aux, tc, bptt, suffix, distill, …
clip_grad_norm_(...)
opt.step()
```

Attach `grad_term_probe` to the **train** log record when present.

---

## amortized-opt reference implementation

| Piece | Location |
|---|---|
| Probe function | `amortized_opt.train_cost.measure_cost_term_grad_norms` |
| Term list | `TERM_KEYS` in `train_cost.py` |
| Training hook | `grad_term_probe` in `train_phase2` loop; `Phase2Config.grad_term_every` |
| CLI | `--grad-term-every N` (0 = off) |
| Log fields | `grad_<term>`, `grad_torque_frac` on train events |

Read logs:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/<tag>/metrics.jsonl")
for line in p.open():
    d = json.loads(line)
    if d.get("event") != "train" or "grad_torque" not in d:
        continue
    print(d["step"], d.get("grad_norm"), d.get("grad_torque_frac"),
          {k: d[k] for k in d if k.startswith("grad_") and k != "grad_norm"})
PY
```

---

## Finer splits (within `terms["torque"]`)

`terms["torque"]` is a **sum** of several mechanisms. To attribute inside it,
grad the **sub-expressions** from `FusedCost.forward`, not the bucket:

| Piece | Priced as | AL group |
|---|---|---|
| Warning | `cfg.w_torque * raw["torque"]` (softplus mean) | none |
| Peak | `λ_peak * w_peak * raw["torque_peak"]` (smooth max) | `torque_peak` |
| Segment hard | `λ_seg * live[seg]` per class | `torque_waist`, … |
| PHR | `0.5 * pen_g * live²` | same segment / peak keys |

Probe each additive term in `forward` if you need “51% segment λ vs 36% peak”
style numbers. `measure_cost_term_grad_norms` only splits **top-level** `TERM_KEYS`.

---

## Common mistakes (checklist)

- [ ] Probing **after** `total.backward()` without a new forward
- [ ] Using **detached** `metrics` from `make_cost_step` (terms are `.detach()` there)
- [ ] Forgetting **`w_cost`** (or `w_aux`) on the scalar before `autograd.grad`
- [ ] Expecting `terms["torque"] == raw["torque_peak"]` (peak includes λ×w; segment λ×live is separate)
- [ ] Running probe **every step** (dominates step time)
- [ ] Interpreting `grad_norm ≈ 1e6` with `grad_clip=1` as “huge torque” — everything is clip-saturated; use **fractions**
- [ ] Tuning from **loss share** while λ is at `lam_max` — phantom pricing from warning or PHR may be tiny in grad but huge in cost
- [ ] Using **hooks** on parameters without `zero_grad` between terms (accumulates)

---

## When balancing training

Prefer, in order:

1. **Dual / λ** — cap (`AMORTIZED_LAM_TORQUE_MAX`), `target` slack, separate peak vs segment groups
2. **Residual shape** — smooth-max ReLU priced + max for AL; segment class-max
3. **Upstream** — `clamp_traj_to_box`, `limit_alpha` ramp
4. **Drop PHR** on chronic groups (`pen_torque=0`)
5. **Grad surgery** (PCGrad, per-term scaling) — only if probe shows sustained gradient **conflict**, not mere dominance

Re-probe after each change; do not assume `c_*` ratios predict `grad_*` ratios.

---

## Adding probes to a new codebase

1. Define stable **term keys** (append-only if logged positionally).
2. Implement `measure_*_grad_norms(model, …, weights)` with `autograd.grad` per key.
3. Call on `step % every == 0`, rank 0, **before** main backward.
4. Log `grad_<key>` + one or two summary fractions.
5. Add a cheap unit test: one violating batch → `grad_<hot_term> > 0`, feasible hold → finite norms.
