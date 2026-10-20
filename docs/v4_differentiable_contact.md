# Differentiable v4 cymbal and hi-hat contact

The v4 Isaac tracker judges physical tip/shaft **probe centres** in each drum's
frame. Let (n) be the outward head normal, (c) the physical top centre,
(R) the drum radius, (H) its 10 cm body height and (r_j) probe (j)'s
radius (10 mm tip, 8 mm shaft). For a probe at (p_j), write

\[
z_j=(p_j-c)\cdot n,\qquad
\rho_j=\|(p_j-c)-z_j n\|,\qquad
v_n=-v_j\cdot n,\qquad
v_t=\|v_j+v_n n\|.
\]

Isaac qualifies a fresh top entry when that *same probe* descends across the
top plane with previous (z_j\ge r_j-2\) mm, current (z_j\le r_j+6\) mm,
at least 0.5 mm descent and radial positions inside (R+2\) mm. Its entry
latches only during the current force-bearing contact episode and can then
continue through the physical body. A side entry needs current
(z_j\in[-5,+2]\) mm and
(\rho_j\in[R-5\text{ mm},R+r_j+3\text{ mm}]\). Both entry routes also need
downward incidence (v_n>0,\;v_t\le1.5v_n\). The contact tracker needs real
PhysX force above 1 N. Finally, the requested impact speed is
(s_\star=1+2\times\text{normalized loudness}\) m/s. Its shape-1 speed quality
is (q(x)=x\exp(1-x)), (x=v_n/s_\star); a beat at its deadline needs
quality at least 0.25 to complete.

`FusedCost` cannot backpropagate PhysX force or an actual bounce. Its v4 mode
therefore uses a conservative **contact proxy**: the physical stick capsule
SDF must be within 3 mm of the drum. It evaluates the same tip and shaft
sample points/radii as its collision model, classifies top entries per probe,
and retains the certificate only through that SDF contact episode. Only a
certified top or upper-side contact within the near-hit window exempts the
target stick from the drum-collision penalty. Wrist and other-body collisions
remain priced. A sampled trajectory below the side band or approaching nearly
horizontally receives a differentiable distance toward the closest qualifying
top/side route. This is a surrogate for the *direction* of repair, not a claim
that SDF proximity guarantees force or hit credit.

At the terminal knot the contact loss is a soft minimum over physical probes
and the union of two nonnegative metre-valued distances. The top branch has
hinges for previous/current axial and radial crossing, downward movement and
incidence; the side branch has hinges for its narrow axial/radial sliver and
incidence. The normal/tangential velocity loss also prices
([v_t-1.5v_n]_+^2) and the speed-quality deficit
([0.25-q(v_n/s_\star)]_+^2). Top-face entry may be anywhere on the full
cymbal disk; the older near-side aim remains a *soft preference* only. Pads
retain their legacy tip-only contact region.

The auxiliary terminal-pose objective has only one pose and therefore cannot
infer a sweep, force or impact speed. Its v4 mode scores static tip/shaft
centres against the physical top-face or upper-side contact regions and removes
the older cymbal half-plane. Fresh phase-2 training upgrades an archived
terminal-role checkpoint to that pose geometry and resets its contact AL
multipliers. Archived `--resume` paths preserve their saved objective;
changing the contact mode requires a new run initialized from weights.

The hard geometry and smooth residual are covered by
`tests/test_target_contact.py`, their FusedCost integration and gradients by
`tests/test_cost.py`, and the auxiliary pose/role migration by
`tests/test_terminal_pose.py` and `tests/test_terminal_finetune.py`.
