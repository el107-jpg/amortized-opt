# Continuous playing-shaft hit scoring (v5)

The physical stick is unchanged: a 40 cm shaft of radius 8 mm and a 10 mm
tip bead. Only the distal 20 cm from the wrist mount to the bead may score a
cymbal or hi-hat. Membrane drums still use the bead alone. The cymbal/hi-hat
colliders remain 10 cm thick.

For each edge drum and physics substep, the Isaac tracker considers a material
fraction `s` along the playing shaft. It builds candidate fractions from every
boundary of the top-face, upper-side, and incidence inequalities, then tests
one interior point of every resulting interval. The tip bead is tested
separately at its 10 mm radius. This detects legal contact between the old
fixed probe locations. Contact still requires a real PhysX force above 1 N;
normal pre-impact speed and the downstream requested-speed/timing quality
determine whether a drum beat receives credit. Top entry is tied to its shaft
fraction for the duration of that contact episode. Side entry is limited to
5 mm below and 2 mm above the head. A 1.5:1 tangential-to-downward-normal
speed ratio remains the incidence limit.

The differentiable `isaac_v5` loss evaluates the same continuous witnesses.
It uses body signed distance within 3 mm as a proxy for force and therefore
does not exactly predict PhysX contact. Candidate fractions are treated as
hard active-set witnesses during backpropagation; gradients pass through the
selected material points, avoiding singular derivatives at grazing quadratic
roots. The separate pose-only terminal role uses static capsule geometry and
does not infer velocity or force from a single pose.

The Isaac rollout runtime is
`distal20cm_capsule_top_side5mm_contact_v5`. The drum USD and layout hashes
remain in worker provenance. v4 outcomes are incompatible with v5 labels;
`scripts/collect_v5_prefirst_dataset.py` collects fresh v5 labels. Fresh
phase-2 and terminal-pose training default to `isaac_v5`. Archived resumes
retain their saved mode; objective changes require a fresh run initialized
from weights with new optimizer and AL state.
