# Distributional critic CPU ablation — 2026-09-16

## Setup

- Fixed actor: `/tmp/timing-actor-only-1000-boundary-loss-4x/last.pt`
- Minimal critic input: 42 values (`q0`, `qd0`, ordered Cartesian goals, two-hand TTH)
- Exact labels: 4,800 rows from 60 actor steps, including ordinary and suffix evaluations
- Chronological split: 3,360 train / 720 calibration / 720 test
- Classifier training: 1,000 updates, batch 256
- Seeds: 123, 456, 789
- All metrics below are post-temperature-scaling mean ± population SD
- `F`/`P`: factorized/perturbed synthetic OOD sets

Only 7.9% of test rows are manageable. Consequently, fixed-threshold safe
accuracy is not useful: most arms predict almost no manageable rows at a 0.5
cutoff. `safe AUROC` measures ranking independent of that arbitrary cutoff.

| arm | bin acc ↑ | NLL ↓ | safe AUROC ↑ | Brier ↓ | entropy F ↑ | entropy P ↑ | disagreement F ↑ | disagreement P ↑ | distance F ↑ | distance P ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Softmax 1h plain | .571±.003 | 1.326±.018 | **.911±.002** | **.057±.000** | .508±.007 | .482±.004 | — | — | — | — |
| Softmax 1h OE | .572±.007 | 1.357±.044 | .896±.002 | .060±.000 | .553±.008 | .487±.004 | — | — | — | — |
| Softmax 3h plain | **.577±.001** | **1.291±.002** | .907±.001 | .058±.000 | .513±.011 | .484±.004 | **.635±.022** | **.567±.011** | — | — |
| Softmax 3h OE | .573±.003 | 1.306±.006 | .899±.002 | .060±.001 | .556±.013 | .487±.005 | .632±.023 | .560±.010 | — | — |
| IsoMax 1h plain | .540±.001 | 1.342±.002 | .877±.005 | .066±.000 | .520±.013 | .526±.012 | — | — | .568±.012 | .552±.007 |
| IsoMax 1h OE | .544±.002 | 1.337±.002 | .862±.013 | .066±.001 | .570±.010 | **.542±.010** | — | — | .620±.013 | .566±.005 |
| IsoMax 3h plain | .540±.001 | 1.343±.001 | .878±.002 | .066±.000 | .528±.010 | .526±.015 | .589±.033 | .523±.013 | .569±.004 | .552±.008 |
| IsoMax 3h OE | .543±.001 | 1.337±.002 | .878±.006 | .066±.000 | **.576±.009** | .541±.014 | .623±.028 | .531±.006 | **.625±.000** | **.567±.007** |

## Decision

Use the **three-head plain Softmax ensemble** as the production candidate. It
has the best violation-bin prediction, best NLL, near-best feasible ranking,
and the strongest broadly useful ensemble-disagreement OOD signal. Retain the
ensemble; one Softmax head has no useful OOD score.

Do not promote the current high-entropy outlier exposure yet. It improves
entropy separation for factorized OOD but does not improve perturbed OOD, and
slightly worsens ID NLL, Brier score, safe ranking, and ensemble disagreement.
Likewise, IsoMax produces a useful prototype-distance OOD score but loses too
much ID accuracy/calibration to justify replacing Softmax in this test.

Choose the operational safe threshold on calibration data for an explicit
false-safe budget. A probability cutoff of 0.5 is invalid under the observed
7.9% manageable prior.

## Fresh global entropy prior

A follow-up replaced the fixed OOD bank with freshly generated global tasks on
every critic update.  Each hand independently chooses uniformly from its six
legally reachable drums plus idle; this yields approximately 73% dual-hit, 12%
left-only, 12% right-only, and 2% both-idle tasks without special mode weights.
Active hand times are independent draws over 0.08–1.0 seconds.  Joint states
span the bounded posture/velocity domain, while goal positions include legal
head contacts and the configured kit-position jitter.  Each ensemble member
receives an independent global stream.

The experiment used 1,024 fresh global tasks per 256 exact labels.  Three-head
Softmax means over three seeds were:

| global entropy weight | bin acc ↑ | ID NLL ↓ | safe AUROC ↑ | global entropy AUROC ↑ |
|---:|---:|---:|---:|---:|
| 0 (plain) | .577 | 1.291 | .907 | .208 |
| .001 | .576 | 1.282 | .904 | .347 |
| **.0025** | .575 | **1.276** | .899 | .606 |
| .005 | .572 | 1.276 | .892 | .859 |
| .01 | .569 | 1.272 | .886 | .953 |

The ensemble's global disagreement is already strong and rises further with
the global prior (about .97 AUROC near weight .0025).  Weight `.0025` is the
recommended starting point for gradual return to uncertainty: it establishes
a measurable entropy prior while preserving most ID ranking.  Larger weights
are valid if fast forgetting matters more than current feasible-set ranking.
This is still an offline critic ablation; production integration should log
age-stratified sentinel entropy to measure the actual forgetting timescale.

### Held-out-region aging test

To test that timescale directly, a follow-up first trained on all exact labels,
then withheld a spatial subset of 297 calibration tasks for 2,000 critic
updates.  The remaining 423 tasks stayed active.  Fresh global samples were
still drawn every update for the entropy-prior condition.

| entropy weight | stale entropy, age 0 | stale entropy, age 2,000 | stale NLL, age 2,000 | active entropy, age 2,000 |
|---:|---:|---:|---:|---:|
| 0 | .469 | .356 | 2.096 | .362 |
| .0025 | .482 | .394 | 1.917 | .393 |

The global prior slows overconfidence and improves stale-region NLL, but it
does **not** cause old points to return to maximum entropy: entropy fell in
both conditions as training continued.  Uniform Monte Carlo points are too
sparse in this 42-dimensional input space to implement a dependable local
expiration clock.  Use the global prior as a low-weight background OOD
regularizer; if explicit recency forgetting is required, add a small set of
age-stratified sentinels or a replay-age-dependent entropy target rather than
expecting global random coverage alone to find every stale neighborhood.

### Distribution quality and selective prediction

A longer selected-model comparison used 2,000 updates, 256 exact replay rows,
and 1,024 freshly generated global entropy anchors in every update.  Results
below are temperature-calibrated means across seeds 123, 456, and 789.

| metric | plain | global entropy `.001` | global entropy `.0025` |
|---|---:|---:|---:|
| all-distribution bin accuracy ↑ | **.599** | .593 | .589 |
| final worst-violation bin accuracy ↑ | **.670** | .663 | .651 |
| all-distribution NLL ↓ | 1.321 | 1.297 | **1.270** |
| final worst-violation NLL ↓ | **.884** | .905 | .950 |
| categorical Brier ↓ | **.518** | .522 | .525 |
| ranked probability score ↓ | **.085** | .086 | .088 |
| safe AUROC ↑ | **.926** | .916 | .905 |
| safe ECE ↓ | .022 | **.020** | .028 |
| entropy/error AUROC ↑ | **.859** | .851 | .852 |
| selective final accuracy at 80% coverage ↑ | **.773** | .761 | .750 |
| selective final accuracy at 50% coverage ↑ | **.916** | .903 | .890 |
| global entropy AUROC ↑ | .185 | .514 | **.831** |
| global disagreement AUROC ↑ | .956 | .981 | **.982** |

The final-output entropy is operationally meaningful in both models: rejecting
the highest-entropy half raises final-bin accuracy from roughly 65–67% to
89–92%.  Fresh global entropy training therefore succeeds at its primary OOD
objective and preserves good selective ranking, but weight `.0025` is not a
free improvement.  It degrades exact final-violation prediction and safe-set
ranking on this dataset.  Weight `.001` is the better compromise: it recovers
roughly half of the `.0025` final-accuracy loss, retains 90.3% accuracy at 50%
selective coverage, improves safe ECE slightly, and moves global entropy AUROC
from .185 to .514.  Use `.001` as the initial production value while preserving
the same large fresh-anchor batch; annealing remains worth testing once the
online exact-label stream is available.

### Combining predictive entropy and ensemble disagreement

For the `.001` model, two fixed combinations were evaluated without tuning on
OOD labels.  Each constituent score was standardized against ID predictions;
the combinations were either their sum or their maximum (an OR-style score).
Temperature-calibrated three-seed means were:

| OOD family | entropy | disagreement | standardized sum | standardized OR |
|---|---:|---:|---:|---:|
| global random | .514 | **.981** | .934 | .971 |
| factorized | .502 | **.701** | .607 | .573 |
| locally perturbed | .481 | **.599** | .513 | .496 |

Neither fixed combination improves OOD identification.  Entropy remains useful
for selective prediction on points resembling the exact-label distribution,
whereas head disagreement is the stronger OOD score.  Combining them into one
scalar dilutes disagreement, especially for factorized and locally perturbed
inputs.  Production should use separate gates: disagreement for unfamiliarity,
then categorical entropy for expected prediction reliability among points that
pass the familiarity gate.

### Optimization length and learning rate at entropy weight `.001`

The weak entropy result at 2,000 updates was partly optimization-limited. The
same three seeds were rerun with twice as many updates and with a three-times
larger AdamW learning rate. All other batch and loss settings were unchanged.

| configuration | final bin acc ↑ | final NLL ↓ | Brier ↓ | safe AUROC ↑ | selective AURC ↓ | global entropy AUROC ↑ | global disagreement AUROC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k, LR `1e-3` | .663 | .905 | .522 | .916 | .118 | .514 | **.981** |
| 4k, LR `1e-3` | .693 | .896 | .503 | .915 | .103 | .659 | .973 |
| **2k, LR `3e-3`** | **.714** | **.883** | **.494** | **.920** | **.094** | **.670** | .979 |

Thus the original run had not fully fit either the exact or entropy objectives.
Higher LR reached a better solution in half as many updates, although final NLL
was more seed-variable (`.883 ± .077`) than at LR `1e-3`. Extra optimization
does not remove the objective tradeoff completely: global entropy AUROC levels
off near .66–.67 at weight `.001`, versus .831 at weight `.0025`. A production
run should start near LR `3e-3`, decay it, and choose checkpoints from a held-out
composite of final NLL, safe AUROC, selective AURC, and global entropy AUROC.

### Stronger entropy after fixing optimization

Repeating the LR `3e-3`, 2,000-update experiment at entropy weight `.0025`
reverses the degradation seen with the earlier, under-optimized LR `1e-3` run:

| metric | weight `.001` | weight `.0025` |
|---|---:|---:|
| all-distribution bin accuracy ↑ | .632 | **.636** |
| final worst-violation bin accuracy ↑ | .714 | **.720** |
| all-distribution NLL ↓ | 1.287 | **1.225** |
| final worst-violation NLL ↓ | .883 | **.883** |
| categorical Brier ↓ | .494 | **.490** |
| safe AUROC ↑ | .920 | **.922** |
| safe ECE ↓ | **.0151** | .0155 |
| selective AURC ↓ | **.0941** | .0945 |
| global ID mean entropy | .446 | .452 |
| global OOD mean entropy | .531 | **.672** |
| global entropy AUROC ↑ | .670 | **.897** |
| global disagreement AUROC ↑ | **.979** | .965 |
| standardized entropy + disagreement AUROC ↑ | .972 | **.980** |

The stronger entropy target now improves OOD uncertainty without sacrificing
exact prediction.  Once entropy becomes informative, the standardized sum also
beats disagreement alone; the earlier conclusion that combination always
dilutes disagreement applies only to the weak `.001` entropy model.  For this
fixed offline dataset, LR `3e-3`, entropy weight `.0025`, 1,024 fresh anchors per
256 exact rows, and 2,000 updates is the best tested configuration.  The small
drop in disagreement is outweighed by the much larger entropy gain.

### Local entropy radius profile

To test whether exact rows become isolated low-entropy islands, each held-out
row was perturbed along one randomly chosen feature group using matched rows and
directions at every radius.  Radius 1 corresponds to RMS perturbations of `.14`
in normalized joint position (about `.44 rad`), `.16` in normalized velocity
(about `1.6 rad/s`), `.12 m` in goals, or `.20 s` in TTH.  Only one group is
changed per row.

| radius | mean entropy | entropy AUROC | disagreement AUROC | combined AUROC |
|---:|---:|---:|---:|---:|
| 0 | .456 | .509 | .489 | .504 |
| .1 | .456 | .509 | .495 | .505 |
| .25 | .456 | .509 | .525 | .511 |
| .5 | .455 | .505 | .588 | .526 |
| 1 | .452 | .498 | .676 | .563 |
| 2 | .449 | .490 | .756 | .616 |
| 4 | .450 | .490 | .828 | .669 |
| independent global | .672 | .897 | .965 | .980 |

There is no immediate entropy sea around exact points: categorical entropy is
flat throughout a broad local neighborhood.  This supports interpolation, but
also means entropy alone remains confidently extrapolative along these local
directions.  Ensemble disagreement supplies the desired gradual distance
signal.  If a smooth entropy transition is required, train on local shells as
well as global anchors: consistency targets near exact rows, then progressively
higher-entropy targets at larger physically meaningful radii.  Assigning a
uniform target immediately outside each row would create precisely the isolated
islands this test rules out.
