## Failure mode analysis (5 rendered episodes, novel-color condition)
Visual inspection of rendered rollouts (not just action logs) revealed a
more precise pattern than raw action counts suggested:

- 3/5: Agent successfully navigates close to the grey target object
  (confirming navigation/vision is still locating it correctly), then
  enters a persistent `right`-turn loop directly adjacent to the object,
  never invoking `pickup`, for the remainder of the episode. The loop does
  not resolve even as the egocentric view changes slightly with each turn.
- 2/5: Agent quickly grabs a different, incorrect object (fast, confident,
  no hesitation) — resembling pre-instruction-conditioning "nearest object"
  behavior.

## Interpretation
The failure is not in navigation broadly, but appears localized to the
interaction decision at close range. The agent's vision/spatial policy
still successfully locates and approaches an object regardless of color,
suggesting the CNN backbone generalizes fine. The breakdown occurs at the
point where the mission signal should resolve "is this the target, pick
it up" — consistent with FiLM's modulation parameters, generated from an
out-of-distribution color embedding, producing a corrupted but STABLE
gating signal that traps the policy in a fixed loop rather than degrading
into random/erratic behavior. This is a more specific claim than generic
"noise breaks the policy": it points to the interaction/decision head
being the failure locus, not perception or locomotion.

This still does not implicate FiLM as an architecture broadly (compositional
generalization on novel combos remained strong), but does suggest that for
target colors never seen in a supervised role, the fusion module needs
either (a) a pretrained text encoder with prior color structure, or (b) an
explicit fallback/uncertainty mechanism so an out-of-distribution mission
degrades to a safe default (e.g. continue searching) rather than a locked
loop.

## Next Move
- Using a pretrained text embedding; specifically sentence-transformers with the all-MiniLM-L6-v2 model
- It produces a single 384-dim embedding for the whole mission sentence, computed once per episode (since mission doesn't change mid-episode), not per-word. 
- This also simplifies things: no more vocab file, no more token IDs — colors and objects inherit real semantic structure from pretraining instead of starting from scratch.

## Follow-up: Pretrained Text Embeddings (MiniLM)

### Motivation
The novel-color failure above showed a specific mechanistic signature (policy
locks into a stable oscillation loop near the target), consistent with a
from-scratch nn.Embedding producing ungrounded, out-of-distribution FiLM
modulation parameters for a color never seen in the target role. Hypothesis:
a pretrained sentence encoder (MiniLM, 384-dim) would give "grey" semantic
proximity to other colors via pretraining, potentially producing more
graceful degradation.

### Results across training duration

| Checkpoint | Seen accuracy | Novel-combo accuracy | Novel-color (grey) accuracy | Grey pickup rate |
|---|---|---|---|---|
| ~400k steps | 70.7% | 51.4% | 45.8% | 83% |
| ~700k steps | 80.8%* | — | 0.0%* | — |
| ~900k-1.1M steps | 92-93% | 92.2% | 0.0% | 28% |

*From the checkpoint-search run (30-episode evals, noisier than the 100-episode
standalone evals used for the 400k row).

### Key findings
1. **The catastrophic failure mode (spinning lockup) was fixable**: at the
   400k checkpoint, pickup rate on novel-color episodes rose from 43%→83%
   and accuracy from 0%→45.8%, confirming the hypothesis that pretrained
   embeddings can prevent the corrupted-modulation lockup behavior.
2. **This capability was unstable, not a stable improvement**: continued
   training pushed seen/combo accuracy above even the original from-scratch
   baseline (93% vs 89.6% seen, 92% vs 68.9% combo) but novel-color accuracy
   collapsed back to 0% by 700k steps and stayed there.
3. **Run-to-run variance was substantial**: a second training run evaluated
   at a similar step count to the first did not reproduce the 45.8% result,
   suggesting the phenomenon is sensitive to training stochasticity, not a
   deterministic function of step count alone.

### Interpretation
Pretrained text embeddings are not a complete or reliable fix for zero-shot
generalization to attributes never observed in a supervised (target) role.
They can, under some training conditions, prevent the worst failure mode
(policy lockup) and produce partial generalization, but standard PPO
fine-tuning has no mechanism to preserve this property once optimization
pressure favors specializing to the training distribution. A more reliable
fix would likely require: (a) explicit regularization or an auxiliary loss
that rewards smooth behavior across embedding space, (b) multi-seed training
with checkpoint selection based on held-out accuracy rather than training
reward, or (c) architectural constraints that prevent FiLM parameters from
diverging far from identity for out-of-distribution text inputs. These are
noted as directions for future work rather than pursued further here, given
project scope.