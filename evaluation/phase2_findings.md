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