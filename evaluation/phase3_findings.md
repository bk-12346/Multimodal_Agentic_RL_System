# Phase 3 — Agentic Layer Findings

## Confidence-based failure detection: limited applicability

Action-probability-derived confidence scoring was implemented as a
failure-detection signal (low, sustained uncertainty triggers early abort).
Testing on a real multi-step plan revealed that the dominant failure mode
in this system -- grabbing the wrong object due to weak instruction
grounding -- produces HIGH confidence (0.90), statistically
indistinguishable from successful attempts (0.92, 0.96). The policy commits
fully to incorrect actions rather than hesitating.

This means confidence-based abort is only useful against a different
failure class (genuine navigation confusion/uncertainty), not the
mission-grounding failures characterized in Phase 2. A more effective
failure-detection signal for THIS failure mode would need to inspect
something the policy's action distribution doesn't expose -- e.g. a
learned verifier/critic trained specifically to detect target mismatches,
or (once implemented) an LLM planner that can reason about the outcome
post-hoc rather than relying on the RL policy's own self-reported certainty.