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

## LLM Planner (AWS Bedrock, moonshotai.kimi-k2.5)

Replaced the rule-based regex parser with an LLM-based parser (flexible
natural language, not constrained to a rigid template) and added a
post-hoc LLM verification step after each execution attempt.

### Parsing
Successfully parsed a naturally-phrased, non-templated instruction
("I need a blue key first, then a green ball, and finally a red key")
into correctly ordered subgoals -- something the regex-based parser could
not handle without exact keyword/separator matching.

### Verification
On a 5-attempt test run, LLM verification agreed with ground-truth outcome
classification in all 5 cases (2 correct matches, 3 correct mismatches),
including a case requiring fine-grained distinction (red ball vs. red key
-- same color, different object type). This directly addresses the Phase 3
finding that RL policy confidence alone cannot distinguish successful from
confidently-wrong pickups (both ~0.90 mean confidence) -- the LLM verifier
provides an external, reliable signal the policy's own outputs cannot.

### Architecture note
Used AWS Bedrock's newer Mantle endpoint (OpenAI-compatible Chat
Completions API) rather than the traditional bedrock-runtime InvokeModel
API, since the available model (Kimi K2.5) is served via that path. This
demonstrates provider-agnostic LLM integration in the planning layer,
consistent with the PRD's original AWS Bedrock design.