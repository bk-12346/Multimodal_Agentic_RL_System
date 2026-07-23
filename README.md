# Multimodal Agentic RL System — Instruction-Conditioned Fetch Agent with Agentic Planning & AWS Deployment

An end-to-end research system building a multimodal, instruction-conditioned reinforcement learning agent, diagnosing and fixing a real multimodal fusion failure, extending it with LLM-based agentic planning, and deploying it live on AWS.

## Key Results

| Configuration | Instruction-following accuracy | Notes |
|---|---|---|
| Concatenation fusion (baseline) | ~55% | Near chance — policy learned to largely ignore the mission |
| **FiLM-conditioned fusion** | **67-90%** | Text modulates vision features directly, forcing the policy to route through language |
| Novel color (never a training target) | 0% | Documented failure mode — see Findings |
| Novel color+object combo (unseen pairing, known parts) | 69-92% | Compositional generalization holds even when the exact pairing is unseen |

**Finding:** the dominant fusion architecture (concatenation + MLP) let the policy learn to ignore a clearly-informative language signal entirely — confirmed via a controlled probe holding the visual input fixed while varying only the mission text, which showed near-identical action distributions regardless of instruction. Switching to FiLM (feature-wise linear modulation, using the text embedding to scale/shift vision features) fixed this, but introduced its own generalization limits, documented in detail in `evaluation/`.

## Why This Project

This project treats multimodal RL as a real diagnostic problem, not just a training run:

- **Mechanistic failure analysis** — when instruction-following was near chance, the fix wasn't "train longer," it was probing the policy's action distribution directly to confirm the language pathway was structurally dead, then redesigning the fusion architecture
- **Honest generalization reporting** — held-out vocabulary and structural (grid size, distractor count) generalization tested separately, with failure modes documented at the mechanism level (e.g. the agent gets stuck in a stable oscillation loop near unfamiliar targets, rather than failing randomly)
- **A negative result taken seriously** — a follow-up using pretrained text embeddings partially fixed the worst failure mode, but further training reversed the improvement; this instability is reported rather than hidden behind a single cherry-picked checkpoint
- **Agentic reliability, not just an LLM wrapper** — the LLM planner's role was chosen specifically to cover a gap identified in testing: the RL policy's own confidence scores could not distinguish correct pickups from confidently-wrong ones, so the LLM verifies outcomes post-hoc instead
- **Real deployment** — a live, containerized API on AWS, not just a notebook

## Architecture

```
Instruction → LLM/rule-based planner → subgoal(s)
↓
MiniGrid observation (image + mission) → FiLM-conditioned PPO agent → action
↓
Outcome → LLM verification → retry / next subgoal
↓
Docker (ECR) → EC2 (FastAPI, IAM instance role) → CloudWatch
```

Model weights and metrics are stored in S3; the deployed container pulls the trained model from S3 at startup rather than baking it into the image.

## Methodology Highlights

- **Environment**: MiniGrid-Fetch (partially-observable, 7×7 egocentric view), varying grid size (5×5 to 8×8) and distractor count (1-3 objects)
- **RL algorithm**: PPO (Stable-Baselines3), custom CNN sized for MiniGrid's small observation space (SB3's default NatureCNN kernel is too large for a 7×7 input)
- **Fusion architectures compared**: naive concatenation + MLP, FiLM (learned text embedding), FiLM (frozen MiniLM sentence embedding)
- **Curriculum learning**: single-object navigation (N1) mastered first, then fine-tuned on multi-object discrimination (N2) — isolates "can it navigate" from "can it discriminate by language"
- **Generalization testing**: held-out color and held-out color+object combination (vocabulary axis), larger grids and more distractors (structural axis), evaluated independently to attribute failures correctly
- **Agentic layer**: rule-based subgoal parser as baseline, replaced with an LLM planner (AWS Bedrock, OpenAI-compatible endpoint) for flexible natural-language parsing and post-hoc outcome verification
- **Failure-case analysis**: rendered GIF playback of failing episodes, not just success-rate numbers — revealed the agent approaches unfamiliar targets correctly but locks into a turning loop right at the interaction point, rather than failing to navigate at all

## Tech Stack

Python · PyTorch · Stable-Baselines3 · Gymnasium/MiniGrid · FastAPI · Docker · AWS (S3, ECR, EC2, IAM, Bedrock, SageMaker, CloudWatch) · boto3 · uv

## Running It

**Setup:**
```bash
uv sync
```

**Train the base agent (curriculum: navigation, then instruction-following):**
```bash
uv run training/04_train_ppo_n1.py
uv run training/05_finetune_ppo_n2.py
```

**Evaluate (decomposed pickup-rate / accuracy metrics):**
```bash
uv run evaluation/04_eval_decomposed.py
```

**Run the agentic planner (rule-based or LLM, with retry/verification):**
```bash
uv run training/11_run_plan.py
```

**Run the API locally:**
```bash
uv run uvicorn api.main:app --reload --port 8000
API + interactive docs: `http://localhost:8000/docs
```

**Build and deploy via Docker:**
```bash
docker build -t multiagent-rl-api .
docker push <your-ecr-repo>/multiagent-rl-api:latest
```

## Project Structure

```
multimodal-agentic-rl-system/
├── envs/                  # MiniGrid wrappers (mission tokenization, held-out vocab splits)
├── models/                # Fusion architectures (concat baseline, FiLM, FiLM + pretrained embeddings)
├── training/              # Training scripts, S3 logging utilities, SageMaker launcher
├── evaluation/            # Eval scripts + phase-by-phase findings write-ups
├── planner/               # Rule-based and LLM-based (Bedrock) planning/verification
├── api/                   # FastAPI inference service
├── sagemaker_training/    # SageMaker script-mode training job
├── configs/               # Vocab and environment configs
├── Dockerfile
└── pyproject.toml
```

## Key Findings

- **A working architecture can still fail silently** — the concat+MLP fusion trained without errors and produced plausible-looking success rates, but a targeted probe (fixing the image, varying only the mission) revealed the language pathway wasn't being used at all
- **FiLM conditioning fixed the core failure but has its own generalization ceiling** — compositional generalization to unseen attribute *combinations* works well; generalization to attributes never seen in the target role does not, and behaves as a stable failure (a lockup loop), not random noise
- **Pretrained embeddings are not a complete fix** — they partially rescued the worst failure mode early in training, but continued optimization reversed the improvement, revealing a training-duration-dependent trade-off between in-distribution sharpness and out-of-distribution robustness
- **Confidence scores from the RL policy are not a reliable failure signal** — the policy was equally confident (~0.90+) whether it was right or wrong, motivating an external (LLM-based) verification step instead

Full write-ups: `evaluation/phase2_findings.md` (generalization), `evaluation/phase3_findings.md` (agentic layer), `evaluation/phase4_findings.md` (deployment decisions)

## Future Work

- Multi-agent extension (cooperative MiniGrid variants) — the original motivation for this project's design choices
- Checkpoint selection based on held-out generalization accuracy rather than training reward, to find a stable point on the robustness/specialization trade-off
- CI/CD pipeline for the Docker build and deployment
- SageMaker-based training at scale (currently used for a lightweight demo job; local/EC2 training was sufficient at this project's scale)