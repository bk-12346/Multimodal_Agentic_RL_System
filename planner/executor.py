import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper

VOCAB_PATH = "configs/mission_vocab.json"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
MAX_STEPS_PER_SUBGOAL = 100
MAX_RETRIES_PER_SUBGOAL = 2

LOW_CONFIDENCE_THRESHOLD = 0.3   # max action-prob below this counts as "uncertain"
CONSECUTIVE_LOW_CONF_ABORT = 8   # abort after this many consecutive uncertain steps


def predict_with_confidence(model, obs):
    """
    Runs the policy and returns both the chosen action AND a confidence
    score = probability the policy assigned to that action. A peaked
    distribution (e.g. 0.95 on one action) means the policy is confident;
    a flat distribution (e.g. ~0.14 across 7 actions) means it's unsure --
    this is the same signal we inspected manually in the mission-sensitivity
    probes earlier, now used programmatically for failure detection.
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.numpy().flatten()
    action = int(np.argmax(probs))
    confidence = float(probs[action])
    return action, confidence


class SubgoalExecutor:
    """
    Executes a single subgoal, tracking per-step confidence. Aborts early
    if the policy shows sustained low confidence (likely confused/lost),
    saving compute versus running the full step budget on a doomed attempt.
    Classifies outcomes as SUCCESS, WRONG_OBJECT, TIMEOUT, or ABORTED_LOW_CONFIDENCE.
    """

    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)

    def _run_once(self, color: str, obj_type: str, seed: int):
        raw_env = gym.make(ENV_ID)
        env = MissionTokenizerWrapper(raw_env, vocab_path=VOCAB_PATH)

        for _ in range(50):
            obs, info = env.reset(seed=seed)
            if (env.unwrapped.targetColor, env.unwrapped.targetType) == (color, obj_type):
                break
            seed += 1000

        step_count = 0
        done = False
        confidences = []
        consecutive_low_conf = 0
        aborted = False

        while not done and step_count < MAX_STEPS_PER_SUBGOAL:
            action, confidence = predict_with_confidence(self.model, obs)
            confidences.append(confidence)

            if confidence < LOW_CONFIDENCE_THRESHOLD:
                consecutive_low_conf += 1
                if consecutive_low_conf >= CONSECUTIVE_LOW_CONF_ABORT:
                    aborted = True
                    break
            else:
                consecutive_low_conf = 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        carrying = env.unwrapped.carrying
        env.close()

        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        if aborted:
            outcome = "ABORTED_LOW_CONFIDENCE"
        elif carrying and (carrying.color, carrying.type) == (color, obj_type):
            outcome = "SUCCESS"
        elif carrying:
            outcome = "WRONG_OBJECT"
        else:
            outcome = "TIMEOUT"

        return outcome, step_count, mean_conf

    def execute_subgoal(self, subgoal: dict, base_seed: int) -> dict:
        color, obj_type = subgoal["color"], subgoal["type"]
        attempts = []

        for attempt in range(1, MAX_RETRIES_PER_SUBGOAL + 2):
            outcome, steps, mean_conf = self._run_once(color, obj_type, seed=base_seed + attempt * 7919)
            attempts.append({"attempt": attempt, "outcome": outcome, "steps": steps, "mean_confidence": round(mean_conf, 3)})
            if outcome == "SUCCESS":
                break

        final_outcome = attempts[-1]["outcome"]
        return {
            "subgoal": subgoal,
            "attempts": attempts,
            "final_outcome": final_outcome,
            "resolved": final_outcome == "SUCCESS",
        }


def execute_plan(subgoals: list[dict], model_path: str, base_seed: int = 0) -> list[dict]:
    executor = SubgoalExecutor(model_path)
    results = []
    for i, subgoal in enumerate(subgoals):
        result = executor.execute_subgoal(subgoal, base_seed=base_seed + i * 100)
        results.append(result)
        status = "✓" if result["resolved"] else "✗ (gave up after retries)"
        last = result["attempts"][-1]
        print(f"Subgoal {i+1}/{len(subgoals)}: '{subgoal['mission']}' -> {result['final_outcome']} "
              f"(conf={last['mean_confidence']:.2f}, {len(result['attempts'])} attempt(s)) {status}")
    return results

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import gymnasium as gym
# import minigrid
# from stable_baselines3 import PPO

# from envs.wrapper import MissionTokenizerWrapper

# VOCAB_PATH = "configs/mission_vocab.json"
# ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
# MAX_STEPS_PER_SUBGOAL = 100
# MAX_RETRIES_PER_SUBGOAL = 2


# class SubgoalExecutor:
#     """
#     Executes a single subgoal by resetting the env with a forced target,
#     running the trained RL policy, and classifying the outcome as SUCCESS,
#     WRONG_OBJECT, or TIMEOUT (no pickup). Retries failed subgoals up to a
#     limit before marking them as a permanent failure and moving on --
#     this is the 'failure detection + retry/fallback logic' the PRD calls for.
#     """

#     def __init__(self, model_path: str):
#         self.model = PPO.load(model_path)

#     def _run_once(self, color: str, obj_type: str, seed: int):
#         raw_env = gym.make(ENV_ID)
#         env = MissionTokenizerWrapper(raw_env, vocab_path=VOCAB_PATH)

#         # Force this env instance to have the requested target by resampling
#         # until it matches -- reuses the same trick as our heldout wrapper.
#         for _ in range(50):
#             obs, info = env.reset(seed=seed)
#             if (env.unwrapped.targetColor, env.unwrapped.targetType) == (color, obj_type):
#                 break
#             seed += 1000  # jump to a different seed to resample

#         step_count = 0
#         done = False
#         while not done and step_count < MAX_STEPS_PER_SUBGOAL:
#             action, _ = self.model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             step_count += 1

#         carrying = env.unwrapped.carrying
#         env.close()

#         if carrying and (carrying.color, carrying.type) == (color, obj_type):
#             return "SUCCESS", step_count
#         elif carrying:
#             return "WRONG_OBJECT", step_count
#         else:
#             return "TIMEOUT", step_count

#     def execute_subgoal(self, subgoal: dict, base_seed: int) -> dict:
#         color, obj_type = subgoal["color"], subgoal["type"]
#         attempts = []

#         for attempt in range(1, MAX_RETRIES_PER_SUBGOAL + 2):  # +2: initial try + retries
#             outcome, steps = self._run_once(color, obj_type, seed=base_seed + attempt * 7919)
#             attempts.append({"attempt": attempt, "outcome": outcome, "steps": steps})
#             if outcome == "SUCCESS":
#                 break

#         final_outcome = attempts[-1]["outcome"]
#         return {
#             "subgoal": subgoal,
#             "attempts": attempts,
#             "final_outcome": final_outcome,
#             "resolved": final_outcome == "SUCCESS",
#         }


# def execute_plan(subgoals: list[dict], model_path: str, base_seed: int = 0) -> list[dict]:
#     executor = SubgoalExecutor(model_path)
#     results = []
#     for i, subgoal in enumerate(subgoals):
#         result = executor.execute_subgoal(subgoal, base_seed=base_seed + i * 100)
#         results.append(result)
#         status = "✓" if result["resolved"] else "✗ (gave up after retries)"
#         print(f"Subgoal {i+1}/{len(subgoals)}: '{subgoal['mission']}' -> {result['final_outcome']} {status}")
#     return results