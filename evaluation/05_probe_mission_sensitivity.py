import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper

MODEL_PATH = "models/ppo_curriculum_n2_finetuned"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"


def tokenize(mission: str, vocab: dict, max_len: int) -> np.ndarray:
    words = mission.lower().split()
    ids = [vocab.get(w, vocab["<unk>"]) for w in words]
    ids = ids[:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def main():
    with open(VOCAB_PATH) as f:
        payload = json.load(f)
    vocab, max_len = payload["vocab"], payload["max_len"]

    env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    obs, _ = env.reset(seed=0)
    fixed_image = obs["image"]  # hold the visual scene completely fixed

    model = PPO.load(MODEL_PATH)

    test_missions = [
        "go get a blue key",
        "go get a red key",
        "go get a green ball",
        "go get a purple ball",
        "go get a yellow key",
    ]

    print(f"Fixed image observation, varying ONLY the mission text:\n")
    for mission in test_missions:
        mission_ids = tokenize(mission, vocab, max_len)
        fake_obs = {"image": fixed_image, "mission": mission_ids}

        obs_tensor, _ = model.policy.obs_to_tensor(fake_obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.numpy().flatten()

        top_action = int(np.argmax(probs))
        print(f"mission='{mission:25s}' action_probs={np.round(probs, 3)} top_action={top_action}")


if __name__ == "__main__":
    main()

##### RESULTS 2: FiLM fusion extractor #####
# mission='go get a blue key        ' action_probs=[0.    0.001 0.949 0.049 0.    0.001 0.   ] top_action=2
# mission='go get a red key         ' action_probs=[0.    0.001 0.938 0.059 0.001 0.001 0.   ] top_action=2
# mission='go get a green ball      ' action_probs=[0.    0.001 0.946 0.051 0.001 0.001 0.   ] top_action=2
# mission='go get a purple ball     ' action_probs=[0.    0.001 0.934 0.063 0.001 0.001 0.   ] top_action=2
# mission='go get a yellow key      ' action_probs=[0.    0.001 0.95  0.048 0.    0.001 0.   ] top_action=2

##### RESULTS 1: NLP+cat fusion extractor #####
# Fixed image observation, varying ONLY the mission text:

# mission='go get a blue key        ' action_probs=[0.041 0.038 0.874 0.    0.035 0.01  0.001] top_action=2
# mission='go get a red key         ' action_probs=[0.041 0.042 0.869 0.    0.036 0.01  0.001] top_action=2
# mission='go get a green ball      ' action_probs=[0.041 0.052 0.855 0.    0.04  0.011 0.001] top_action=2
# mission='go get a purple ball     ' action_probs=[0.042 0.062 0.841 0.    0.042 0.011 0.001] top_action=2
# mission='go get a yellow key      ' action_probs=[0.041 0.041 0.87  0.    0.036 0.01  0.001] top_action=2