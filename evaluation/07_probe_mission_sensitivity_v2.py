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

# MiniGrid OBJECT_TO_IDX: key=5, ball=6
OBJECT_IDS = {5, 6}


def tokenize(mission, vocab, max_len):
    words = mission.lower().split()
    ids = [vocab.get(w, vocab["<unk>"]) for w in words]
    ids = ids[:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def find_frame_with_object(env, seed):
    """Step randomly until an object (key/ball) is visible in the partial view."""
    obs, _ = env.reset(seed=seed)
    for _ in range(30):
        if np.isin(obs["image"][:, :, 0], list(OBJECT_IDS)).any():
            return obs
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)
    return None  # didn't find one, caller should try another seed


def main():
    with open(VOCAB_PATH) as f:
        payload = json.load(f)
    vocab, max_len = payload["vocab"], payload["max_len"]

    raw_env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(raw_env, vocab_path=VOCAB_PATH)

    fixed_image = None
    for seed in range(20):
        obs = find_frame_with_object(env, seed)
        if obs is not None:
            fixed_image = obs["image"]
            print(f"Using frame from seed={seed} (object visible in view)\n")
            break

    if fixed_image is None:
        print("Couldn't find a frame with a visible object in 20 tries — env/seed issue.")
        return

    model = PPO.load(MODEL_PATH)

    test_missions = [
        "go get a blue key",
        "go get a red key",
        "go get a green ball",
        "go get a purple ball",
        "go get a yellow key",
    ]

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

##### RESULTS #####
# Using frame from seed=0 (object visible in view)

# mission='go get a blue key        ' action_probs=[0.    0.001 0.949 0.049 0.    0.001 0.   ] top_action=2
# mission='go get a red key         ' action_probs=[0.    0.001 0.938 0.059 0.001 0.001 0.   ] top_action=2
# mission='go get a green ball      ' action_probs=[0.    0.001 0.946 0.051 0.001 0.001 0.   ] top_action=2
# mission='go get a purple ball     ' action_probs=[0.    0.001 0.934 0.063 0.001 0.001 0.   ] top_action=2
# mission='go get a yellow key      ' action_probs=[0.    0.001 0.95  0.048 0.    0.001 0.   ] top_action=2
