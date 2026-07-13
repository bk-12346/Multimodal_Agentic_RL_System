import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper

MODEL_PATH = "models/ppo_curriculum_n2_finetuned"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"
N_EPISODES = 100


def main():
    env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    model = PPO.load(MODEL_PATH)

    picked_something = 0
    correct = 0

    for seed in range(N_EPISODES):
        obs, info = env.reset(seed=seed)
        target = (env.unwrapped.targetColor, env.unwrapped.targetType)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        carrying = env.unwrapped.carrying
        if carrying:
            picked_something += 1
            if (carrying.color, carrying.type) == target:
                correct += 1

    env.close()

    pickup_rate = picked_something / N_EPISODES
    accuracy_given_pickup = correct / picked_something if picked_something else 0
    overall_success = correct / N_EPISODES

    print(f"Pickup rate (grabbed anything):      {picked_something}/{N_EPISODES} ({100*pickup_rate:.1f}%)")
    print(f"Accuracy given pickup (right object): {correct}/{picked_something} ({100*accuracy_given_pickup:.1f}%)")
    print(f"Overall success rate:                 {correct}/{N_EPISODES} ({100*overall_success:.1f}%)")


if __name__ == "__main__":
    main()

# This separates the two skills cleanly going forward: pickup rate tells us if navigation/interaction is solid, accuracy given pickup tells us if instruction-following (the fusion module) is actually working. 
# That's the number we ultimately care about proving out for the PRD.

##### RESULTS 2: FiLM fusion extractor #####
# Pickup rate (grabbed anything):      99/100 (99.0%)
# Accuracy given pickup (right object): 66/99 (66.7%)
# Overall success rate:                 66/100 (66.0%)

##### RESULTS 1: NLP+cat fusion extractor #####
# Pickup rate (grabbed anything):      67/100 (67.0%)
# Accuracy given pickup (right object): 37/67 (55.2%)
# Overall success rate:                 37/100 (37.0%)