import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper

MODEL_PATH = "models/ppo_instruction_fetch"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"
N_EPISODES = 100


def main():
    env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)

    model = PPO.load(MODEL_PATH)

    successes = 0
    rewards = []

    for seed in range(N_EPISODES):
        obs, info = env.reset(seed=seed)
        mission = env.unwrapped.mission
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        success = ep_reward > 0
        successes += success
        if seed < 10:  # print first 10 for a spot-check
            print(f"seed={seed} mission='{mission}' reward={ep_reward:.3f} success={success}")

    print(f"\nSuccess rate: {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.1f}%)")
    print(f"Mean reward: {sum(rewards)/len(rewards):.3f}")


if __name__ == "__main__":
    main()

##### RESULTS 1 #####
# seed=0 mission='go get a blue key' reward=0.000 success=False
# seed=1 mission='you must fetch a purple key' reward=0.000 success=False
# seed=2 mission='go fetch a green ball' reward=0.993 success=True
# seed=3 mission='fetch a grey ball' reward=0.000 success=False
# seed=4 mission='go get a grey key' reward=0.000 success=False
# seed=5 mission='go get a red ball' reward=0.000 success=False
# seed=6 mission='go fetch a purple key' reward=0.000 success=False
# seed=7 mission='fetch a purple ball' reward=0.000 success=False
# seed=8 mission='fetch a green ball' reward=0.000 success=False
# seed=9 mission='you must fetch a yellow key' reward=0.000 success=False

# Success rate: 14/100 (14.0%)
# Mean reward: 0.139