import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper
from envs.heldout_force_wrapper import HeldoutMissionForceWrapper
from envs.build_vocab_heldout import HELD_OUT_COLOR, HELD_OUT_COMBO

MODEL_PATH = "models/ppo_heldout_split"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"
N_EPISODES = 100


def run_eval(is_target_fn, label, n_episodes=N_EPISODES):
    env = gym.make(ENV_ID)
    env = HeldoutMissionForceWrapper(env, is_heldout_fn=is_target_fn)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    model = PPO.load(MODEL_PATH)

    picked, correct = 0, 0
    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        target = (env.unwrapped.targetColor, env.unwrapped.targetType)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        carrying = env.unwrapped.carrying
        if carrying:
            picked += 1
            correct += (carrying.color, carrying.type) == target

    print(f"\n--- {label} ---")
    print(f"Pickup rate: {picked}/{n_episodes} ({100*picked/n_episodes:.1f}%)")
    if picked:
        print(f"Accuracy given pickup: {correct}/{picked} ({100*correct/picked:.1f}%)")
    print(f"Overall success: {correct}/{n_episodes} ({100*correct/n_episodes:.1f}%)")


if __name__ == "__main__":
    run_eval(lambda c, t: c == HELD_OUT_COLOR, f"Novel COLOR only (grey, any object)")
    run_eval(lambda c, t: (c, t) == HELD_OUT_COMBO, f"Novel COMBO only ({HELD_OUT_COMBO[0]} {HELD_OUT_COMBO[1]}, both parts individually trained)")

##### RESULTS #####

# --- Novel COLOR only (grey, any object) ---
# Pickup rate: 43/100 (43.0%)
# Accuracy given pickup: 0/43 (0.0%)
# Overall success: 0/100 (0.0%)

# --- Novel COMBO only (purple key, both parts individually trained) ---
# Pickup rate: 74/100 (74.0%)
# Accuracy given pickup: 51/74 (68.9%)
# Overall success: 51/100 (51.0%)