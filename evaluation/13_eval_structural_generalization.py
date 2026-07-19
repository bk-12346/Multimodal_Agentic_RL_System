import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

# from envs.wrapper import MissionTokenizerWrapper
from envs.pretrained_text_wrapper import PretrainedTextWrapper

# MODEL_PATH = "models/ppo_curriculum_n2_finetuned"   # original FiLM, scratch embedding, full vocab
MODEL_PATH = "models/ppo_pretrained_checkpoint_search_final"  # MiniLM, most-trained checkpoint (93% seen accuracy)
VOCAB_PATH = "configs/mission_vocab.json"
N_EPISODES = 100

TEST_ENVS = {
    "Train dist: Fetch-5x5-N2": "MiniGrid-Fetch-5x5-N2-v0",
    "Structural: Fetch-6x6-N2 (bigger grid)": "MiniGrid-Fetch-6x6-N2-v0",
    "Structural: Fetch-8x8-N3 (bigger grid + more distractors)": "MiniGrid-Fetch-8x8-N3-v0",
}


def run_eval(env_id, model, n_episodes):
    env = gym.make(env_id)
    # env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    env = PretrainedTextWrapper(env)  # changed from MissionTokenizerWrapper

    picked, correct = 0, 0
    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        target = (env.unwrapped.targetColor, env.unwrapped.targetType)
        done = False
        step_count = 0
        step_count = 0
        max_allowed_steps = env.unwrapped.max_steps  # use the env's real budget, not an arbitrary cap
        while not done and step_count < max_allowed_steps:
        # while not done and step_count < 200:  # safety cap for larger grids (higher max_steps)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        carrying = env.unwrapped.carrying
        if carrying:
            picked += 1
            correct += (carrying.color, carrying.type) == target

    env.close()
    pickup_rate = picked / n_episodes
    acc = correct / picked if picked else 0
    return pickup_rate, acc, correct / n_episodes


def main():
    model = PPO.load(MODEL_PATH)

    print(f"Model: {MODEL_PATH}\n")
    for label, env_id in TEST_ENVS.items():
        pickup_rate, acc, overall = run_eval(env_id, model, N_EPISODES)
        print(f"--- {label} ---")
        print(f"Pickup rate: {100*pickup_rate:.1f}%")
        print(f"Accuracy given pickup: {100*acc:.1f}%")
        print(f"Overall success: {100*overall:.1f}%\n")


if __name__ == "__main__":
    main()

##### RESULTS 3: MiniLM #####
# --- Train dist: Fetch-5x5-N2 ---
# Pickup rate: 85.0%
# Accuracy given pickup: 77.6%
# Overall success: 66.0%

# --- Structural: Fetch-6x6-N2 (bigger grid) ---
# Pickup rate: 59.0%
# Accuracy given pickup: 78.0%
# Overall success: 46.0%

# --- Structural: Fetch-8x8-N3 (bigger grid + more distractors) ---
# Pickup rate: 38.0%
# Accuracy given pickup: 34.2%
# Overall success: 13.0%

##### RESULTS 2: using the env's real budget, not an arbitrary cap #####
# --- Train dist: Fetch-5x5-N2 ---
# Pickup rate: 99.0%
# Accuracy given pickup: 66.7%
# Overall success: 66.0%

# --- Structural: Fetch-6x6-N2 (bigger grid) ---
# Pickup rate: 84.0%
# Accuracy given pickup: 67.9%
# Overall success: 57.0%

# --- Structural: Fetch-8x8-N3 (bigger grid + more distractors) ---
# Pickup rate: 59.0%
# Accuracy given pickup: 52.5%
# Overall success: 31.0%

##### RESULTS 1: step cap=200 #####
# --- Train dist: Fetch-5x5-N2 ---
# Pickup rate: 99.0%
# Accuracy given pickup: 66.7%
# Overall success: 66.0%

# --- Structural: Fetch-6x6-N2 (bigger grid) ---
# Pickup rate: 84.0%
# Accuracy given pickup: 67.9%
# Overall success: 57.0%

# --- Structural: Fetch-8x8-N3 (bigger grid + more distractors) ---
# Pickup rate: 59.0%
# Accuracy given pickup: 52.5%
# Overall success: 31.0%
