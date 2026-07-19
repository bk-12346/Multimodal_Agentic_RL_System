import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

from envs.pretrained_text_wrapper import PretrainedTextWrapper
from envs.heldout_filter_wrapper import HeldoutMissionFilterWrapper
from envs.heldout_force_wrapper import HeldoutMissionForceWrapper
from envs.build_vocab_heldout import is_training_mission, HELD_OUT_COLOR, HELD_OUT_COMBO

# MODEL_PATH = "models/ppo_pretrained_heldout_n2"
MODEL_PATH = "models/ppo_pretrained_heldout_n2_extended"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
N_EPISODES = 100


def run_eval(env, model, n_episodes, label):
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

    pickup_rate = picked / n_episodes
    acc = correct / picked if picked else 0
    print(f"\n--- {label} ---")
    print(f"Pickup rate: {picked}/{n_episodes} ({100*pickup_rate:.1f}%)")
    print(f"Accuracy given pickup: {correct}/{picked} ({100*acc:.1f}%)" if picked else "Accuracy given pickup: N/A")
    print(f"Overall success: {correct}/{n_episodes} ({100*correct/n_episodes:.1f}%)")
    return acc


def main():
    model = PPO.load(MODEL_PATH)

    seen_env = gym.make(ENV_ID)
    seen_env = HeldoutMissionFilterWrapper(seen_env, is_allowed_fn=is_training_mission)
    seen_env = PretrainedTextWrapper(seen_env)
    run_eval(seen_env, model, N_EPISODES, "SEEN vocab")

    combo_env = gym.make(ENV_ID)
    combo_env = HeldoutMissionForceWrapper(combo_env, is_heldout_fn=lambda c, t: (c, t) == HELD_OUT_COMBO)
    combo_env = PretrainedTextWrapper(combo_env)
    run_eval(combo_env, model, N_EPISODES, f"NOVEL COMBO ({HELD_OUT_COMBO[0]} {HELD_OUT_COMBO[1]})")

    color_env = gym.make(ENV_ID)
    color_env = HeldoutMissionForceWrapper(color_env, is_heldout_fn=lambda c, t: c == HELD_OUT_COLOR)
    color_env = PretrainedTextWrapper(color_env)
    run_eval(color_env, model, N_EPISODES, f"NOVEL COLOR ({HELD_OUT_COLOR}) — the one that was 0% before")


if __name__ == "__main__":
    main()

##### RESULTS 2: With extended training #####
# --- SEEN vocab ---
# Pickup rate: 90/100 (90.0%)
# Accuracy given pickup: 84/90 (93.3%)
# Overall success: 84/100 (84.0%)

# --- NOVEL COMBO (purple key) ---
# Pickup rate: 90/100 (90.0%)
# Accuracy given pickup: 83/90 (92.2%)
# Overall success: 83/100 (83.0%)

# --- NOVEL COLOR (grey) — the one that was 0% before ---
# Pickup rate: 28/100 (28.0%)
# Accuracy given pickup: 0/28 (0.0%)
# Overall success: 0/100 (0.0%)

##### RESULTS 1 #####
# --- SEEN vocab ---
# Pickup rate: 82/100 (82.0%)
# Accuracy given pickup: 58/82 (70.7%)
# Overall success: 58/100 (58.0%)

# --- NOVEL COMBO (purple key) ---
# Pickup rate: 70/100 (70.0%)
# Accuracy given pickup: 36/70 (51.4%)
# Overall success: 36/100 (36.0%)

# --- NOVEL COLOR (grey) — the one that was 0% before ---
# Pickup rate: 83/100 (83.0%)
# Accuracy given pickup: 38/83 (45.8%)
# Overall success: 38/100 (38.0%)