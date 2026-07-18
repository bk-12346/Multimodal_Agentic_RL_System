import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper
from envs.heldout_filter_wrapper import HeldoutMissionFilterWrapper
from envs.heldout_force_wrapper import HeldoutMissionForceWrapper
from envs.build_vocab_heldout import is_training_mission, HELD_OUT_COLOR, HELD_OUT_COMBO

MODEL_PATH = "models/ppo_heldout_split"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"
N_EPISODES = 100


def is_heldout_mission(color, obj_type):
    return not is_training_mission(color, obj_type)


def run_eval(env, model, n_episodes, label):
    picked_something = 0
    correct = 0

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
            picked_something += 1
            if (carrying.color, carrying.type) == target:
                correct += 1

    pickup_rate = picked_something / n_episodes
    accuracy_given_pickup = correct / picked_something if picked_something else 0
    overall = correct / n_episodes

    print(f"\n--- {label} ---")
    print(f"Pickup rate:              {picked_something}/{n_episodes} ({100*pickup_rate:.1f}%)")
    print(f"Accuracy given pickup:    {correct}/{picked_something} ({100*accuracy_given_pickup:.1f}%)")
    print(f"Overall success:          {correct}/{n_episodes} ({100*overall:.1f}%)")
    return accuracy_given_pickup


def main():
    model = PPO.load(MODEL_PATH)

    # Seen vocab (matches training distribution)
    seen_env = gym.make(ENV_ID)
    seen_env = HeldoutMissionFilterWrapper(seen_env, is_allowed_fn=is_training_mission)
    seen_env = MissionTokenizerWrapper(seen_env, vocab_path=VOCAB_PATH)
    seen_acc = run_eval(seen_env, model, N_EPISODES, "SEEN vocab (in training distribution)")

    # Held-out vocab (never seen during training)
    heldout_env = gym.make(ENV_ID)
    heldout_env = HeldoutMissionForceWrapper(heldout_env, is_heldout_fn=is_heldout_mission)
    heldout_env = MissionTokenizerWrapper(heldout_env, vocab_path=VOCAB_PATH)
    heldout_acc = run_eval(heldout_env, model, N_EPISODES, f"HELD-OUT vocab (color='{HELD_OUT_COLOR}', combo={HELD_OUT_COMBO})")

    gap = seen_acc - heldout_acc
    print(f"\n=== Generalization gap: {100*gap:.1f} percentage points ===")


if __name__ == "__main__":
    main()

##### RESULTS #####

# --- SEEN vocab (in training distribution) ---
# Pickup rate:              96/100 (96.0%)
# Accuracy given pickup:    86/96 (89.6%)
# Overall success:          86/100 (86.0%)

# --- HELD-OUT vocab (color='grey', combo=('purple', 'key')) ---
# Pickup rate:              44/100 (44.0%)
# Accuracy given pickup:    9/44 (20.5%)
# Overall success:          9/100 (9.0%)

# === Generalization gap: 69.1 percentage points ===
