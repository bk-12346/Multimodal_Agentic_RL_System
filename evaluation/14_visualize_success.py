import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
import imageio
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper

MODEL_PATH = "models/ppo_curriculum_n2_finetuned"
VOCAB_PATH = "configs/mission_vocab.json"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
OUTPUT_DIR = Path("evaluation/demo_renders")
N_CANDIDATES_TO_TRY = 30   # scan several seeds, keep only clean successes
MAX_EPISODES_TO_SAVE = 3
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_env = gym.make(ENV_ID, render_mode="rgb_array")
    env = MissionTokenizerWrapper(raw_env, vocab_path=VOCAB_PATH)
    model = PPO.load(MODEL_PATH)

    saved = 0
    for seed in range(N_CANDIDATES_TO_TRY):
        if saved >= MAX_EPISODES_TO_SAVE:
            break

        obs, info = env.reset(seed=seed)
        mission = env.unwrapped.mission
        target = (env.unwrapped.targetColor, env.unwrapped.targetType)

        frames = [env.render()]
        actions_taken = []
        done = False
        step_count = 0

        while not done and step_count < 30:  # keep GIFs short and clean for a README demo
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(ACTION_NAMES[int(action)])
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            done = terminated or truncated
            step_count += 1

        carrying = env.unwrapped.carrying
        success = carrying and (carrying.color, carrying.type) == target

        if success:
            gif_path = OUTPUT_DIR / f"demo_success_{saved}_{target[0]}-{target[1]}.gif"
            imageio.mimsave(gif_path, frames, duration=400, loop=0)
            print(f"seed={seed} mission='{mission}' -> SUCCESS in {step_count} steps")
            print(f"  actions: {actions_taken}")
            print(f"  saved: {gif_path}\n")
            saved += 1

    env.close()
    print(f"Saved {saved} success demo GIF(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()