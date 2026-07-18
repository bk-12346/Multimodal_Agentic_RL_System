import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
import imageio
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper
from envs.heldout_force_wrapper import HeldoutMissionForceWrapper
from envs.build_vocab_heldout import HELD_OUT_COLOR

MODEL_PATH = "models/ppo_heldout_split"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"
OUTPUT_DIR = Path("evaluation/failure_renders")
N_EPISODES_TO_RENDER = 5
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_env = gym.make(ENV_ID, render_mode="rgb_array")
    filtered_env = HeldoutMissionForceWrapper(raw_env, is_heldout_fn=lambda c, t: c == HELD_OUT_COLOR)
    env = MissionTokenizerWrapper(filtered_env, vocab_path=VOCAB_PATH)
    model = PPO.load(MODEL_PATH)

    for seed in range(N_EPISODES_TO_RENDER):
        obs, info = env.reset(seed=seed)
        mission = env.unwrapped.mission
        target = (env.unwrapped.targetColor, env.unwrapped.targetType)

        frames = [env.render()]
        actions_taken = []

        done = False
        step_count = 0
        while not done and step_count < 40:
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(ACTION_NAMES[int(action)])
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            done = terminated or truncated
            step_count += 1

        carrying = env.unwrapped.carrying
        picked = f"{carrying.color} {carrying.type}" if carrying else "NOTHING"

        gif_path = OUTPUT_DIR / f"seed{seed}_target-{target[0]}-{target[1]}_picked-{picked.replace(' ', '-')}.gif"
        imageio.mimsave(gif_path, frames, duration=300, loop=0)

        print(f"seed={seed} mission='{mission}' target={target} picked={picked}")
        print(f"  actions: {actions_taken}")
        print(f"  saved: {gif_path}\n")

    env.close()


if __name__ == "__main__":
    main()

##### RESULTS #####

# saved to evaluation/failure_renders

# seed=0 mission='go get a grey key' target=('grey', 'key') picked=NOTHING
#   actions: ['forward', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
#   saved: evaluation\failure_renders\seed0_target-grey-key_picked-NOTHING.gif

# seed=1 mission='you must fetch a grey ball' target=('grey', 'ball') picked=NOTHING
#   actions: ['right', 'right', 'forward', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
#   saved: evaluation\failure_renders\seed1_target-grey-ball_picked-NOTHING.gif

# seed=2 mission='go get a grey ball' target=('grey', 'ball') picked=NOTHING
#   actions: ['right', 'right', 'forward', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
#   saved: evaluation\failure_renders\seed2_target-grey-ball_picked-NOTHING.gif

# seed=3 mission='fetch a grey ball' target=('grey', 'ball') picked=blue ball
#   actions: ['right', 'right', 'forward', 'right', 'pickup']
#   saved: evaluation\failure_renders\seed3_target-grey-ball_picked-blue-ball.gif

# seed=4 mission='go get a grey key' target=('grey', 'key') picked=yellow ball
#   actions: ['right', 'right', 'right', 'pickup']
#   saved: evaluation\failure_renders\seed4_target-grey-key_picked-yellow-ball.gif