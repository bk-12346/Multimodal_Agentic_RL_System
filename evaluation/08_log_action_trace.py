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
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

def main():
    env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    model = PPO.load(MODEL_PATH)

    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        mission = env.unwrapped.mission
        actions_taken = []
        for _ in range(40):
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(ACTION_NAMES[int(action)])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        print(f"seed={seed} mission='{mission}'\n  actions: {actions_taken}\n")

if __name__ == "__main__":
    main()

##### RESULTS #####
# seed=0 mission='go get a blue key'
#   actions: ['forward', 'right', 'pickup']

# seed=1 mission='you must fetch a purple key'
#   actions: ['right', 'right', 'forward', 'pickup']

# seed=2 mission='go fetch a green ball'
#   actions: ['pickup']

# seed=3 mission='fetch a grey ball'
#   actions: ['right', 'forward', 'right', 'forward', 'pickup']

# seed=4 mission='go get a grey key'
#   actions: ['right', 'right', 'pickup']