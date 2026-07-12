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
N_EPISODES = 50


def main():
    env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    model = PPO.load(MODEL_PATH)

    for seed in range(N_EPISODES):
        obs, info = env.reset(seed=seed)
        target_color = env.unwrapped.targetColor
        target_type = env.unwrapped.targetType
        mission = env.unwrapped.mission

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        carrying = env.unwrapped.carrying
        picked = f"{carrying.color} {carrying.type}" if carrying else "NOTHING"
        target = f"{target_color} {target_type}"
        match = "✓" if picked == target else "✗"

        print(f"seed={seed:2d} mission='{mission:35s}' target={target:12s} picked={picked:12s} {match}")

    env.close()


if __name__ == "__main__":
    main()

##### RESULTS #####
# seed= 0 mission='go get a blue key                  ' target=blue key     picked=NOTHING      ✗
# seed= 1 mission='you must fetch a purple key        ' target=purple key   picked=NOTHING      ✗
# seed= 2 mission='go fetch a green ball              ' target=green ball   picked=green ball   ✓
# seed= 3 mission='fetch a grey ball                  ' target=grey ball    picked=NOTHING      ✗
# seed= 4 mission='go get a grey key                  ' target=grey key     picked=NOTHING      ✗
# seed= 5 mission='go get a red ball                  ' target=red ball     picked=NOTHING      ✗
# seed= 6 mission='go fetch a purple key              ' target=purple key   picked=NOTHING      ✗
# seed= 7 mission='fetch a purple ball                ' target=purple ball  picked=NOTHING      ✗
# seed= 8 mission='fetch a green ball                 ' target=green ball   picked=NOTHING      ✗
# seed= 9 mission='you must fetch a yellow key        ' target=yellow key   picked=NOTHING      ✗
# seed=10 mission='you must fetch a red ball          ' target=red ball     picked=NOTHING      ✗
# seed=11 mission='fetch a purple ball                ' target=purple ball  picked=purple ball  ✓
# seed=12 mission='go get a green ball                ' target=green ball   picked=NOTHING      ✗
# seed=13 mission='go get a yellow ball               ' target=yellow ball  picked=NOTHING      ✗
# seed=14 mission='you must fetch a red key           ' target=red key      picked=red key      ✓
# seed=15 mission='go get a red ball                  ' target=red ball     picked=red ball     ✓
# seed=16 mission='you must fetch a purple key        ' target=purple key   picked=NOTHING      ✗
# seed=17 mission='get a grey ball                    ' target=grey ball    picked=NOTHING      ✗
# seed=18 mission='go fetch a green ball              ' target=green ball   picked=grey ball    ✗
# seed=19 mission='fetch a grey ball                  ' target=grey ball    picked=grey ball    ✓
# seed=20 mission='fetch a green ball                 ' target=green ball   picked=NOTHING      ✗
# seed=21 mission='you must fetch a red key           ' target=red key      picked=NOTHING      ✗
# seed=22 mission='go get a grey key                  ' target=grey key     picked=grey key     ✓
# seed=23 mission='go fetch a blue key                ' target=blue key     picked=blue key     ✓
# seed=24 mission='go get a purple ball               ' target=purple ball  picked=NOTHING      ✗
# seed=25 mission='go fetch a grey ball               ' target=grey ball    picked=NOTHING      ✗
# seed=26 mission='go fetch a blue key                ' target=blue key     picked=grey ball    ✗
# seed=27 mission='go get a red key                   ' target=red key      picked=NOTHING      ✗
# seed=28 mission='you must fetch a yellow ball       ' target=yellow ball  picked=red ball     ✗
# seed=29 mission='go fetch a blue ball               ' target=blue ball    picked=purple key   ✗
# seed=30 mission='fetch a green key                  ' target=green key    picked=green key    ✓
# seed=31 mission='you must fetch a grey key          ' target=grey key     picked=NOTHING      ✗
# seed=32 mission='fetch a green key                  ' target=green key    picked=NOTHING      ✗
# seed=33 mission='you must fetch a grey ball         ' target=grey ball    picked=NOTHING      ✗
# seed=34 mission='get a yellow key                   ' target=yellow key   picked=NOTHING      ✗
# seed=35 mission='go get a yellow key                ' target=yellow key   picked=NOTHING      ✗
# seed=36 mission='go fetch a green key               ' target=green key    picked=green key    ✓
# seed=37 mission='fetch a blue key                   ' target=blue key     picked=NOTHING      ✗
# seed=38 mission='go fetch a grey key                ' target=grey key     picked=NOTHING      ✗
# seed=39 mission='you must fetch a blue ball         ' target=blue ball    picked=NOTHING      ✗
# seed=40 mission='go fetch a yellow key              ' target=yellow key   picked=yellow key   ✓
# seed=41 mission='fetch a red ball                   ' target=red ball     picked=NOTHING      ✗
# seed=42 mission='go fetch a yellow key              ' target=yellow key   picked=NOTHING      ✗
# seed=43 mission='get a purple ball                  ' target=purple ball  picked=NOTHING      ✗
# seed=44 mission='go get a yellow key                ' target=yellow key   picked=NOTHING      ✗
# seed=45 mission='go fetch a red ball                ' target=red ball     picked=NOTHING      ✗
# seed=46 mission='go fetch a purple key              ' target=purple key   picked=NOTHING      ✗
# seed=47 mission='you must fetch a grey key          ' target=grey key     picked=NOTHING      ✗
# seed=48 mission='you must fetch a grey key          ' target=grey key     picked=purple ball  ✗
# seed=49 mission='go get a grey key                  ' target=grey key     picked=NOTHING      ✗
