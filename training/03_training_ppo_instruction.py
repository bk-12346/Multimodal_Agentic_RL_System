import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.wrapper import MissionTokenizerWrapper
from models.fusion_extractor import MultimodalFusionExtractor

ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"
TOTAL_TIMESTEPS = 200_000  # harder task than Empty, give it more steps
MODEL_SAVE_PATH = "models/ppo_instruction_fetch"
LOG_PATH = "logs/ppo_instruction"


def make_env():
    env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    env = Monitor(env)
    return env


def main():
    with open(VOCAB_PATH) as f:
        vocab_size = len(json.load(f)["vocab"])

    env = make_vec_env(make_env, n_envs=4)

    policy_kwargs = dict(
        features_extractor_class=MultimodalFusionExtractor,
        features_extractor_kwargs=dict(
            vocab_size=vocab_size,
            vision_dim=128,
            text_dim=64,
            features_dim=128,
        ),
    )

    model = PPO(
        policy="MultiInputPolicy",   # required for Dict observation spaces
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_PATH,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    main()

##### RESULTS 1 #####
# - ep_rew_mean plateaued around 0.47–0.48 (not near 1.0 like Phase 0)
# - explained_variance is very low (0.017–0.043, should ideally trend toward 1.0 as the value function learns to predict returns)
# - Combined with clip_fraction = 0, this tells me the policy stopped meaningfully updating a while ago, and the value function still doesn't understand the task well. 
# Two real possibilities to distinguish:
# -> Fusion is working, agent just needs more training (200k wasn't enough for this task)
# -> Fusion is broken/being ignored, agent picks a strategy like "always grab the nearest object" which happens to get it right ~50% of the time by chance (Fetch-5x5-N2 has 2 objects, so random-ish object choice → roughly half-decent hit rate)
