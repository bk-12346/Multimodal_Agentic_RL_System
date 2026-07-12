import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
import json
from minigrid.envs.fetch import FetchEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.wrapper import MissionTokenizerWrapper
from models.fusion_extractor import MultimodalFusionExtractor

VOCAB_PATH = "configs/mission_vocab.json"
TOTAL_TIMESTEPS = 300_000
MODEL_SAVE_PATH = "models/ppo_curriculum_n1"
LOG_PATH = "logs/ppo_curriculum_n1"


def make_env():
    env = FetchEnv(size=5, numObjs=1)   # numObjs=1 = no distractor, matches Fetch-5x5-N2's size otherwise
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
            vocab_size=vocab_size, vision_dim=128, text_dim=64, features_dim=128,
        ),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_PATH,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    main()