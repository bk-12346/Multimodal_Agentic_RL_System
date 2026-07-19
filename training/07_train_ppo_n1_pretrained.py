# Since the observation space and feature extractor both changed shape, we can't warm-start from the old checkpoint 
# need a fresh N1 run first.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from minigrid.envs.fetch import FetchEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.pretrained_text_wrapper import PretrainedTextWrapper
from models.fusion_extractor_pretrained import PretrainedFiLMExtractor

TOTAL_TIMESTEPS = 300_000
MODEL_SAVE_PATH = "models/ppo_pretrained_n1"
LOG_PATH = "logs/ppo_pretrained_n1"


def make_env():
    env = FetchEnv(size=5, numObjs=1)
    env = PretrainedTextWrapper(env)
    env = Monitor(env)
    return env


def main():
    env = make_vec_env(make_env, n_envs=4)

    policy_kwargs = dict(
        features_extractor_class=PretrainedFiLMExtractor,
        features_extractor_kwargs=dict(vision_dim=128, text_dim=64, features_dim=128),
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

##### RESULTS #####

# ------------------------------------------
# | rollout/                |              |
# |    ep_len_mean          | 4.12         |
# |    ep_rew_mean          | 0.97         |
# | time/                   |              |
# |    fps                  | 407          |
# |    iterations           | 292          |
# |    time_elapsed         | 734          |
# |    total_timesteps      | 299008       |
# | train/                  |              |
# |    approx_kl            | 0.0012108255 |
# |    clip_fraction        | 0.0117       |
# |    clip_range           | 0.2          |
# |    entropy_loss         | -0.0185      |
# |    explained_variance   | 0.68         |
# |    learning_rate        | 0.0003       |
# |    loss                 | -0.00172     |
# |    n_updates            | 1164         |
# |    policy_gradient_loss | -0.0003      |
# |    value_loss           | 0.000136     |
# ------------------------------------------
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 3.7         |
# |    ep_rew_mean          | 0.973       |
# | time/                   |             |
# |    fps                  | 406         |
# |    iterations           | 293         |
# |    time_elapsed         | 737         |
# |    total_timesteps      | 300032      |
# | train/                  |             |
# |    approx_kl            | 0.013905624 |
# |    clip_fraction        | 0.0127      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.0196     |
# |    explained_variance   | 0.652       |
# |    learning_rate        | 0.0003      |
# |    loss                 | -0.00883    |
# |    n_updates            | 1168        |
# |    policy_gradient_loss | -0.00146    |
# |    value_loss           | 0.000143    |
# -----------------------------------------
