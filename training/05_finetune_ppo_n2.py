import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.wrapper import MissionTokenizerWrapper

ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"
BASE_MODEL_PATH = "models/ppo_curriculum_n1"
FINETUNE_TIMESTEPS = 400_000
MODEL_SAVE_PATH = "models/ppo_curriculum_n2_finetuned"
LOG_PATH = "logs/ppo_curriculum_n2_finetune"


def make_env():
    env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    env = Monitor(env)
    return env


def main():
    env = make_vec_env(make_env, n_envs=4)

    model = PPO.load(BASE_MODEL_PATH, env=env, tensorboard_log=LOG_PATH)

    model.learn(total_timesteps=FINETUNE_TIMESTEPS, reset_num_timesteps=False)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved fine-tuned model to {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    main()

# reset_num_timesteps=False keeps the timestep counter (and thus the learning-rate/entropy schedules, if you add decay later) continuing from where N1 left off,
# rather than restarting — matters for accurate TensorBoard comparisons across the two phases.

##### RESULTS 2: FiLM fusion extractor #####
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 3.66        |
# |    ep_rew_mean          | 0.652       |
# | time/                   |             |
# |    fps                  | 845         |
# |    iterations           | 390         |
# |    time_elapsed         | 472         |
# |    total_timesteps      | 699392      |
# | train/                  |             |
# |    approx_kl            | 0.020980349 |
# |    clip_fraction        | 0.0325      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.0882     |
# |    explained_variance   | 0.498       |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.0295      |
# |    n_updates            | 2728        |
# |    policy_gradient_loss | 0.00446     |
# |    value_loss           | 0.0542      |
# -----------------------------------------
# ------------------------------------------
# | rollout/                |              |
# |    ep_len_mean          | 8.26         |
# |    ep_rew_mean          | 0.683        |
# | time/                   |              |
# |    fps                  | 845          |
# |    iterations           | 391          |
# |    time_elapsed         | 473          |
# |    total_timesteps      | 700416       |
# | train/                  |              |
# |    approx_kl            | 0.0147127565 |
# |    clip_fraction        | 0.0244       |
# |    clip_range           | 0.2          |
# |    entropy_loss         | -0.0781      |
# |    explained_variance   | 0.427        |
# |    learning_rate        | 0.0003       |
# |    loss                 | 0.0192       |
# |    n_updates            | 2732         |
# |    policy_gradient_loss | -0.00291     |
# |    value_loss           | 0.0603       |
# ------------------------------------------
##### RESULTS 1: NLP+cat fusion extractor #####
# ------------------------------------------
# | rollout/                |              |
# |    ep_len_mean          | 28.3         |
# |    ep_rew_mean          | 0.514        |
# | time/                   |              |
# |    fps                  | 433          |
# |    iterations           | 390          |
# |    time_elapsed         | 921          |
# |    total_timesteps      | 699392       |
# | train/                  |              |
# |    approx_kl            | 0.0121974265 |
# |    clip_fraction        | 0.0813       |
# |    clip_range           | 0.2          |
# |    entropy_loss         | -0.637       |
# |    explained_variance   | 0.228        |
# |    learning_rate        | 0.0003       |
# |    loss                 | 0.00572      |
# |    n_updates            | 2728         |
# |    policy_gradient_loss | 0.00181      |
# |    value_loss           | 0.0325       |
# ------------------------------------------
# ------------------------------------------
# | rollout/                |              |
# |    ep_len_mean          | 23.3         |
# |    ep_rew_mean          | 0.553        |
# | time/                   |              |
# |    fps                  | 434          |
# |    iterations           | 391          |
# |    time_elapsed         | 922          |
# |    total_timesteps      | 700416       |
# | train/                  |              |
# |    approx_kl            | 0.0055704964 |
# |    clip_fraction        | 0.0613       |
# |    clip_range           | 0.2          |
# |    entropy_loss         | -0.678       |
# |    explained_variance   | 0.194        |
# |    learning_rate        | 0.0003       |
# |    loss                 | -0.0111      |
# |    n_updates            | 2732         |
# |    policy_gradient_loss | -0.00304     |
# |    value_loss           | 0.0198       |
# ------------------------------------------