import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.pretrained_text_wrapper import PretrainedTextWrapper
from envs.heldout_filter_wrapper import HeldoutMissionFilterWrapper
from envs.build_vocab_heldout import is_training_mission

ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
BASE_MODEL_PATH = "models/ppo_pretrained_heldout_n2"   # continue from where we left off
ADDITIONAL_TIMESTEPS = 500_000
MODEL_SAVE_PATH = "models/ppo_pretrained_heldout_n2_extended"
LOG_PATH = "logs/ppo_pretrained_heldout_n2"  # same log dir -> continuous curve in TensorBoard


def make_env():
    env = gym.make(ENV_ID)
    env = HeldoutMissionFilterWrapper(env, is_allowed_fn=is_training_mission)
    env = PretrainedTextWrapper(env)
    env = Monitor(env)
    return env


def main():
    env = make_vec_env(make_env, n_envs=4)
    model = PPO.load(BASE_MODEL_PATH, env=env, tensorboard_log=LOG_PATH)
    model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, reset_num_timesteps=False)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    main()

##### RESULTS #####

# ----------------------------------------
# | rollout/                |            |
# |    ep_len_mean          | 6.43       |
# |    ep_rew_mean          | 0.884      |
# | time/                   |            |
# |    fps                  | 352        |
# |    iterations           | 486        |
# |    time_elapsed         | 1411       |
# |    total_timesteps      | 1198080    |
# | train/                  |            |
# |    approx_kl            | 0.06845171 |
# |    clip_fraction        | 0.0876     |
# |    clip_range           | 0.2        |
# |    entropy_loss         | -0.23      |
# |    explained_variance   | 0.238      |
# |    learning_rate        | 0.0003     |
# |    loss                 | 0.00394    |
# |    n_updates            | 4676       |
# |    policy_gradient_loss | 0.00844    |
# |    value_loss           | 0.0307     |
# ----------------------------------------
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 7.28        |
# |    ep_rew_mean          | 0.89        |
# | time/                   |             |
# |    fps                  | 352         |
# |    iterations           | 487         |
# |    time_elapsed         | 1413        |
# |    total_timesteps      | 1199104     |
# | train/                  |             |
# |    approx_kl            | 0.041498326 |
# |    clip_fraction        | 0.0657      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.285      |
# |    explained_variance   | 0.381       |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.0385      |
# |    n_updates            | 4680        |
# |    policy_gradient_loss | 0.00377     |
# |    value_loss           | 0.0248      |
# -----------------------------------------
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 8.99        |
# |    ep_rew_mean          | 0.83        |
# | time/                   |             |
# |    fps                  | 352         |
# |    iterations           | 488         |
# |    time_elapsed         | 1415        |
# |    total_timesteps      | 1200128     |
# | train/                  |             |
# |    approx_kl            | 0.045018565 |
# |    clip_fraction        | 0.1         |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.365      |
# |    explained_variance   | 0.31        |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.000116    |
# |    n_updates            | 4684        |
# |    policy_gradient_loss | 0.00484     |
# |    value_loss           | 0.0251      |
# -----------------------------------------
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 8.97        |
# |    ep_rew_mean          | 0.886       |
# | time/                   |             |
# |    fps                  | 353         |
# |    iterations           | 489         |
# |    time_elapsed         | 1418        |
# |    total_timesteps      | 1201152     |
# | train/                  |             |
# |    approx_kl            | 0.030729625 |
# |    clip_fraction        | 0.0881      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.329      |
# |    explained_variance   | 0.225       |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.00283     |
# |    n_updates            | 4688        |
# |    policy_gradient_loss | -0.00153    |
# |    value_loss           | 0.0362      |
# -----------------------------------------
