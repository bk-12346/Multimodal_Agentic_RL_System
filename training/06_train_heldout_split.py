import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.wrapper import MissionTokenizerWrapper
from envs.heldout_filter_wrapper import HeldoutMissionFilterWrapper
from envs.build_vocab_heldout import is_training_mission

ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
VOCAB_PATH = "configs/mission_vocab.json"
BASE_MODEL_PATH = "models/ppo_curriculum_n1"   # start from the N1 navigation skill, same as before
TOTAL_TIMESTEPS = 400_000
MODEL_SAVE_PATH = "models/ppo_heldout_split"
LOG_PATH = "logs/ppo_heldout_split"


def make_env():
    env = gym.make(ENV_ID)
    env = HeldoutMissionFilterWrapper(env, is_allowed_fn=is_training_mission)
    env = MissionTokenizerWrapper(env, vocab_path=VOCAB_PATH)
    env = Monitor(env)
    return env


def main():
    env = make_vec_env(make_env, n_envs=4)
    model = PPO.load(BASE_MODEL_PATH, env=env, tensorboard_log=LOG_PATH)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    main()

##### RESULTS #####
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 5.39        |
# |    ep_rew_mean          | 0.777       |
# | time/                   |             |
# |    fps                  | 927         |
# |    iterations           | 390         |
# |    time_elapsed         | 430         |
# |    total_timesteps      | 699392      |
# | train/                  |             |
# |    approx_kl            | 0.009956846 |
# |    clip_fraction        | 0.0374      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.116      |
# |    explained_variance   | 0.387       |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.00892     |
# |    n_updates            | 2728        |
# |    policy_gradient_loss | -0.00239    |
# |    value_loss           | 0.042       |
# -----------------------------------------
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 7.71        |
# |    ep_rew_mean          | 0.821       |
# | time/                   |             |
# |    fps                  | 927         |
# |    iterations           | 391         |
# |    time_elapsed         | 431         |
# |    total_timesteps      | 700416      |
# | train/                  |             |
# |    approx_kl            | 0.048548926 |
# |    clip_fraction        | 0.0811      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.172      |
# |    explained_variance   | 0.216       |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.0335      |
# |    n_updates            | 2732        |
# |    policy_gradient_loss | 0.00375     |
# |    value_loss           | 0.0707      |
# -----------------------------------------