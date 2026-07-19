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
BASE_MODEL_PATH = "models/ppo_pretrained_n1"
FINETUNE_TIMESTEPS = 400_000
MODEL_SAVE_PATH = "models/ppo_pretrained_heldout_n2"
LOG_PATH = "logs/ppo_pretrained_heldout_n2"


def make_env():
    env = gym.make(ENV_ID)
    env = HeldoutMissionFilterWrapper(env, is_allowed_fn=is_training_mission)  # same held-out split: no grey, no purple+key
    env = PretrainedTextWrapper(env)
    env = Monitor(env)
    return env


def main():
    env = make_vec_env(make_env, n_envs=4)
    model = PPO.load(BASE_MODEL_PATH, env=env, tensorboard_log=LOG_PATH)
    model.learn(total_timesteps=FINETUNE_TIMESTEPS, reset_num_timesteps=False)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    main()

##### RESULTS #####

# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 4.62        |
# |    ep_rew_mean          | 0.629       |
# | time/                   |             |
# |    fps                  | 293         |
# |    iterations           | 390         |
# |    time_elapsed         | 1359        |
# |    total_timesteps      | 699392      |
# | train/                  |             |
# |    approx_kl            | 0.030246498 |
# |    clip_fraction        | 0.0625      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.596      |
# |    explained_variance   | 0.415       |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.0489      |
# |    n_updates            | 2728        |
# |    policy_gradient_loss | -4.38e-05   |
# |    value_loss           | 0.109       |
# -----------------------------------------
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 5.22        |
# |    ep_rew_mean          | 0.559       |
# | time/                   |             |
# |    fps                  | 293         |
# |    iterations           | 391         |
# |    time_elapsed         | 1362        |
# |    total_timesteps      | 700416      |
# | train/                  |             |
# |    approx_kl            | 0.008293975 |
# |    clip_fraction        | 0.0337      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.628      |
# |    explained_variance   | 0.39        |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.0475      |
# |    n_updates            | 2732        |
# |    policy_gradient_loss | -0.000102   |
# |    value_loss           | 0.103       |
# -----------------------------------------