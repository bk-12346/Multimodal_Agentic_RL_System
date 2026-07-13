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

##### RESULTS 2: from FiLM fusion extractor #####
# ------------------------------------------
# | rollout/                |              |
# |    ep_len_mean          | 3.86         |
# |    ep_rew_mean          | 0.972        |
# | time/                   |              |
# |    fps                  | 963          |
# |    iterations           | 291          |
# |    time_elapsed         | 309          |
# |    total_timesteps      | 297984       |
# | train/                  |              |
# |    approx_kl            | 0.0031526578 |
# |    clip_fraction        | 0.00171      |
# |    clip_range           | 0.2          |
# |    entropy_loss         | -0.0122      |
# |    explained_variance   | 0.671        |
# |    learning_rate        | 0.0003       |
# |    loss                 | 1.4e-05      |
# |    n_updates            | 1160         |
# |    policy_gradient_loss | -0.00026     |
# |    value_loss           | 0.000127     |
# ------------------------------------------
# ------------------------------------------
# | rollout/                |              |
# |    ep_len_mean          | 4.08         |
# |    ep_rew_mean          | 0.971        |
# | time/                   |              |
# |    fps                  | 963          |
# |    iterations           | 292          |
# |    time_elapsed         | 310          |
# |    total_timesteps      | 299008       |
# | train/                  |              |
# |    approx_kl            | 0.0007598886 |
# |    clip_fraction        | 0.00244      |
# |    clip_range           | 0.2          |
# |    entropy_loss         | -0.0176      |
# |    explained_variance   | 0.698        |
# |    learning_rate        | 0.0003       |
# |    loss                 | -0.00328     |
# |    n_updates            | 1164         |
# |    policy_gradient_loss | -0.000183    |
# |    value_loss           | 0.000123     |
# ------------------------------------------
# ----------------------------------------
# | rollout/                |            |
# |    ep_len_mean          | 3.66       |
# |    ep_rew_mean          | 0.974      |
# | time/                   |            |
# |    fps                  | 963        |
# |    iterations           | 293        |
# |    time_elapsed         | 311        |
# |    total_timesteps      | 300032     |
# | train/                  |            |
# |    approx_kl            | 0.01181892 |
# |    clip_fraction        | 0.0127     |
# |    clip_range           | 0.2        |
# |    entropy_loss         | -0.0258    |
# |    explained_variance   | 0.658      |
# |    learning_rate        | 0.0003     |
# |    loss                 | -0.00479   |
# |    n_updates            | 1168       |
# |    policy_gradient_loss | -0.00126   |
# |    value_loss           | 0.000134   |
# ----------------------------------------

##### RESULTS 1: from NLP+cat fusion extractor #####
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 3.93        |
# |    ep_rew_mean          | 0.972       |
# | time/                   |             |
# |    fps                  | 617         |
# |    iterations           | 291         |
# |    time_elapsed         | 482         |
# |    total_timesteps      | 297984      |
# | train/                  |             |
# |    approx_kl            | 0.012318168 |
# |    clip_fraction        | 0.00659     |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.0242     |
# |    explained_variance   | 0.578       |
# |    learning_rate        | 0.0003      |
# |    loss                 | -0.00552    |
# |    n_updates            | 1160        |
# |    policy_gradient_loss | -0.00144    |
# |    value_loss           | 0.000159    |
# -----------------------------------------
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 4.32        |
# |    ep_rew_mean          | 0.969       |
# | time/                   |             |
# |    fps                  | 618         |
# |    iterations           | 292         |
# |    time_elapsed         | 483         |
# |    total_timesteps      | 299008      |
# | train/                  |             |
# |    approx_kl            | 0.029981898 |
# |    clip_fraction        | 0.0239      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -0.0298     |
# |    explained_variance   | 0.562       |
# |    learning_rate        | 0.0003      |
# |    loss                 | 0.00249     |
# |    n_updates            | 1164        |
# |    policy_gradient_loss | -0.00313    |
# |    value_loss           | 0.000256    |
# -----------------------------------------
# ----------------------------------------
# | rollout/                |            |
# |    ep_len_mean          | 5.42       |
# |    ep_rew_mean          | 0.961      |
# | time/                   |            |
# |    fps                  | 619        |
# |    iterations           | 293        |
# |    time_elapsed         | 484        |
# |    total_timesteps      | 300032     |
# | train/                  |            |
# |    approx_kl            | 0.04152449 |
# |    clip_fraction        | 0.0474     |
# |    clip_range           | 0.2        |
# |    entropy_loss         | -0.0562    |
# |    explained_variance   | 0.0545     |
# |    learning_rate        | 0.0003     |
# |    loss                 | 0.0476     |
# |    n_updates            | 1168       |
# |    policy_gradient_loss | 0.0101     |
# |    value_loss           | 0.00118    |
# ----------------------------------------