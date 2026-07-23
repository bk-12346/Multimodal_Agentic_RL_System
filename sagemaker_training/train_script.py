import argparse
import os
import gymnasium as gym
import minigrid
from minigrid.envs.fetch import FetchEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, "/opt/ml/code")  # SageMaker copies source_dir contents here
from models.fusion_extractor import MultimodalFusionExtractor
from envs.wrappers import MissionTokenizerWrapper


def make_env():
    env = FetchEnv(size=5, numObjs=1)  # N1: fast-converging navigation task, good for a short demo run
    env = MissionTokenizerWrapper(env, vocab_path="/opt/ml/code/configs/mission_vocab.json")
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    model_dir = os.environ["SM_MODEL_DIR"]

    env = make_vec_env(make_env, n_envs=4)

    with open("/opt/ml/code/configs/mission_vocab.json") as f:
        import json
        vocab_size = len(json.load(f)["vocab"])

    policy_kwargs = dict(
        features_extractor_class=MultimodalFusionExtractor,
        features_extractor_kwargs=dict(vocab_size=vocab_size, vision_dim=128, text_dim=64, features_dim=128),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save(os.path.join(model_dir, "ppo_sagemaker_demo"))
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()