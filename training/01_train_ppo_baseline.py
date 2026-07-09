from turtle import forward
import gymnasium as gym
import minigrid
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ENV_ID = "MiniGrid-Empty-8x8-v0"
TOTAL_TIMESTEPS = 100_000
MODEL_SAVE_PATH = "models/ppo_baseline_empty8x8"
LOG_PATH = "logs/ppo_baseline"

class MiniGridCNN(BaseFeaturesExtractor):
    """
    Small CNN sized for MiniGrid's 7x7x33 partial observations.
    SB3's default NatureCNN assumes ~84x84 Atari frames and breaks on
    inputs this small (kernel bigger than the image), so we use 3x3 kernels
    with no aggressive downsampling instead.
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]     # after VecTransposeImage: (C, H, W)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),    # padding=1 with stride=1 keeps spatial dimensions from shrinking too fast on a 7×7 input.
            nn.ReLU(),                                                                # No pooling layers — pooling would destroy spatial info almost immediately on something this small.
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),                  # features_dim=128 is the output embedding size that will later get concatenated with the text/instruction embedding.
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def make_env():
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)          # Dict obs -> just the 'image' array (7, 7, 3)
    env = Monitor(env)      # tracks episode reward/length for logging

    return env
    
def main():
    env = make_vec_env(make_env, n_envs=4)  # 4 parallel environments to speed-up training

    policy_kwargs = dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        policy = "CnnPolicy",
        env = env,
        verbose = 1,        # 1 for info messages
        policy_kwargs=policy_kwargs,
        tensorboard_log = LOG_PATH,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=256,     # SB3's built-in small CNN, fine for a 7×7×3 input.
        n_epochs=4,
        gamma=0.99
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}.zip")

if __name__ == "__main__":
    main()