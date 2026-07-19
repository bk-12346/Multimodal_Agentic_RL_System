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
from training.callbacks import GeneralizationCheckpointCallback

ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
BASE_MODEL_PATH = "models/ppo_pretrained_n1"   # start fresh from N1, same starting point as before
TOTAL_TIMESTEPS = 900_000   # matches our two prior data points: 400k (45.8%) and 900k (0%)
BEST_MODEL_PATH = "models/ppo_pretrained_best_grey"
FINAL_MODEL_PATH = "models/ppo_pretrained_checkpoint_search_final"
LOG_PATH = "logs/ppo_pretrained_checkpoint_search"


def make_env():
    env = gym.make(ENV_ID)
    env = HeldoutMissionFilterWrapper(env, is_allowed_fn=is_training_mission)
    env = PretrainedTextWrapper(env)
    env = Monitor(env)
    return env


def main():
    env = make_vec_env(make_env, n_envs=4)
    model = PPO.load(BASE_MODEL_PATH, env=env, tensorboard_log=LOG_PATH)

    callback = GeneralizationCheckpointCallback(
        eval_freq=50_000,      # check every 50k steps -> 18 checkpoints across the run
        n_eval_episodes=30,    # enough to reduce noise without slowing training much
        save_path=BEST_MODEL_PATH,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, callback=callback)
    model.save(FINAL_MODEL_PATH)

    print("\n=== Full trajectory ===")
    for step, seen_acc, grey_acc in callback.history:
        print(f"step={step:>8}  seen={seen_acc:.3f}  grey={grey_acc:.3f}")
    print(f"\nBest grey_acc={callback.best_grey_acc:.3f}, saved at {BEST_MODEL_PATH}.zip")


if __name__ == "__main__":
    main()

##### RESULTS #####
# === Full trajectory ===
# step=  500032  seen=0.654  grey=0.154
# step=  700032  seen=0.808  grey=0.000
# step=  900032  seen=0.931  grey=0.000
# step= 1100032  seen=0.926  grey=0.000

# Best grey_acc=0.154, saved at models/ppo_pretrained_best_grey.zip