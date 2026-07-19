import gymnasium as gym
import minigrid
from stable_baselines3.common.callbacks import BaseCallback

from envs.pretrained_text_wrapper import PretrainedTextWrapper
from envs.heldout_filter_wrapper import HeldoutMissionFilterWrapper
from envs.heldout_force_wrapper import HeldoutMissionForceWrapper
from envs.build_vocab_heldout import is_training_mission, HELD_OUT_COLOR


def _quick_eval(env, model, n_episodes):
    picked, correct = 0, 0
    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        target = (env.unwrapped.targetColor, env.unwrapped.targetType)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        carrying = env.unwrapped.carrying
        if carrying:
            picked += 1
            correct += (carrying.color, carrying.type) == target
    acc = correct / picked if picked else 0.0
    return acc, picked / n_episodes


class GeneralizationCheckpointCallback(BaseCallback):
    """
    Every eval_freq steps, measures accuracy on BOTH seen-vocab and
    held-out (grey) missions. Saves the checkpoint that maximizes
    grey accuracy (our target metric), separate from the final model,
    so we capture the peak trade-off point rather than whatever the
    training run happens to end on.
    """

    ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"

    def __init__(self, eval_freq: int, n_eval_episodes: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_grey_acc = -1.0
        self.history = []  # (timestep, seen_acc, grey_acc) for later plotting/reporting

        seen_env = gym.make(self.ENV_ID)
        seen_env = HeldoutMissionFilterWrapper(seen_env, is_allowed_fn=is_training_mission)
        self.seen_env = PretrainedTextWrapper(seen_env)

        grey_env = gym.make(self.ENV_ID)
        grey_env = HeldoutMissionForceWrapper(grey_env, is_heldout_fn=lambda c, t: c == HELD_OUT_COLOR)
        self.grey_env = PretrainedTextWrapper(grey_env)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            seen_acc, seen_pickup = _quick_eval(self.seen_env, self.model, self.n_eval_episodes)
            grey_acc, grey_pickup = _quick_eval(self.grey_env, self.model, self.n_eval_episodes)

            self.history.append((self.num_timesteps, seen_acc, grey_acc))
            self.logger.record("generalization/seen_accuracy", seen_acc)
            self.logger.record("generalization/grey_accuracy", grey_acc)
            self.logger.record("generalization/grey_pickup_rate", grey_pickup)

            print(f"[step {self.num_timesteps:>8}] seen_acc={seen_acc:.3f} (pickup={seen_pickup:.2f})  "
                  f"grey_acc={grey_acc:.3f} (pickup={grey_pickup:.2f})")

            if grey_acc > self.best_grey_acc:
                self.best_grey_acc = grey_acc
                self.model.save(self.save_path)
                print(f"  -> new best grey_acc={grey_acc:.3f}, saved to {self.save_path}.zip")

        return True