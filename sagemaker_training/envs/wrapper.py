import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MissionTokenizerWrapper(gym.ObservationWrapper):
    """Converts MiniGrid's Dict obs (image, direction, mission) into
    (image, mission_token_ids) so SB3 can consume it as a Dict observation
    space with fixed-shape arrays instead of a raw string.
    """

    def __init__(self, env, vocab_path: str):
        super().__init__(env)
        with open(vocab_path) as f:
            payload = json.load(f)
        self.vocab = payload["vocab"]
        self.max_len = payload["max_len"]

        image_space = env.observation_space["image"]
        self.observation_space = spaces.Dict({
            "image": image_space,
            "mission": spaces.Box(
                low=0, high=len(self.vocab) - 1,
                shape=(self.max_len,), dtype=np.int64,
            ),
        })

    def observation(self, obs):
        words = obs["mission"].lower().split()
        ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in words]
        ids = ids[: self.max_len]
        ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return {"image": obs["image"], "mission": np.array(ids, dtype=np.int64)}