import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sentence_transformers import SentenceTransformer

_model_cache = {}  # avoid reloading the model for every parallel env instance


def get_encoder():
    if "model" not in _model_cache:
        _model_cache["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache["model"]


class PretrainedTextWrapper(gym.ObservationWrapper):
    """
    Replaces MiniGrid's raw mission string with a fixed 384-dim sentence
    embedding from a pretrained MiniLM model, computed once at episode
    start (mission is constant within an episode). Unlike a from-scratch
    nn.Embedding, this gives words like 'grey' real semantic structure
    from pretraining, even if 'grey' was never a training TARGET here.
    """

    EMBED_DIM = 384

    def __init__(self, env):
        super().__init__(env)
        self.encoder = get_encoder()
        image_space = env.observation_space["image"]
        self.observation_space = spaces.Dict({
            "image": image_space,
            "mission_embedding": spaces.Box(
                low=-10.0, high=10.0, shape=(self.EMBED_DIM,), dtype=np.float32,
            ),
        })
        self._cached_mission = None
        self._cached_embedding = None

    def observation(self, obs):
        mission = obs["mission"]
        if mission != self._cached_mission:
            self._cached_embedding = self.encoder.encode(mission, convert_to_numpy=True).astype(np.float32)
            self._cached_mission = mission
        return {"image": obs["image"], "mission_embedding": self._cached_embedding}