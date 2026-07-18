import gymnasium as gym


class HeldoutMissionForceWrapper(gym.Wrapper):
    """
    Opposite of HeldoutMissionFilterWrapper: re-samples until the mission
    DOES match one of the held-out conditions (never seen in training).
    Used only at eval time, to measure generalization specifically.
    """

    def __init__(self, env, is_heldout_fn, max_resample_tries: int = 100):
        super().__init__(env)
        self.is_heldout_fn = is_heldout_fn
        self.max_resample_tries = max_resample_tries

    def reset(self, **kwargs):
        for _ in range(self.max_resample_tries):
            obs, info = self.env.reset(**kwargs)
            color = self.env.unwrapped.targetColor
            obj_type = self.env.unwrapped.targetType
            if self.is_heldout_fn(color, obj_type):
                return obs, info
            kwargs.pop("seed", None)
        raise RuntimeError("Could not sample a held-out mission after max tries")