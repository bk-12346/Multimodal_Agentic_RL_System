import gymnasium as gym


class HeldoutMissionFilterWrapper(gym.Wrapper):
    """
    Re-samples the episode (new seed) whenever the environment's mission
    would violate the held-out generalization split. Used only during
    TRAINING — at test time we do the opposite: force these combos.
    """

    def __init__(self, env, is_allowed_fn, max_resample_tries: int = 50):
        super().__init__(env)
        self.is_allowed_fn = is_allowed_fn
        self.max_resample_tries = max_resample_tries

    def reset(self, **kwargs):
        for _ in range(self.max_resample_tries):
            obs, info = self.env.reset(**kwargs)
            color = self.env.unwrapped.targetColor
            obj_type = self.env.unwrapped.targetType
            if self.is_allowed_fn(color, obj_type):
                return obs, info
            kwargs.pop("seed", None)  # drop seed after first try so resampling actually varies
        raise RuntimeError("Could not sample an allowed mission after max tries")