import gymnasium as gym
import  minigrid

# Confirm the environment and see how missions vary
env = gym.make("MiniGrid-Fetch-5x5-N2-v0", render_mode="rgb_array")

for seed in range(5):
    obs, info = env.reset(seed=seed)
    print(f"seed={seed}, mission='{obs['mission']}'")

env.close()

# RESULT
# seed=0, mission='go get a blue key'
# seed=1, mission='you must fetch a purple key'
# seed=2, mission='go fetch a green ball'
# seed=3, mission='fetch a grey ball'
# seed=4, mission='go get a grey key'

