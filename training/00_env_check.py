import gymnasium as gym
import minigrid     # registers MiniGrid envs with gym


env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Sample obs keys:", obs.keys() if isinstance(obs, dict) else type(obs))

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"action={action} reward={reward} terminated={terminated}")

env.close()

# RESULT
# Observation space: Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function EmptyEnv._gen_mission at 0x000001FC698F2160>, None))
# Action space: Discrete(7)
# Sample obs keys: dict_keys(['image', 'direction', 'mission'])
# action=0 reward=0 terminated=False
# action=2 reward=0 terminated=False
# action=6 reward=0 terminated=False
# action=0 reward=0 terminated=False
# action=5 reward=0 terminated=False

## image: 7×7×3 (small, partially-observable local view — easy for a tiny CNN)
## direction: which way the agent faces
## mission: the instruction string, already present even in this baseline
## action: 7 discrete actions (turn left/right, move forward, pickup, drop, toggle, done)