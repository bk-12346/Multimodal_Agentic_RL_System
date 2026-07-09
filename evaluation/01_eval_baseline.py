import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ENV_ID = "MiniGrid-Empty-8x8-v0"
MODEL_PATH = "models/ppo_baseline_empty8x8"
N_EPISODES = 20

def main():
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = Monitor(env)

    model = PPO.load(MODEL_PATH)

    successes = 0
    rewards = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = truncated or terminated

        rewards.append(ep_reward)

        if ep_reward > 0:       # MiniGrid gives positive reward only on reaching goal
            successes += 1
    
    print(f"Success rate: {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.1f}%)")
    print(f"Mean reward: {sum(rewards)/len(rewards):.3f}")

if __name__ == "__main__":
    main()