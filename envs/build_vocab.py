import json
import gymnasium as gym
import minigrid

ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
N_SAMPLES = 500     # enough resets to see the full vocabulary
VOCAB_SAVE_PATH = "configs/mission_vocab.json"

def main():
    env = gym.make(ENV_ID)
    vocab = {"<pad>":0, "<unk>":1}

    max_len = 0
    for seed in range(N_SAMPLES):
        obs, _ = env.reset(seed=seed)
        words = obs["mission"].lower().split()
        max_len = max(max_len, len(words))
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)

    env.close()

    payload = {"vocab": vocab, "max_len": max_len}
    with open(VOCAB_SAVE_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Vocab size: {len(vocab)}")
    print(f"Max mission length (words): {max_len}")
    print(f"Saved to {VOCAB_SAVE_PATH}")

if __name__ == "__main__":
    main()
