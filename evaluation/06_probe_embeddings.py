import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
from stable_baselines3 import PPO

MODEL_PATH = "models/ppo_curriculum_n2_finetuned"
VOCAB_PATH = "configs/mission_vocab.json"

def main():
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)["vocab"]

    model = PPO.load(MODEL_PATH)
    embedding_layer = model.policy.features_extractor.embedding

    colors = ["blue", "red", "green", "purple", "yellow", "grey"]
    for c in colors:
        idx = vocab[c]
        vec = embedding_layer.weight[idx].detach()
        print(f"{c:8s} idx={idx:2d} norm={vec.norm().item():.4f} first5={vec[:5].numpy().round(3)}")

if __name__ == "__main__":
    main()

##### RESULTS #####
# blue     idx= 5 norm=7.8900 first5=[-0.397 -0.237  0.068  1.33   0.111]
# red      idx=14 norm=8.4747 first5=[-0.959 -0.194 -0.018  1.905 -1.02 ]
# green    idx=11 norm=8.2597 first5=[-0.846 -0.294  0.131  1.548  0.057]
# purple   idx=10 norm=8.5884 first5=[ 0.04   0.277 -1.339  1.214  3.364]
# yellow   idx=15 norm=8.9061 first5=[ 0.879  0.798 -0.284 -0.577 -0.859]
# grey     idx=13 norm=7.6584 first5=[ 1.684 -1.33   1.254  0.213  0.237]