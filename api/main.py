import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import boto3
from fastapi import FastAPI
from pydantic import BaseModel
import gymnasium as gym
import minigrid
from stable_baselines3 import PPO

from envs.wrapper import MissionTokenizerWrapper

BUCKET_NAME = "bk-multiagent-rl-2026"
S3_MODEL_KEY = "models/ppo_curriculum_n2_finetuned/ppo_curriculum_n2_finetuned.zip"
LOCAL_MODEL_PATH = "models/ppo_curriculum_n2_finetuned.zip"
VOCAB_PATH = "configs/mission_vocab.json"
ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
MAX_STEPS = 100

app = FastAPI(title="Multimodal Agentic RL API", version="0.1.0")
_model = None


@app.on_event("startup")
def load_model():
    global _model
    local_path = Path(LOCAL_MODEL_PATH)

    if not local_path.exists():
        print(f"Model not found locally, downloading from s3://{BUCKET_NAME}/{S3_MODEL_KEY} ...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(BUCKET_NAME, S3_MODEL_KEY, str(local_path))
        print("Download complete.")
    else:
        print(f"Using existing local model at {local_path}")

    _model = PPO.load(str(local_path).removesuffix(".zip"))
    print("Model loaded and ready.")


class FetchRequest(BaseModel):
    color: str
    object_type: str
    seed: int = 0


class FetchResponse(BaseModel):
    outcome: str
    steps_taken: int
    picked_object: str
    mean_confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/fetch", response_model=FetchResponse)
def fetch(request: FetchRequest):
    raw_env = gym.make(ENV_ID)
    env = MissionTokenizerWrapper(raw_env, vocab_path=VOCAB_PATH)

    seed = request.seed
    for _ in range(50):
        obs, info = env.reset(seed=seed)
        if (env.unwrapped.targetColor, env.unwrapped.targetType) == (request.color, request.object_type):
            break
        seed += 1000

    step_count = 0
    done = False
    while not done and step_count < MAX_STEPS:
        action, _ = _model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    carrying = env.unwrapped.carrying
    env.close()

    picked = f"{carrying.color} {carrying.type}" if carrying else "nothing"
    if carrying and (carrying.color, carrying.type) == (request.color, request.object_type):
        outcome = "SUCCESS"
    elif carrying:
        outcome = "WRONG_OBJECT"
    else:
        outcome = "TIMEOUT"

    return FetchResponse(outcome=outcome, steps_taken=step_count, picked_object=picked, mean_confidence=0.0)

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from fastapi import FastAPI
# from pydantic import BaseModel
# import gymnasium as gym
# import minigrid
# from stable_baselines3 import PPO

# from envs.wrapper import MissionTokenizerWrapper

# MODEL_PATH = "models/ppo_curriculum_n2_finetuned"
# VOCAB_PATH = "configs/mission_vocab.json"
# ENV_ID = "MiniGrid-Fetch-5x5-N2-v0"
# MAX_STEPS = 100

# app = FastAPI(title="Multimodal Agentic RL API", version="0.1.0")

# _model = None  # loaded once at startup, not per-request


# @app.on_event("startup")
# def load_model():
#     global _model
#     _model = PPO.load(MODEL_PATH)
#     print(f"Loaded model from {MODEL_PATH}")


# class FetchRequest(BaseModel):
#     color: str
#     object_type: str
#     seed: int = 0


# class FetchResponse(BaseModel):
#     outcome: str          # SUCCESS, WRONG_OBJECT, or TIMEOUT
#     steps_taken: int
#     picked_object: str
#     mean_confidence: float


# @app.get("/health")
# def health():
#     return {"status": "ok", "model_loaded": _model is not None}


# @app.post("/fetch", response_model=FetchResponse)
# def fetch(request: FetchRequest):
#     raw_env = gym.make(ENV_ID)
#     env = MissionTokenizerWrapper(raw_env, vocab_path=VOCAB_PATH)

#     seed = request.seed
#     for _ in range(50):
#         obs, info = env.reset(seed=seed)
#         if (env.unwrapped.targetColor, env.unwrapped.targetType) == (request.color, request.object_type):
#             break
#         seed += 1000

#     step_count = 0
#     done = False
#     confidences = []

#     while not done and step_count < MAX_STEPS:
#         action, _ = _model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         step_count += 1

#     carrying = env.unwrapped.carrying
#     env.close()

#     picked = f"{carrying.color} {carrying.type}" if carrying else "nothing"
#     if carrying and (carrying.color, carrying.type) == (request.color, request.object_type):
#         outcome = "SUCCESS"
#     elif carrying:
#         outcome = "WRONG_OBJECT"
#     else:
#         outcome = "TIMEOUT"

#     return FetchResponse(
#         outcome=outcome,
#         steps_taken=step_count,
#         picked_object=picked,
#         mean_confidence=0.0,  # placeholder, wire in predict_with_confidence from planner/executor.py if you want this populated
#     )