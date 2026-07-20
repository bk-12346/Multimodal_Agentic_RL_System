import os
import json
from openai import OpenAI

REGION = "eu-north-1"
MODEL_ID = "moonshotai.kimi-k2.5"

_client = OpenAI(
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/v1",
    api_key=os.environ["BEDROCK_API_KEY"],
)


def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Calls the LLM and expects a JSON object back. 
    Used for both instruction parsing and outcome verification, so we centralize the
    'ask for JSON, parse it safely' logic in one place.
    """
    response = _client.chat.completions.create(
        model=MODEL_ID,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"LLM did not return valid JSON. Raw output:\n{raw}")