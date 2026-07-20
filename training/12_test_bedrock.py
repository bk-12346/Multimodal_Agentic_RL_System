import os
from openai import OpenAI

REGION = "eu-north-1"

client = OpenAI(
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/v1",
    api_key=os.environ["BEDROCK_API_KEY"],
)

response = client.chat.completions.create(
    model="moonshotai.kimi-k2.5",
    max_tokens=64,
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

print(response.choices[0].message.content)