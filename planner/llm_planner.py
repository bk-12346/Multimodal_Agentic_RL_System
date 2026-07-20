import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.bedrock_client import call_llm_json

COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
OBJECT_TYPES = ["key", "ball"]

PARSE_SYSTEM_PROMPT = f"""You convert natural language instructions into a JSON list of subgoals for a robot that fetches objects.

Valid colors: {COLORS}
Valid object types: {OBJECT_TYPES}

The robot can only fetch ONE object per subgoal. Break multi-step instructions into ordered subgoals.
If a color or object type isn't in the valid lists, pick the closest match or omit that subgoal.

Respond ONLY with valid JSON in this exact format, no other text:
{{"subgoals": [{{"color": "...", "type": "...", "mission": "go get a ... ..."}}]}}
"""


def parse_instruction_llm(instruction: str) -> list[dict]:
    result = call_llm_json(PARSE_SYSTEM_PROMPT, instruction)
    return result.get("subgoals", [])


VERIFY_SYSTEM_PROMPT = """You verify whether a robot completed its fetch task correctly.

Respond ONLY with valid JSON in this exact format, no other text:
{"matches": true/false, "reason": "short explanation"}
"""


def verify_outcome_llm(mission: str, picked_description: str) -> dict:
    """
    Post-hoc check: did the robot pick up what the mission asked for?
    This targets the specific failure mode found in Phase 3 testing --
    the RL policy's own confidence score does NOT distinguish correct
    from incorrect pickups (both ~0.90+), so an external check is needed.
    """
    user_prompt = f"Mission: \"{mission}\"\nRobot picked up: \"{picked_description}\"\nDoes this match the mission?"
    return call_llm_json(VERIFY_SYSTEM_PROMPT, user_prompt)