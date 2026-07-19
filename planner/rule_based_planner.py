import re

COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
OBJECT_TYPES = ["key", "ball"]


def parse_instruction(instruction: str) -> list[dict]:
    """
    Splits a multi-step instruction like 'get the blue key, then get the
    green ball' into an ordered list of subgoals. 
    Simple rule-based parsing (regex over a known color/object vocabulary) -- deliberately not an LLM,
    per the PRD's own guidance to start simple.
    """
    segments = re.split(r"\bthen\b|,", instruction.lower())
    subgoals = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        color = next((c for c in COLORS if c in seg), None)
        obj_type = next((t for t in OBJECT_TYPES if t in seg), None)
        if color and obj_type:
            subgoals.append({
                "color": color,
                "type": obj_type,
                "mission": f"go get a {color} {obj_type}",
            })
    return subgoals