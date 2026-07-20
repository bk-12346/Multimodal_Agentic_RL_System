import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.llm_planner import parse_instruction_llm, verify_outcome_llm

# Test 1: flexible parsing -- deliberately NOT matching the rigid "X, then Y" template
instruction = "I need a red key first, and after that grab me a green ball please"
subgoals = parse_instruction_llm(instruction)
print("Parsed subgoals:")
for sg in subgoals:
    print(f"  {sg}")

# Test 2: verification -- both a matching and mismatching case
print("\nVerification tests:")
print(verify_outcome_llm("go get a red key", "blue ball"))
print(verify_outcome_llm("go get a red key", "red key"))

##### RESULTS #####
# Parsed subgoals:
#   {'color': 'red', 'type': 'key', 'mission': 'go get a red key'}
#   {'color': 'green', 'type': 'ball', 'mission': 'go get a green ball'}

# Verification tests:
# {'matches': False, 'reason': 'mission asked for red key, robot got blue ball'}
# {'matches': True, 'reason': 'Robot picked up the exact item requested (red key).'}