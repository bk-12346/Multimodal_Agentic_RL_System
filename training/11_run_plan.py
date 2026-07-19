import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.rule_based_planner import parse_instruction
from planner.executor import execute_plan

MODEL_PATH = "models/ppo_curriculum_n2_finetuned"

if __name__ == "__main__":
    instruction = "get the blue key, then get the green ball, then get the red key"
    subgoals = parse_instruction(instruction)

    print(f"Instruction: '{instruction}'")
    print(f"Parsed into {len(subgoals)} subgoals:")
    for sg in subgoals:
        print(f"  - {sg['mission']}")
    print()

    results = execute_plan(subgoals, MODEL_PATH)

    n_success = sum(r["resolved"] for r in results)
    print(f"\nCompleted {n_success}/{len(subgoals)} subgoals successfully.")

##### RESULTS 2: aftter adding confidence scoring #####
# Instruction: 'get the blue key, then get the green ball, then get the red key'                                                                 
# Parsed into 3 subgoals:                                                                                                                        
#   - go get a blue key
#   - go get a green ball
#   - go get a red key

# Subgoal 1/3: 'go get a blue key' -> SUCCESS (conf=0.92, 1 attempt(s)) ✓
# Subgoal 2/3: 'go get a green ball' -> SUCCESS (conf=0.96, 1 attempt(s)) ✓
# Subgoal 3/3: 'go get a red key' -> WRONG_OBJECT (conf=0.90, 3 attempt(s)) ✗ (gave up after retries)

# Completed 2/3 subgoals successfully.

##### RESULTS 1 #####
# Instruction: 'get the blue key, then get the green ball, then get the red key'
# Parsed into 3 subgoals:
#   - go get a blue key
#   - go get a green ball
#   - go get a red key

# Subgoal 1/3: 'go get a blue key' -> SUCCESS ✓
# Subgoal 2/3: 'go get a green ball' -> SUCCESS ✓
# Subgoal 3/3: 'go get a red key' -> WRONG_OBJECT ✗ (gave up after retries)

# Completed 2/3 subgoals successfully.