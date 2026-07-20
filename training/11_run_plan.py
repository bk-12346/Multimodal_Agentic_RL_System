# updating this file so that it now uses LLM parser and prints the verification info
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.llm_planner import parse_instruction_llm
from planner.executor import execute_plan

MODEL_PATH = "models/ppo_curriculum_n2_finetuned"

if __name__ == "__main__":
    instruction = "I need a blue key first, then a green ball, and finally a red key"
    subgoals = parse_instruction_llm(instruction)

    print(f"Instruction: '{instruction}'")
    print(f"Parsed into {len(subgoals)} subgoals (via LLM):")
    for sg in subgoals:
        print(f"  - {sg['mission']}")
    print()

    results = execute_plan(subgoals, MODEL_PATH)

    print("\n=== Verification detail ===")
    for r in results:
        for a in r["attempts"]:
            print(f"'{r['subgoal']['mission']}' attempt {a['attempt']}: picked={a['picked']}, "
                  f"llm_says_match={a['llm_verified_match']} ({a['llm_reason']})")

    n_success = sum(r["resolved"] for r in results)
    print(f"\nCompleted {n_success}/{len(subgoals)} subgoals successfully.")

##### RESULTS ######
# Instruction: 'I need a blue key first, then a green ball, and finally a red key'
# Parsed into 3 subgoals (via LLM):
#   - go get a blue key
#   - go get a green ball
#   - go get a red key

# Subgoal 1/3: 'go get a blue key' -> SUCCESS (conf=0.92, 1 attempt(s)) ✓
# Subgoal 2/3: 'go get a green ball' -> SUCCESS (conf=0.96, 1 attempt(s)) ✓
# Subgoal 3/3: 'go get a red key' -> WRONG_OBJECT (conf=0.90, 3 attempt(s)) ✗ (gave up after retries)

# === Verification detail ===
# 'go get a blue key' attempt 1: picked=blue key, llm_says_match=True (Robot picked up the exact object requested (blue key))
# 'go get a green ball' attempt 1: picked=green ball, llm_says_match=True (Robot picked up the exact object requested (green ball).)
# 'go get a red key' attempt 1: picked=red ball, llm_says_match=False (Robot picked up a red ball, not a red key)
# 'go get a red key' attempt 2: picked=green ball, llm_says_match=False (Robot needed to get a red key, but picked up a green ball instead)      
# 'go get a red key' attempt 3: picked=yellow ball, llm_says_match=False (Robot picked up yellow ball instead of required red key)

# Completed 2/3 subgoals successfully.
# ----------------------------------------------------------------------------------------------

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from planner.rule_based_planner import parse_instruction
# from planner.executor import execute_plan

# MODEL_PATH = "models/ppo_curriculum_n2_finetuned"

# if __name__ == "__main__":
#     instruction = "get the blue key, then get the green ball, then get the red key"
#     subgoals = parse_instruction(instruction)

#     print(f"Instruction: '{instruction}'")
#     print(f"Parsed into {len(subgoals)} subgoals:")
#     for sg in subgoals:
#         print(f"  - {sg['mission']}")
#     print()

#     results = execute_plan(subgoals, MODEL_PATH)

#     n_success = sum(r["resolved"] for r in results)
#     print(f"\nCompleted {n_success}/{len(subgoals)} subgoals successfully.")

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