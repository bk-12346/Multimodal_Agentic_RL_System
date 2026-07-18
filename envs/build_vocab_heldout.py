# build a version of the vocab/env setup that deliberately excludes a color and a specific color+object pairing 
# from training, so we can test whether FiLM learned a general "match color+type to target" rule 
# or just memorized specific pairings.

import json

VOCAB_SAVE_PATH = "configs/mission_vocab.json"  # reuse the same vocab file — the vocab
                                                  # itself doesn't change, only which
                                                  # missions we allow during training

HELD_OUT_COLOR = "grey"          # never appears in ANY training mission
HELD_OUT_COMBO = ("purple", "key")  # purple and key both appear elsewhere in training,
                                    # just never paired together

def is_training_mission(color: str, obj_type: str) -> bool:
    if color == HELD_OUT_COLOR:
        return False
    if (color, obj_type) == HELD_OUT_COMBO:
        return False
    return True

if __name__ == "__main__":
    with open(VOCAB_SAVE_PATH) as f:
        print(json.load(f)["vocab"])
    print(f"\nHeld out entirely: color='{HELD_OUT_COLOR}'")
    print(f"Held out combo: {HELD_OUT_COMBO[0]} {HELD_OUT_COMBO[1]}")