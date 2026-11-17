
import json
from src.dataset.data_utils import llava_to_openai_lvr
from src.constants import LVR_TOKEN, LVR_START_TOKEN, LVR_END_TOKEN

with open("mac_demo/data/lvr_train.json", "r") as f:
    data = json.load(f)

example = data[0]

# Simulate the actual output from the bounding-box mapping
lvr_indices = list(range(627))

converted = llava_to_openai_lvr(
    example["conversations"],
    is_video=False,
    lvr_token_idxs_list=[lvr_indices]
)

print("Converted conversation:")
for message in converted:
    print("\nRole:", message["role"])
    print("Content:", message["content"][:150])

assistant_text = converted[1]["content"]

print("\nValidation:")
print("LVR start count:", assistant_text.count(LVR_START_TOKEN))
print("LVR token count:", assistant_text.count(LVR_TOKEN))
print("LVR end count:", assistant_text.count(LVR_END_TOKEN))

assert assistant_text.count(LVR_START_TOKEN) == 1
assert assistant_text.count(LVR_TOKEN) == 627
assert assistant_text.count(LVR_END_TOKEN) == 1

print("\nConversation conversion validation passed!")
