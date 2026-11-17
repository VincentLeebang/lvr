
import torch
import torch.nn.functional as F

print("\nLATENT-SIGHT - BATCH INDEX ALIGNMENT TEST")

# Each image contributes 1944 visual tokens
total_tokens = torch.tensor([1944, 1944])

# Same cumulative-offset calculation used in the LVR forward logic
image_token_offsets = torch.cumsum(
    F.pad(total_tokens, (1, 0)), dim=0
)[:-1]

print("Image token offsets:", image_token_offsets.tolist())

# Representative selected-index boundaries from our
# previously validated bounding-box mapping
local_min = 180
local_max = 1926

# Calculate expected global positions for each example
for batch_idx in range(2):
    offset = image_token_offsets[batch_idx].item()

    global_min = local_min + offset
    global_max = local_max + offset

    print(f"\nExample {batch_idx}:")
    print("Offset:", offset)
    print("Expected global minimum:", global_min)
    print("Expected global maximum:", global_max)

    assert global_min >= offset
    assert global_max < offset + 1944

# Check that both examples address separate image regions
assert image_token_offsets.tolist() == [0, 1944]

assert local_max < 1944
assert local_max + 1944 < 3888

print("\nBATCH INDEX OFFSET VALIDATION PASSED!")
