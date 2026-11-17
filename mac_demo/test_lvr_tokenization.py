
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer

# Register LVR special tokens
special_tokens = [
    "<|lvr_start|>",
    "<|lvr|>",
    "<|lvr_latent_end|>",
    "<|lvr_end|>",
]

tokenizer.add_tokens(special_tokens, special_tokens=True)

# Get token IDs
lvr_start_id = tokenizer.convert_tokens_to_ids("<|lvr_start|>")
lvr_id = tokenizer.convert_tokens_to_ids("<|lvr|>")
lvr_latent_end_id = tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")
lvr_end_id = tokenizer.convert_tokens_to_ids("<|lvr_end|>")

print("\nRegistered token IDs:")
print("LVR start:", lvr_start_id)
print("LVR token:", lvr_id)
print("LVR latent end:", lvr_latent_end_id)
print("LVR end:", lvr_end_id)

# Construct a sequence for the 627 selected visual tokens
text = (
    "<|lvr_start|>"
    + "<|lvr|>" * 627
    + "<|lvr_end|>"
)

encoded = tokenizer(
    text,
    add_special_tokens=False,
    return_tensors="pt"
)

token_ids = encoded["input_ids"][0]

print("\nTokenization results:")
print("Total token IDs:", len(token_ids))
print("LVR token occurrences:", (token_ids == lvr_id).sum().item())
print("Start token occurrences:", (token_ids == lvr_start_id).sum().item())
print("End token occurrences:", (token_ids == lvr_end_id).sum().item())

# Validate
assert len(token_ids) == 629
assert (token_ids == lvr_id).sum().item() == 627
assert (token_ids == lvr_start_id).sum().item() == 1
assert (token_ids == lvr_end_id).sum().item() == 1

print("\nLVR tokenization validation passed!")
