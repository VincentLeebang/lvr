
import torch

print("\nLATENT-SIGHT - FORWARD MAPPING GRADIENT TEST")

# Synthetic visual embeddings with gradients enabled
image_embeds = torch.randn(
    20, 32, requires_grad=True
)

# Synthetic language embeddings
inputs_embeds = torch.randn(2, 10, 32)

# Example ROI indices
roi_indices = torch.tensor([2, 4, 7, 12, 15])

# Example batch and sequence positions
batch_indices = torch.tensor([0, 0, 0, 1, 1])
seq_positions = torch.tensor([2, 4, 6, 3, 5])

# Select ROI features
selected_lvr_embeds = image_embeds[roi_indices]

# Insert selected features into LVR placeholders
inputs_embeds[
    batch_indices, seq_positions
] = selected_lvr_embeds

# Retain gradient on the modified embedding tensor
inputs_embeds.retain_grad()

# Simulate language-model hidden states
hidden_states = inputs_embeds * 1.5

# Calculate synthetic LVR loss
predicted = hidden_states[
    batch_indices, seq_positions
]

loss = torch.nn.functional.mse_loss(
    predicted,
    selected_lvr_embeds
)

print("LVR loss:", loss.item())

# Backpropagation
loss.backward()

print(
    "Visual embedding gradient exists:",
    image_embeds.grad is not None
)

print(
    "Visual gradient is non-zero:",
    image_embeds.grad is not None
    and image_embeds.grad.abs().sum().item() > 0
)

print(
    "Input embedding gradient exists:",
    inputs_embeds.grad is not None
)

assert image_embeds.grad is not None
assert image_embeds.grad.abs().sum().item() > 0
assert inputs_embeds.grad is not None

print("\nFORWARD MAPPING GRADIENT TEST PASSED!")
