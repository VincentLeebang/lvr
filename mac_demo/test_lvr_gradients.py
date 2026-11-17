
import torch
import torch.nn.functional as F

print("LVR GRADIENT PROPAGATION TEST")

# 1. Create visual embeddings
image_embeds = torch.randn(
    100, 16, requires_grad=True
)

# 2. Select ROI visual embeddings
selected_indices = torch.tensor([10, 11, 12, 13])
selected_lvr_embeds = image_embeds[selected_indices]

# 3. Create input embeddings WITHOUT making them leaf tensors
inputs_embeds = torch.randn(1, 20, 16)

# 4. Insert selected visual embeddings
batch_indices = torch.tensor([0, 0, 0, 0])
seq_positions = torch.tensor([5, 6, 7, 8])

inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

# Retain gradients for this non-leaf tensor
inputs_embeds.retain_grad()

# Simulate a differentiable language-model operation
hidden_states = inputs_embeds * 1.5

# 6. Extract hidden states before LVR placeholders
seq_positions_start = seq_positions - 1

predicted = hidden_states[
    batch_indices, seq_positions_start
]

# 7. Calculate LVR loss
loss_lvr = F.mse_loss(
    predicted.float(),
    selected_lvr_embeds.float()
)

# 8. Backpropagation
loss_lvr.backward()

# 9. Display results
print("\n--- RESULTS ---")
print("LVR loss:", loss_lvr.item())

print(
    "Visual embedding gradient:",
    image_embeds.grad is not None
)

print(
    "Visual gradient non-zero:",
    image_embeds.grad.abs().sum().item() > 0
)

print(
    "Input embedding gradient:",
    inputs_embeds.grad is not None
)

print("\nGradient test completed.")
