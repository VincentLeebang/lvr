
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# LATENT-SIGHT: LVR LOSS EXPERIMENT
# ==========================================

print("=" * 60)
print("LATENT-SIGHT - LVR LOSS EXPERIMENT")
print("=" * 60)

# 1. Select device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("PyTorch Version:", torch.__version__)
print("Device:", DEVICE)


# 2. Loss function implementation
def set_lvr_loss_fct(loss_type):

    if loss_type == "mse":
        return nn.MSELoss()

    elif loss_type == "mae":
        return nn.L1Loss()

    elif loss_type == "cosine":

        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(
                x, y, dim=-1
            ).mean()

        return cosine_loss

    else:
        raise ValueError(
            f"Unsupported loss type: {loss_type}"
        )


# 3. Experimental configuration
LVR_LOSS_WEIGHT = 0.1
MODE_SWITCH_LOSS_WEIGHT = 0.1

MODE_SWITCH_ENABLED = False

# Reproducible test data
torch.manual_seed(42)

# 4. Create sample visual embeddings
visual_embeddings = torch.randn(
    4, 16, device=DEVICE
)

visual_embeddings = visual_embeddings.detach()

# 5. Create predicted hidden states
predicted_hidden_states = nn.Parameter(
    torch.randn(4, 16, device=DEVICE)
)

# 6. Create latent-end embeddings
predicted_latent_end = nn.Parameter(
    torch.randn(4, 16, device=DEVICE)
)

target_latent_end = torch.randn(
    4, 16, device=DEVICE
).detach()


# 7. Test all three loss functions automatically
loss_types = ["mse", "mae", "cosine"]

print("\nLOSS FUNCTION COMPARISON")
print("-" * 60)

for loss_type in loss_types:

    # Reset gradients before each experiment
    predicted_hidden_states.grad = None
    predicted_latent_end.grad = None

    # Select loss functions
    lvr_loss_function = set_lvr_loss_fct(
        loss_type
    )

    mode_switch_loss_function = set_lvr_loss_fct(
        loss_type
    )

    # Calculate LVR loss
    loss_lvr = lvr_loss_function(
        predicted_hidden_states,
        visual_embeddings
    )

    # Calculate mode-switch loss
    loss_mode_switch = mode_switch_loss_function(
        predicted_latent_end,
        target_latent_end
    )

    # Apply original repository's default LVR weight
    weighted_lvr_loss = (
        LVR_LOSS_WEIGHT * loss_lvr
    )

    # Mode-switch is disabled by default
    if MODE_SWITCH_ENABLED:

        weighted_mode_switch_loss = (
            MODE_SWITCH_LOSS_WEIGHT
            * loss_mode_switch
        )

    else:

        weighted_mode_switch_loss = torch.zeros(
            (), device=DEVICE
        )

    # Experimental objective
    total_loss = (
        weighted_lvr_loss
        + weighted_mode_switch_loss
    )

    # Backpropagation
    total_loss.backward()

    # Display results
    print(f"\nLoss Type: {loss_type.upper()}")
    print("LVR Loss:", round(loss_lvr.item(), 6))
    print(
        "Mode Switch Loss:",
        round(loss_mode_switch.item(), 6)
    )
    print(
        "Weighted LVR Loss:",
        round(weighted_lvr_loss.item(), 6)
    )
    print(
        "Total Experimental Loss:",
        round(total_loss.item(), 6)
    )
    print(
        "Hidden State Gradient Available:",
        predicted_hidden_states.grad is not None
    )
    print(
        "Latent-End Gradient Available:",
        predicted_latent_end.grad is not None
    )

    if DEVICE.type == "mps":
        torch.mps.synchronize()


# 8. Final result
print("\n" + "=" * 60)
print("ALL THREE LOSS EXPERIMENTS COMPLETED!")
print("=" * 60)

print("MSE: Tested")
print("MAE: Tested")
print("Cosine: Tested")
print("Gradient calculation: Tested")
print("Mode-switch enabled:", MODE_SWITCH_ENABLED)
print("Execution device:", DEVICE)
