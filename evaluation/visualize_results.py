
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load category results
file_path = Path("outputs/dataset_evaluation/category_analysis/category_summary.csv")
df = pd.read_csv(file_path)

# Create graph folder
output_dir = Path("outputs/dataset_evaluation/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

# Create two graphs side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graph 1: Lexical F1
bars1 = axes[0].bar(
    df["category"],
    df["average_lexical_f1"]
)

axes[0].set_title("Category-wise Lexical F1")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Average Lexical F1")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=25)

for bar in bars1:
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.015,
        f"{height:.3f}",
        ha="center"
    )

# Graph 2: Inference Time
bars2 = axes[1].bar(
    df["category"],
    df["average_inference_time_seconds"]
)

axes[1].set_title("Category-wise Average Inference Time")
axes[1].set_xlabel("Category")
axes[1].set_ylabel("Inference Time (seconds)")
axes[1].tick_params(axis="x", rotation=25)

for bar in bars2:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.05,
        f"{height:.3f}s",
        ha="center"
    )

plt.tight_layout()

# Save both graphs
fig.savefig(output_dir / "combined_results.png", dpi=300)

plt.show()

print("Both graphs saved successfully!")
