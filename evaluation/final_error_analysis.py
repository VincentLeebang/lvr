
import csv
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_evaluation"
    / "advanced_analysis"
    / "manual_review.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_evaluation"
    / "final_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load manual review
with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

if not rows:
    raise ValueError("Manual review CSV is empty.")

required_columns = {
    "category",
    "human_correctness",
    "hallucination_check"
}

missing = required_columns - set(rows[0].keys())

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

total = len(rows)

# Overall statistics
correct = 0
partial = 0
incorrect = 0
hallucinations = 0

category_data = defaultdict(
    lambda: {
        "total": 0,
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
        "hallucinations": 0
    }
)

for row in rows:
    category = row["category"].strip()
    correctness = row["human_correctness"].strip().lower()
    hallucination = row["hallucination_check"].strip().lower()

    category_data[category]["total"] += 1

    if correctness == "correct":
        correct += 1
        category_data[category]["correct"] += 1

    elif correctness == "partially correct":
        partial += 1
        category_data[category]["partially_correct"] += 1

    elif correctness == "incorrect":
        incorrect += 1
        category_data[category]["incorrect"] += 1

    else:
        raise ValueError(
            f"Invalid correctness value for {row.get('image')}: "
            f"{row['human_correctness']}"
        )

    if hallucination == "yes":
        hallucinations += 1
        category_data[category]["hallucinations"] += 1
    elif hallucination != "no":
        raise ValueError(
            f"Invalid hallucination value for {row.get('image')}: "
            f"{row['hallucination_check']}"
        )

# Calculate percentages
def percentage(value, denominator):
    return round((value / denominator) * 100, 2) if denominator else 0

summary = {
    "total_images_reviewed": total,
    "correct_predictions": correct,
    "partially_correct_predictions": partial,
    "incorrect_predictions": incorrect,
    "correctness_percentages": {
        "correct": percentage(correct, total),
        "partially_correct": percentage(partial, total),
        "incorrect": percentage(incorrect, total)
    },
    "hallucination_flags": hallucinations,
    "hallucination_percentage": percentage(hallucinations, total),
    "category_analysis": {}
}

# Category-wise analysis
category_rows = []

for category, values in sorted(category_data.items()):
    count = values["total"]

    result = {
        **values,
        "correct_percentage": percentage(values["correct"], count),
        "partial_percentage": percentage(
            values["partially_correct"], count
        ),
        "incorrect_percentage": percentage(values["incorrect"], count),
        "hallucination_percentage": percentage(
            values["hallucinations"], count
        )
    }

    summary["category_analysis"][category] = result
    category_rows.append({"category": category, **result})

# Save timestamped outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

json_file = OUTPUT_DIR / f"final_analysis_{timestamp}.json"
csv_file = OUTPUT_DIR / f"category_analysis_{timestamp}.csv"

with open(json_file, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

with open(csv_file, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=category_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(category_rows)

# Display results
print("\nLATENT-SIGHT FINAL ERROR ANALYSIS")
print("=" * 45)

print("Total images reviewed:", total)
print("Correct:", correct)
print("Partially correct:", partial)
print("Incorrect:", incorrect)

print("\nCorrectness percentages:")
print("Correct:", summary["correctness_percentages"]["correct"], "%")
print("Partially correct:", summary["correctness_percentages"]["partially_correct"], "%")
print("Incorrect:", summary["correctness_percentages"]["incorrect"], "%")

print("\nHallucination flags:", hallucinations)
print("Hallucination percentage:", summary["hallucination_percentage"], "%")

print("\nCATEGORY-WISE ANALYSIS")
print("=" * 45)

for category, result in summary["category_analysis"].items():
    print(f"\n{category}")
    print("Images:", result["total"])
    print("Correct:", result["correct"])
    print("Partially correct:", result["partially_correct"])
    print("Incorrect:", result["incorrect"])
    print("Hallucination flags:", result["hallucinations"])

print("\nFiles saved successfully:")
print(json_file)
print(csv_file)
