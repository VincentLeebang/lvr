
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "outputs" / "dataset_evaluation"
    / "advanced_analysis"
    / "advanced_results_20261117_162629.json"
)

OUTPUT = (
    ROOT / "outputs" / "dataset_evaluation"
    / "advanced_analysis"
    / "manual_review.csv"
)

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

fields = [
    "id",
    "image",
    "category",
    "question",
    "ground_truth",
    "prediction",
    "lexical_f1",
    "human_correctness",
    "hallucination_check",
    "review_notes"
]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for item in data["results"]:
        writer.writerow({
            **{key: item.get(key, "") for key in fields},
            "human_correctness": "Not reviewed",
            "hallucination_check": "Not reviewed",
            "review_notes": ""
        })

print("Manual review sheet created:")
print(OUTPUT)
