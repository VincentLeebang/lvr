import json
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "outputs" / "dataset_evaluation"
OUTPUT_DIR = RESULTS_DIR / "category_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    files = sorted(
        RESULTS_DIR.glob("results_*.json"),
        key=lambda p: p.stat().st_mtime
    )

    if not files:
        raise FileNotFoundError("No dataset evaluation JSON found.")

    result_file = files[-1]

    print("Reading:", result_file.name)

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = defaultdict(list)

    for record in data["results"]:
        categories[record["category"]].append(record)

    summary = []

    print("\nCATEGORY-WISE ANALYSIS")
    print("=" * 60)

    for category, items in sorted(categories.items()):

        avg_f1 = sum(
            item["lexical_f1"] for item in items
        ) / len(items)

        avg_time = sum(
            item["inference_time_seconds"] for item in items
        ) / len(items)

        row = {
            "category": category,
            "number_of_samples": len(items),
            "average_lexical_f1": round(avg_f1, 4),
            "average_inference_time_seconds": round(avg_time, 3)
        }

        summary.append(row)

        print("\nCategory:", category)
        print("Samples:", len(items))
        print("Average lexical F1:", round(avg_f1, 4))
        print("Average inference time:", round(avg_time, 3), "seconds")

    json_path = OUTPUT_DIR / "category_summary.json"
    csv_path = OUTPUT_DIR / "category_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\nCATEGORY ANALYSIS COMPLETED")
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
