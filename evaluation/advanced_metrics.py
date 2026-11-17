
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Project directories
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "outputs" / "dataset_evaluation"
OUTPUT_DIR = INPUT_DIR / "advanced_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number words for counting evaluation
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12
}


def extract_count(text):
    """Extract the first number from a counting answer."""
    text = text.lower()

    # Check digits first
    digit_match = re.search(r"\b\d+\b", text)
    if digit_match:
        return int(digit_match.group())

    # Check number words
    words = re.findall(r"\b[a-z]+\b", text)

    for word in words:
        if word in NUMBER_WORDS:
            return NUMBER_WORDS[word]

    return None


def normalize_text(text):
    """Normalize text for exact-match comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():

    print("=" * 60)
    print("LATENT-SIGHT ADVANCED EVALUATION")
    print("=" * 60)

    # Find previous evaluation files
    files = list(INPUT_DIR.glob("results_*.json"))

    if not files:
        print("No previous evaluation JSON found.")
        return

    # Select latest evaluation file
    latest_file = max(files, key=lambda f: f.stat().st_mtime)

    print("Reading:", latest_file.name)

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    advanced_results = []

    for record in results:

        reference = record["ground_truth"]
        prediction = record["prediction"]
        question = record["question"]

        # Exact match
        exact_match = (
            normalize_text(reference) ==
            normalize_text(prediction)
        )

        # Counting evaluation
        is_counting = question.lower().strip().startswith("how many")

        reference_count = None
        prediction_count = None
        count_correct = None

        if is_counting:
            reference_count = extract_count(reference)
            prediction_count = extract_count(prediction)

            if reference_count is not None and prediction_count is not None:
                count_correct = (
                    reference_count == prediction_count
                )

        # Flag records for manual inspection
        review_required = (
            record["lexical_f1"] < 0.5
            or (is_counting and count_correct is not True)
        )

        result = dict(record)

        result.update({
            "exact_match": exact_match,
            "is_counting_question": is_counting,
            "reference_count": reference_count,
            "prediction_count": prediction_count,
            "count_correct": count_correct,
            "manual_review_required": review_required,
            "human_correctness": "Not reviewed",
            "hallucination_check": "Not reviewed"
        })

        advanced_results.append(result)

    # Calculate overall metrics
    total = len(advanced_results)

    exact_matches = sum(
        r["exact_match"] for r in advanced_results
    )

    counting_records = [
        r for r in advanced_results
        if r["is_counting_question"]
        and r["count_correct"] is not None
    ]

    correct_counts = sum(
        r["count_correct"] for r in counting_records
    )

    average_f1 = sum(
        r["lexical_f1"] for r in advanced_results
    ) / total

    average_time = sum(
        r["inference_time_seconds"] for r in advanced_results
    ) / total

    summary = {
        "model": data.get("model", "SmolVLM-256M-Instruct"),
        "number_of_samples": total,
        "average_lexical_f1": round(average_f1, 4),
        "exact_match_accuracy": round(exact_matches / total, 4),
        "counting_accuracy": (
            round(correct_counts / len(counting_records), 4)
            if counting_records else None
        ),
        "counting_questions_evaluated": len(counting_records),
        "average_inference_time_seconds": round(average_time, 3),
        "manual_review_required": sum(
            r["manual_review_required"]
            for r in advanced_results
        ),
        "results": advanced_results
    }

    # Category-wise analysis
    categories = defaultdict(list)

    for record in advanced_results:
        categories[record["category"]].append(record)

    category_summary = []

    for category, records in categories.items():

        count_items = [
            r for r in records
            if r["is_counting_question"]
            and r["count_correct"] is not None
        ]

        category_summary.append({
            "category": category,
            "number_of_samples": len(records),
            "average_lexical_f1": round(
                sum(r["lexical_f1"] for r in records)
                / len(records), 4
            ),
            "exact_match_accuracy": round(
                sum(r["exact_match"] for r in records)
                / len(records), 4
            ),
            "counting_accuracy": (
                round(
                    sum(r["count_correct"] for r in count_items)
                    / len(count_items), 4
                )
                if count_items else "N/A"
            ),
            "average_inference_time_seconds": round(
                sum(r["inference_time_seconds"] for r in records)
                / len(records), 3
            )
        })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIR / f"advanced_results_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"advanced_results_{timestamp}.csv"
    category_path = OUTPUT_DIR / f"advanced_category_summary_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(advanced_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(advanced_results)

    with open(category_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(category_summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(category_summary)

    # Display summary
    print("\nADVANCED EVALUATION SUMMARY")
    print("-" * 45)
    print("Images evaluated:", total)
    print("Average Lexical F1:", round(average_f1, 4))
    print("Exact Match Accuracy:", summary["exact_match_accuracy"])
    print("Counting Accuracy:", summary["counting_accuracy"])
    print("Average Inference Time:", round(average_time, 3), "seconds")
    print("Records requiring manual review:",
          summary["manual_review_required"])

    print("\nCATEGORY-WISE RESULTS")

    for item in category_summary:
        print(
            item["category"],
            "| F1:", item["average_lexical_f1"],
            "| Exact Match:", item["exact_match_accuracy"]
        )

    print("\nFiles saved:")
    print(json_path)
    print(csv_path)
    print(category_path)

    print("\nAdvanced evaluation completed!")


if __name__ == "__main__":
    main()
