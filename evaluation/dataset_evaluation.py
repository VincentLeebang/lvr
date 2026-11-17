import sys
import json
import csv
import time
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

from PIL import Image

# Project directories
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "dataset"
IMAGE_DIR = DATASET / "images"
ANNOTATIONS = DATASET / "annotations.json"
OUTPUT_DIR = ROOT / "outputs" / "dataset_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import existing SmolVLM service
sys.path.insert(0, str(ROOT / "website"))
from model_service import analyze_image


def calculate_f1(reference, prediction):
    ref = Counter(re.findall(r"\b[a-z0-9]+\b", reference.lower()))
    pred = Counter(re.findall(r"\b[a-z0-9]+\b", prediction.lower()))

    if not ref or not pred:
        return 0.0

    common = sum((ref & pred).values())
    precision = common / sum(pred.values())
    recall = common / sum(ref.values())

    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 4)


def main():
    print("=" * 55)
    print("LATENT-SIGHT DATASET EVALUATION")
    print("Baseline: SmolVLM-256M-Instruct")
    print("=" * 55)

    with open(ANNOTATIONS, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = []

    for record in records:
        image_path = IMAGE_DIR / record["image"]

        if not image_path.exists():
            print("Missing image:", image_path)
            continue

        print("\nEvaluating:", record["image"])
        print("Question:", record["question"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        start = time.perf_counter()
        prediction = analyze_image(image, record["question"])
        elapsed = time.perf_counter() - start

        if not isinstance(prediction, str):
            prediction = str(prediction)

        f1 = calculate_f1(record["ground_truth"], prediction)

        result = {
            "id": record["id"],
            "image": record["image"],
            "category": record["category"],
            "question": record["question"],
            "ground_truth": record["ground_truth"],
            "prediction": prediction,
            "inference_time_seconds": round(elapsed, 3),
            "lexical_f1": f1
        }

        results.append(result)

        print("Ground truth:", record["ground_truth"])
        print("Model answer:", prediction)
        print("Inference time:", round(elapsed, 3), "seconds")
        print("Lexical F1:", f1)

    if not results:
        print("\nNo images were evaluated.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_file = OUTPUT_DIR / f"results_{timestamp}.json"
    csv_file = OUTPUT_DIR / f"results_{timestamp}.csv"

    summary = {
        "model": "SmolVLM-256M-Instruct",
        "number_of_samples": len(results),
        "average_inference_time_seconds": round(
            sum(r["inference_time_seconds"] for r in results)
            / len(results), 3
        ),
        "average_lexical_f1": round(
            sum(r["lexical_f1"] for r in results)
            / len(results), 4
        ),
        "results": results
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED")
    print("Images evaluated:", len(results))
    print("Average inference time:",
          summary["average_inference_time_seconds"], "seconds")
    print("Average lexical F1:", summary["average_lexical_f1"])
    print("JSON:", json_file)
    print("CSV:", csv_file)


if __name__ == "__main__":
    main()
