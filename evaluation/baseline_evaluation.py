
import sys
import csv
import json
import time
import resource

from pathlib import Path
from datetime import datetime

import torch
from PIL import Image


# ==========================================
# LATENT-SIGHT BASELINE EVALUATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEBSITE_DIR = PROJECT_DIR / "website"

sys.path.insert(0, str(WEBSITE_DIR))

from model_service import analyze_image


# ==========================================
# TEST QUESTIONS
# ==========================================

TEST_QUESTIONS = [
    {
        "category": "Object Recognition",
        "question": "Identify all the main objects visible in this image."
    },
    {
        "category": "Spatial Reasoning",
        "question": "Describe the positions of the main objects in this image."
    },
    {
        "category": "Scene Understanding",
        "question": "Describe what is happening in this image."
    },
    {
        "category": "Visual Counting",
        "question": "How many distinct animals or people are visible? Identify them."
    },
    {
        "category": "Appearance Recognition",
        "question": "Describe the colors and appearance of the main objects."
    }
]


# ==========================================
# MEMORY MEASUREMENT
# ==========================================

def get_memory_usage_mb():

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 2)

    return round(usage / 1024, 2)


def get_mps_memory_mb():

    if torch.backends.mps.is_available():
        try:
            return round(
                torch.mps.current_allocated_memory() / (1024 * 1024),
                2
            )
        except Exception:
            return None

    return None


# ==========================================
# EVALUATE ONE QUESTION
# ==========================================

def evaluate_question(image, category, question, index):

    print("\n" + "=" * 60)
    print(f"TEST {index}: {category}")
    print("=" * 60)

    print("Question:", question)

    start_time = time.perf_counter()

    answer = analyze_image(image, question)

    inference_time = round(
        time.perf_counter() - start_time, 3
    )

    print("\nAI ANSWER:")
    print(answer)

    print("\nInference time:", inference_time, "seconds")

    return {
        "test_number": index,
        "category": category,
        "question": question,
        "answer": answer,
        "inference_time_seconds": inference_time,
        "process_peak_memory_mb": get_memory_usage_mb(),
        "mps_memory_mb": get_mps_memory_mb(),
        "manual_review": ""
    }


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(results, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"baseline_results_{timestamp}.json"
    csv_path = output_dir / f"baseline_results_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved:")
    print("JSON:", json_path)
    print("CSV:", csv_path)


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\nLATENT-SIGHT BASELINE EVALUATION")
    print("=" * 60)

    image_path = (
        PROJECT_DIR / "mac_demo" / "images" / "test.jpg"
    )

    output_dir = (
        PROJECT_DIR / "outputs" / "baseline_evaluation"
    )

    print("Model: SmolVLM-256M-Instruct")
    print(
        "Device:",
        "MPS" if torch.backends.mps.is_available() else "CPU"
    )

    print("Image:", image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    print("Image loaded:", image.size)

    results = []

    for index, item in enumerate(TEST_QUESTIONS, start=1):

        result = evaluate_question(
            image,
            item["category"],
            item["question"],
            index
        )

        results.append(result)

    average_time = sum(
        item["inference_time_seconds"] for item in results
    ) / len(results)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("Total questions:", len(results))
    print("Average inference time:", round(average_time, 3), "seconds")
    print("Peak process memory:", get_memory_usage_mb(), "MB")

    save_results(results, output_dir)

    print("\nBaseline evaluation completed successfully!")


if __name__ == "__main__":
    main()
