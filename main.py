# ---------------------------------------------------
# Main Entry Point - Hugging Face Lab Project
# ---------------------------------------------------
# This script runs the full pipeline demo from src/
# It acts as the central execution point of the project.
# ---------------------------------------------------

from src.pipeline_demo import run_sentiment_pipeline


def main():
    print("\n=== HUGGING FACE PIPELINE DEMO ===")

    results = run_sentiment_pipeline()

    print("\n=== RESULTS ===")
    for text, result in results:
        print("\nText:", text)
        print("Prediction:", result)


if __name__ == "__main__":
    main()