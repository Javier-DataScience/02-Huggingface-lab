# ---------------------------------------------------
# Pipeline Demo - Hugging Face
# ---------------------------------------------------
# This module demonstrates how to use Hugging Face
# pipelines for zero-code transformer inference.
# It loads a pretrained sentiment analysis model and
# runs predictions on sample text inputs.
# ---------------------------------------------------

from transformers import pipeline


def run_sentiment_pipeline():
    # Load pretrained sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis")

    # Sample inputs
    texts = [
        "I love building AI projects step by step.",
        "This is very confusing and frustrating."
    ]

    # Run inference
    results = classifier(texts)

    # Return results
    return list(zip(texts, results))


if __name__ == "__main__":
    outputs = run_sentiment_pipeline()
    for text, result in outputs:
        print("\nText:", text)
        print("Result:", result)