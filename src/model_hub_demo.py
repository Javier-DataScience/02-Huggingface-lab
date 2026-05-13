# ---------------------------------------------------
# Model Hub Demo - Hugging Face (Fixed Version)
# ---------------------------------------------------
# Demonstrates:
# - Model Hub usage
# - Model comparison
# - Structured inference output
# ---------------------------------------------------

from transformers import pipeline


def run_model_hub_demo():

    # -----------------------------
    # MODELS FROM HUGGING FACE HUB
    # -----------------------------
    models = {
        "default": pipeline("sentiment-analysis"),
        "distilbert": pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    }

    text = "I really enjoy building AI systems and learning new models"

    results = {}

    # -----------------------------
    # RUN INFERENCE FOR EACH MODEL
    # -----------------------------
    for name, model in models.items():
        results[name] = model(text)

    return {
        "text": text,
        "results": results
    }


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":

    result = run_model_hub_demo()

    print("\n=== MODEL HUB DEMO ===")

    print("\nText:", result["text"])

    print("\n=== MODEL RESULTS ===")

    for model_name, output in result["results"].items():
        print(f"\nModel: {model_name}")
        print("Output:", output)