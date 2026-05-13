# ---------------------------------------------------
# Tokenizer Demo - Hugging Face
# ---------------------------------------------------
# This module demonstrates how tokenizers convert
# raw text into token IDs that models can understand.
# ---------------------------------------------------

from transformers import AutoTokenizer


def run_tokenizer_demo():
    # Load a pretrained tokenizer from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "I love learning Hugging Face Transformers"

    # Convert text into tokens
    tokens = tokenizer(text)

    # Convert text into token IDs
    token_ids = tokenizer.encode(text)

    # Decode back to text
    decoded_text = tokenizer.decode(token_ids)

    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "decoded": decoded_text
    }


if __name__ == "__main__":
    result = run_tokenizer_demo()

    print("\n=== TOKENIZER DEMO ===")
    print("\nOriginal Text:", result["text"])
    print("\nToken IDs:", result["token_ids"])
    print("\nDecoded Text:", result["decoded"])