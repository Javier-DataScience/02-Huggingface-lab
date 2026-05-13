# ---------------------------------------------------
# Embedding Demo - Sentence Transformers
# ---------------------------------------------------
# This module demonstrates how sentences are converted
# into embeddings (vector representations of meaning).
# ---------------------------------------------------

from sentence_transformers import SentenceTransformer, util


def run_embedding_demo():
    # Load pretrained embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "I love artificial intelligence",
        "I enjoy studying machine learning",
        "The weather is very cold today"
    ]

    # Generate embeddings
    embeddings = model.encode(sentences)

    # Compute similarity between first sentence and others
    similarity_1_2 = util.cos_sim(embeddings[0], embeddings[1])
    similarity_1_3 = util.cos_sim(embeddings[0], embeddings[2])

    return {
        "sentences": sentences,
        "similarity_1_2": similarity_1_2.item(),
        "similarity_1_3": similarity_1_3.item()
    }


if __name__ == "__main__":
    result = run_embedding_demo()

    print("\n=== EMBEDDING DEMO ===")

    for s in result["sentences"]:
        print("\nSentence:", s)

    print("\nSimilarity (1 vs 2):", result["similarity_1_2"])
    print("Similarity (1 vs 3):", result["similarity_1_3"])