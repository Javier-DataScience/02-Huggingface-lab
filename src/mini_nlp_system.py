# ---------------------------------------------------
# Mini NLP System (Orchestrator)
# ---------------------------------------------------
# This is the "brain" of the application.
# It coordinates:
# - SentimentService
# - EmbeddingService
# ---------------------------------------------------

from src.sentiment_service import SentimentService
from src.embedding_service import EmbeddingService


class MiniNLPSystem:
    def __init__(self):
        self.sentiment_service = SentimentService()
        self.embedding_service = EmbeddingService()

    def analyze(self, sentences):
        """
        Full AI pipeline:
        1. sentiment analysis
        2. embeddings
        3. similarity computation
        """

        # ---------------------------
        # SENTIMENT ANALYSIS
        # ---------------------------
        sentiment_results = self.sentiment_service.analyze(sentences)

        # ---------------------------
        # EMBEDDINGS
        # ---------------------------
        embeddings = self.embedding_service.encode(sentences)

        # ---------------------------
        # SIMILARITY
        # ---------------------------
        sim_1_2 = self.embedding_service.similarity(embeddings[0], embeddings[1])
        sim_1_3 = self.embedding_service.similarity(embeddings[0], embeddings[2])

        # ---------------------------
        # STRUCTURED OUTPUT
        # ---------------------------
        analysis = []

        for i, sentence in enumerate(sentences):
            analysis.append({
                "sentence": sentence,
                "sentiment": sentiment_results[i]
            })

        return {
            "analysis": analysis,
            "similarity_1_2": sim_1_2,
            "similarity_1_3": sim_1_3
        }


# Optional test run
if __name__ == "__main__":

    system = MiniNLPSystem()

    sentences = [
        "I love building AI systems step by step",
        "This project is very interesting and powerful",
        "I feel tired and frustrated with debugging"
    ]

    result = system.analyze(sentences)

    print("\n=== MINI NLP SYSTEM (ARCHITECTURE VERSION) ===\n")

    for item in result["analysis"]:
        print("Sentence:", item["sentence"])
        print("Sentiment:", item["sentiment"])
        print()

    print("Similarity (1 vs 2):", result["similarity_1_2"])
    print("Similarity (1 vs 3):", result["similarity_1_3"])