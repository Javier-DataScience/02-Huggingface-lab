# ---------------------------------------------------
# Embedding Service
# ---------------------------------------------------
# Responsible ONLY for sentence embeddings and similarity
# ---------------------------------------------------

from sentence_transformers import SentenceTransformer, util


class EmbeddingService:
    def __init__(self):
        # Load embedding model once
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, sentences):
        """
        Converts sentences into embeddings (vectors)
        """
        return self.model.encode(sentences)

    def similarity(self, emb1, emb2):
        """
        Computes cosine similarity between two embeddings
        """
        return util.cos_sim(emb1, emb2).item()