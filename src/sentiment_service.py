# ---------------------------------------------------
# Sentiment Service
# ---------------------------------------------------
# Responsible ONLY for sentiment analysis logic
# ---------------------------------------------------

from transformers import pipeline


class SentimentService:
    def __init__(self):
        # Load model once (important for efficiency)
        self.model = pipeline("sentiment-analysis")

    def analyze(self, texts):
        """
        Input: list of strings
        Output: sentiment predictions
        """
        return self.model(texts)