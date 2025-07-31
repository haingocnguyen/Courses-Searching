from sentence_transformers import SentenceTransformer
import numpy as np

class SBERTEmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)