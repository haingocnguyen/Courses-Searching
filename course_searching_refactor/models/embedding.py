import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CACHE_TTL, MAX_CACHE_ENTRIES

class SBERTEmbeddingModel:
    """SBERT embedding model with caching for performance"""
    
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model.to('cpu')
        self.model_name = model_name
        
    @st.cache_data(ttl=CACHE_TTL, max_entries=MAX_CACHE_ENTRIES)
    def get_embedding_cached(_self, text_hash: str, text: str) -> np.ndarray:
        """Cache embeddings with hash key for performance"""
        emb = _self.model.encode(text, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
        return emb
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching support"""
        text_hash = str(hash(text))
        return self.get_embedding_cached(text_hash, text)

@st.cache_resource
def get_embedding_model():
    """Cached embedding model instance"""
    return SBERTEmbeddingModel()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if a is None or b is None: 
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na and nb else 0.0