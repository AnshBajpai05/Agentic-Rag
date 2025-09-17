# embedding.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Load SBERT model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
