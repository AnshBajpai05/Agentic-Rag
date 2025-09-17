import os
import json
import faiss
import numpy as np
from PyPDF2 import PdfReader
from embedding import embed_text  # your embedding function

INDEX_FILE = "faiss_index.bin"
META_FILE = "metadata.json"
DIM = 384  # embedding dimension of your embeddings

# Load or create FAISS index
def load_memory(dim=DIM):
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    else:
        index = faiss.IndexFlatL2(dim)
        metadata = []
        return index, metadata

# Save index + metadata
def save_memory(index, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

# Read PDF text
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Chunk PDF text
def chunk_pdf(file_path, max_tokens=1000):
    text = read_pdf(file_path)
    chunks = []
    for i in range(0, len(text), max_tokens):
        chunks.append({"text": text[i:i+max_tokens], "source": file_path})
    return chunks

# Store PDF chunks in FAISS
def store_pdf(pdf_path):
    global index, metadata
    chunks = chunk_pdf(pdf_path)
    for chunk in chunks:
        vec = embed_text(chunk["text"])
        vec = np.array([vec], dtype=np.float32)
        index.add(vec)
        metadata.append({"pdf": pdf_path, "text": chunk["text"]})
    save_memory(index, metadata)


# Retrieve relevant chunks for a PDF
def retrieve_relevant_chunks(pdf_path, top_k=50):
    results = [m["text"] for m in metadata if m["pdf"] == pdf_path]
    return results[:top_k]

# Initialize memory
index, metadata = load_memory()
