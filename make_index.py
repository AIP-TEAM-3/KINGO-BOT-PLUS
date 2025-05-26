import os
import faiss
import pickle
import torch
import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

PKL_PATH = "./benchmark_chunks_split.pkl" 
INDEX_DIR = "./vector_indices"
os.makedirs(INDEX_DIR, exist_ok=True)

MODELS = {
    "FT_20": "expr1_ckpt/FT_epoch20",
    "FT_100": "expr1_ckpt/FT_epoch100",
    "BASE": "dragonkue/BGE-m3-ko",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load document chunks
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
chunks = data["texts"]

# Build FAISS index
def build_index_from_embeddings(embeddings: np.ndarray, dim: int):
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index

# Save index to disk
def save_index(index, name):
    path = os.path.join(INDEX_DIR, f"vector_{name.lower()}.index")
    faiss.write_index(index, path)
    print(f"Saved: {path}")

# Encode and index for each model
for tag, path in MODELS.items():
    print(f"\nBuilding index for: {tag}")

    model = SentenceTransformer(path, device=device)
    model.max_seq_length = 512

    vecs = []
    for i in tqdm.tqdm(range(0, len(chunks), 64), desc=f"Encode ({tag})"):
        batch = chunks[i:i+64]
        vecs.append(model.encode(
            batch,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ))

    embeddings = np.vstack(vecs)
    index = build_index_from_embeddings(embeddings, embeddings.shape[1])
    save_index(index, tag)
