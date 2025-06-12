import os
import pickle
import tqdm
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# 경로 설정
PKL_PATH = "./data/chunk/chunks_split.pkl"
INDEX_DIR = "./vector_indices_tfidf"
os.makedirs(INDEX_DIR, exist_ok=True)

# 모델(전처리 config) 태그
TAGS = ["IFIDF_BASE"]

# 문서 로드
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
chunks = data["texts"]

# 벡터화 및 저장
def build_index_from_embeddings(embeddings: np.ndarray, dim: int):
    faiss.normalize_L2(embeddings)  # cosine 유사도 기반
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index

def save_index(index, name):
    path = os.path.join(INDEX_DIR, f"vector_{name.lower()}.index")
    faiss.write_index(index, path)
    print(f"Saved: {path}")

def save_vectorizer(vectorizer, name):
    path = os.path.join(INDEX_DIR, f"vectorizer_{name.lower()}.pkl")
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved: {path}")

# 각 태그에 대해 처리
for tag in TAGS:
    print(f"\n🛠️  Building TF-IDF index for: {tag}")
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    dense_embeddings = tfidf_matrix.astype(np.float32).toarray()

    index = build_index_from_embeddings(dense_embeddings, dense_embeddings.shape[1])
    save_index(index, tag)
    save_vectorizer(vectorizer, tag)
