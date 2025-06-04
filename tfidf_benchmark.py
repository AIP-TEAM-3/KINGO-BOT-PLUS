import os, pickle, tqdm, pandas as pd, numpy as np, faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# 경로 설정
PKL_PATH = "./data/chunk/chunks_split.pkl"
CSV_PATH = "./data/csv/test.csv"
INDEX_PATH = "./vector_indices_tfidf/vector_tfidf_base.index"
VECTORIZER_PATH = "./vector_indices_tfidf/vectorizer_tfidf_base.pkl"

# 로딩
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
chunks, file_map = data["texts"], data["file_map"]
idx2fname = {v: k for k, v in file_map.items()}

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

index = faiss.read_index(INDEX_PATH)

# 테스트 데이터
df = pd.read_csv(CSV_PATH)
df["Doc_index"] = df["Doc_index"].astype(str).str.replace(".txt", "", regex=False)
test_df = df

# 평가
hits, records = 0, []
for row in tqdm.tqdm(test_df.itertuples(index=False), total=len(test_df), desc="TF-IDF Evaluate"):
    q, gold = row.Question, str(row.Doc_index)

    # 쿼리 임베딩
    q_vec = vectorizer.transform([q]).toarray().astype("float32")
    faiss.normalize_L2(q_vec)

    _, I = index.search(q_vec, 3)
    pred_chunks = [idx2fname[int(i)] for i in I[0]]

    hits += int(any(p.split('_')[0] == gold for p in pred_chunks))

    if len(records) < 10:
        records.append({
            "Question": q, "Gold": gold,
            "TF-IDF": ", ".join(pred_chunks),
            "TF-IDF_C": "O" if any(p.split('_')[0] == gold for p in pred_chunks) else "X"
        })

# 출력
print("\n=== Top 10 examples =============================")
for i, r in enumerate(records, 1):
    print(f"[{i}] Q: {r['Question'][:80]}...")
    print(f"    TF-IDF  : {r['TF-IDF']} ({r['TF-IDF_C']})")
    print(f"    GOLD  : {r['Gold']}\n" + "-" * 100)

print("\nRecall@3")
total = len(test_df)
print(f"TF-IDF  : {hits}/{total} = {hits/total:.2%}")
