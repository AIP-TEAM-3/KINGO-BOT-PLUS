import os, pickle, tqdm, pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
from collections import defaultdict

# 경로 설정
PKL_PATH = "./data/chunk/chunks_split.pkl"
CSV_PATH = "./data/csv/test.csv"

# 하이퍼파라미터
k1 = 1.5
b = 0.75

# 문서 및 메타 로딩
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
chunks, file_map = data["texts"], data["file_map"]
idx2fname = {v: k for k, v in file_map.items()}
docs = chunks

# 전처리 및 통계 수집
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
terms = vectorizer.get_feature_names_out()
doc_lengths = X.sum(axis=1).A1
avgdl = np.mean(doc_lengths)
df = np.asarray((X > 0).sum(axis=0)).ravel()
idf = {
    t: math.log((len(docs) - df[i] + 0.5) / (df[i] + 0.5) + 1)
    for i, t in enumerate(terms)
}

# 테스트 데이터 로딩
test_df = pd.read_csv(CSV_PATH)
test_df["Doc_index"] = test_df["Doc_index"].astype(str).str.replace(".txt", "", regex=False)

# 평가
hits, records = 0, []
vocab = vectorizer.vocabulary_

for row in tqdm.tqdm(test_df.itertuples(index=False), total=len(test_df), desc="BM25 Evaluate"):
    q, gold = row.Question, str(row.Doc_index)

    # 쿼리 토큰화
    q_tokens = vectorizer.build_tokenizer()(q.lower())
    q_tokens = [t for t in q_tokens if t in vocab]

    # BM25 점수 계산
    scores = []
    for i, doc in enumerate(docs):
        doc_len = doc_lengths[i]
        tf = X[i].toarray()[0]
        score = 0.0
        for t in q_tokens:
            term_id = vocab[t]
            f = tf[term_id]
            idf_val = idf[t]
            denom = f + k1 * (1 - b + b * doc_len / avgdl)
            score += idf_val * (f * (k1 + 1)) / denom if denom != 0 else 0.0
        scores.append((score, i))

    # Top 3 예측
    top3 = sorted(scores, key=lambda x: x[0], reverse=True)[:3]
    pred_chunks = [idx2fname[i] for _, i in top3]
    correct = any(p.split("_")[0] == gold for p in pred_chunks)
    hits += int(correct)

    if len(records) < 10:
        records.append({
            "Question": q, "Gold": gold,
            "BM25": ", ".join(pred_chunks),
            "BM25_C": "O" if correct else "X"
        })

# 결과 출력
print("\n=== Top 10 examples =============================")
for i, r in enumerate(records, 1):
    print(f"[{i}] Q: {r['Question'][:80]}...")
    print(f"    BM25  : {r['BM25']} ({r['BM25_C']})")
    print(f"    GOLD  : {r['Gold']}\n" + "-" * 100)

print("\nRecall@3")
total = len(test_df)
print(f"BM25  : {hits}/{total} = {hits/total:.2%}")
