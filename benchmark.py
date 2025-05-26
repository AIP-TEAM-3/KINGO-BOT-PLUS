import faiss, pickle, torch, tqdm, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
import os

# Paths
PKL_PATH  = "./chunks_split.pkl"
CSV_PATH  = "./data/csv/test.csv"
INDEX_DIR = "./vector_indices"

# Model config
MODELS = {
    "FT_20": "expr1_ckpt/FT_epoch20",
    "FT_100": "expr1_ckpt/FT_epoch100",
    "BASE": "dragonkue/BGE-m3-ko",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load chunk data
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
chunks, file_map = data["texts"], data["file_map"]
idx2fname = {v: k for k, v in file_map.items()}

# Load test data
df = pd.read_csv(CSV_PATH)
df["Doc_index"] = df["Doc_index"].astype(str).str.replace(".txt", "", regex=False)
test_df = df

# Load indices and models
indices, models = {}, {}
for tag in MODELS:
    path = os.path.join(INDEX_DIR, f"vector_{tag.lower()}.index")
    print(f"Loading index for {tag}: {path}")
    indices[tag] = faiss.read_index(path)

    model = SentenceTransformer(MODELS[tag], device=device)
    model.max_seq_length = 512
    models[tag] = model

print("All indices loaded.\n")

# Evaluate Recall@3
hits, records = {t: 0 for t in MODELS}, []
for row in tqdm.tqdm(test_df.itertuples(index=False), total=len(test_df), desc="Evaluate"):
    q, gold = row.Question, str(row.Doc_index)
    preds = {}

    for tag in MODELS:
        q_emb = models[tag].encode([q], normalize_embeddings=True,
                                   convert_to_numpy=True).astype("float32")
        _, I = indices[tag].search(q_emb, 3)
        pred_chunks = [idx2fname[int(i)] for i in I[0]]

        preds[tag] = pred_chunks
        hits[tag] += int(any(p.split('_')[0] == gold for p in pred_chunks))

    if len(records) < 10:
        records.append({
            "Question": q, "Gold": gold,
            **{t: ", ".join(preds[t]) for t in MODELS},
            **{f"{t}_C": "O" if any(p.split('_')[0] == gold for p in preds[t]) else "X"
              for t in MODELS}
        })

# Print top-10 examples
print("\n=== Top 10 examples =============================")
for i, r in enumerate(records, 1):
    print(f"[{i}] Q: {r['Question'][:80]}...")
    for tag in MODELS:
        print(f"    {tag:<6}: {r[tag]} ({r[f'{tag}_C']})")
    print(f"    GOLD : {r['Gold']}\n" + "-" * 100)

# Final recall@3 summary
print("\nRecall@3")
total = len(test_df)
for tag in MODELS:
    print(f"{tag:6}: {hits[tag]}/{total} = {hits[tag]/total:.2%}")
