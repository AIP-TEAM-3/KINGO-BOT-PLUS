import os, pickle, tqdm
from transformers import AutoTokenizer

SRC_PKL = "./data/chunk/chunks.pkl"
DST_PKL = "./data/chunk/chunks_split.pkl"
MODEL   = "dragonkue/BGE-m3-ko"  
MAX_LEN = 512
STRIDE  = 128                    

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

def split_slide(txt, max_len=512, stride=128):
    ids = tok.encode(txt, add_special_tokens=False)
    if len(ids) <= max_len:
        return [txt]
    out = []
    for i in range(0, len(ids), max_len - stride):
        chunk_ids = ids[i : i + max_len]
        out.append(tok.decode(chunk_ids, skip_special_tokens=True))
    return out

with open(SRC_PKL, "rb") as f:
    src = pickle.load(f)

orig_texts = src["texts"]        
orig_fmap  = src["file_map"]      

new_texts, new_fmap = [], {}

for fname, idx in tqdm.tqdm(orig_fmap.items(), total=len(orig_fmap)):
    doc_id   = os.path.splitext(fname)[0]       
    doc_text = orig_texts[idx]

    for j, chunk in enumerate(split_slide(doc_text, MAX_LEN, STRIDE)):
        new_idx = len(new_texts)
        new_texts.append(chunk)
        new_fmap[f"{doc_id}_{j}"] = new_idx      

with open(DST_PKL, "wb") as f:
    pickle.dump({"texts": new_texts, "file_map": new_fmap}, f)

print(f"Complete: {DST_PKL}")
print(f"Doc : {len(orig_fmap):,}")
print(f"Chunks : {len(new_texts):,}")