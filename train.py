import os
import pickle
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
import json

EXPR_NAME = "expr1"
CSV_PATH = "./data/csv/train.csv"
PKL_PATH = "./chunks_split.pkl"
OUTPUT_DIR = "./training_results/" + EXPR_NAME + "/ckpt"

EPOCHS = 20
BATCH = 64
LR = 2e-5
MAXLEN = 512
GRAD_ACC = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_LOG = "./training_results/" + EXPR_NAME + "/log.json"

def load_data(csv_path, pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        breakpoint()
    texts = data["texts"]
    file_map = data["file_map"]

    df = pd.read_csv(csv_path)
    df["Doc_index"] = df["Doc_index"].astype(str).str.replace(".txt", "", regex=False)
    return df, texts, file_map

def build_examples(df_, chunks, fmap):
    examples = []
    for _, r in df_.iterrows():
        doc_id = str(r.Doc_index)
        matched_chunks = [k for k in fmap if k.startswith(f"{doc_id}_")]
        for k in matched_chunks:
            examples.append(InputExample(texts=[r.Question, chunks[fmap[k]]]))
    return examples

def to_dev(batch):
    return [{k: v.to(DEVICE) for k, v in sf.items()} for sf in batch]

def eval_loader(model, loader, loss_fn):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for ft, _ in loader:
            ft = to_dev(ft)
            tot += loss_fn(ft, None).item()
    return tot / len(loader)

def save_checkpoint(model, tag, epoch):
    save_path = f"{OUTPUT_DIR}/{tag}_epoch{epoch}"
    model.save(save_path)

def log_epoch(log_path, epoch, tag, train_loss, val_loss):
    log_entry = {
        "epoch": epoch,
        "tag": tag,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    run = 0.0
    bar = tqdm.tqdm(dataloader, desc="Training")
    optimizer.zero_grad()
    for st, (ft, _) in enumerate(bar):
        ft = to_dev(ft)
        loss = loss_fn(ft, None) / GRAD_ACC
        loss.backward()
        run += loss.item()
        if (st + 1) % GRAD_ACC == 0:
            optimizer.step()
            optimizer.zero_grad()
    return run / len(dataloader)

def train_and_save(model, tr_df, va_df, chunks, fmap, tag):
    os.makedirs(os.path.dirname(JSON_LOG), exist_ok=True)
    model.max_seq_length = MAXLEN
    collate = model.smart_batching_collate
    trL = DataLoader(build_examples(tr_df, chunks, fmap), shuffle=True, batch_size=BATCH, collate_fn=collate)
    vaL = DataLoader(build_examples(va_df, chunks, fmap), shuffle=False, batch_size=BATCH, collate_fn=collate)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    for ep in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, trL, loss_fn, optimizer)
        val_loss = eval_loader(model, vaL, loss_fn)
        print(f"[{tag} Epoch {ep}] Train {train_loss:.4f} | Val {val_loss:.4f}")

        if ep % 10 == 0:
            save_checkpoint(model, tag, ep)
        log_epoch(JSON_LOG, ep, tag, train_loss, val_loss)

def main():
    df, chunks, fmap = load_data(CSV_PATH, PKL_PATH)
    tr_df, va_df = train_test_split(df, test_size=0.1, random_state=42)
    model_ft = SentenceTransformer("dragonkue/BGE-m3-ko", device=DEVICE)
    train_and_save(model_ft, tr_df, va_df, chunks, fmap, tag="FT")

if __name__ == "__main__":
    main()
