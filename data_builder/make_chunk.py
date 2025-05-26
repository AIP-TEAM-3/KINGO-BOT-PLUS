import os
import pickle

TEXT_FOLDER = "./data/texts" 
SAVE_PATH = "./data/chunk/chunks.pkl"

chunks = {}
sources = {}

for filename in sorted(os.listdir(TEXT_FOLDER)):
    if filename.endswith(".txt") and filename[:-4].isdigit():
        file_path = os.path.join(TEXT_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            chunks[filename] = text
            sources[filename] = filename

with open(SAVE_PATH, "wb") as f:
    pickle.dump({"texts": chunks, "file_map": sources}, f)


print(f"Saving {len(chunks)} doc → {SAVE_PATH}")