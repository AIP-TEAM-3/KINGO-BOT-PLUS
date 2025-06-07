import os
import pandas as pd

def load_documents_from_folder(folder_path):
    """Load all .txt files from a folder into a dict: {filename: content}"""
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                documents[filename] = f.read().strip()
    return documents

def load_qa_mapping(file_path):
    """Load Excel with question-answer-doc_id and build QA_SET dict"""
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df["query"] = df["query"].str.strip()
    df["answer"] = df["answer"].str.strip()
    df["idex"] = df["idex"].astype(str).str.strip()
    
    qa_set = {
        row["query"]: (row["answer"], row["idex"])
        for _, row in df.iterrows()
    }
    return qa_set

# Usage
#DOCUMENTS = load_documents_from_folder("texts/")
#QA_SET = load_qa_mapping("QA_simple.xlsx")

# Preview
#print("DOCUMENTS keys:", list(DOCUMENTS.keys()))
#print("QA_SET sample:", list(QA_SET.items())[:1])