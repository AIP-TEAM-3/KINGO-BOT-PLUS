import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import openai
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = ""

qdrant_client = QdrantClient(host="qdrant", port=6333)

llm_model = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "google/gemini-2.0-flash-001"
]
embedding_model_path = "../training_results/expr3/ckpt/FT_epoch10"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(embedding_model_path, device=device)
model.max_seq_length = 512

def sentence2vec(text):
    return model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]

def get_reference(question):
    vector = sentence2vec(question)
    
    related_docs = []

    query_result = qdrant_client.search(
    collection_name="skku_doc",
    query_vector= vector,
    limit=3,
    )

    for result in query_result:
        related_docs.append(result.payload["text"])

    return related_docs

def make_prompt(question,ref_docs):
    prompt = f"""
            당신은 다음 문서들을 통해 정보를 제공하는 챗봇입니다.
            질문에 답하기 전에 먼저 문서를 바탕으로 단계별로 생각을 전개하세요.
            문서에 정보가 없으면 "문서에 해당 내용이 없습니다"라고 답하세요.

            [참고 문서들]
            {ref_docs[0]},

            {ref_docs[1]},

            {ref_docs[2]}.

            [질문]
            {question}"""
    
    return prompt.strip()

def llm_request(model_name, prompt, max_tokens=150):
    try:
        client = OpenAI(
            base_url=openai.api_base,
            api_key=openai.api_key,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling model {model_name}: {e}")
        return ""