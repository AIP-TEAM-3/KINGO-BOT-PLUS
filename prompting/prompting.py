import os
import random
import openai
from openai import OpenAI
import pandas as pd
from load_data import load_documents_from_folder, load_qa_mapping

# Set OpenRouter API base URL and API key
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENROUTER_API_KEY") 

MODELS = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "google/gemini-2.0-flash-001"
]

DOCUMENTS = load_documents_from_folder("/data/texts/")
QA_SET = load_qa_mapping("QA_simple.xlsx")

def sample_documents(correct_id, k=2):
    #정답 문서와 함께 섞을 랜덤 문서 k개를 반환
    pool = set(DOCUMENTS.keys()) - {correct_id}
    random_docs = random.sample(list(pool), k)
    return [correct_id] + random_docs

def format_prompt(prompt_type, docs, query):
    if prompt_type == "vanilla":
        return f"""이것은 너가 참조해야 하는 문서야:
                {docs[0]},

                {docs[1]},

                {docs[2]}.

                [질문] 
                {query}"""
    
    elif prompt_type == "task":
        return f"""[시스템 역할]
            당신은 성균관대학교 학사 제도에 특화된 지식형 챗봇입니다.
            참고 문서들 중 필요한 정보만 바탕으로, 학생의 질문에 정확히 답변하세요. 불필요하거나 관련 없는 내용은 포함하지 마세요.

            [참고 문서들]
            {docs[0]},

            {docs[1]},

            {docs[2]}.

            [질문]
            {query}"""

    elif prompt_type == "cot":
        return f"""
            당신은 다음 문서들을 통해 정보를 제공하는 챗봇입니다.
            질문에 답하기 전에 먼저 문서를 바탕으로 단계별로 생각을 전개하세요.
            문서에 정보가 없으면 "문서에 해당 내용이 없습니다"라고 답하세요.

            [참고 문서들]
            {docs[0]},

            {docs[1]},

            {docs[2]}.

            [질문]
            {query}"""

    else:
        raise ValueError("Unknown prompt type")


def build_prompt(prompt_type, query, doc_ids):
    #3개의 문서를 포함하는 프롬프트 생성
    doc_sections = []
    for idx, doc_id in enumerate(doc_ids, start=1):
        content = DOCUMENTS[doc_id]
        doc_sections.append(f"문서 {idx}: {content}")
    prompt = format_prompt(prompt_type, query, doc_sections)
    return prompt.strip()

def call_model(model_name, prompt, max_tokens=150):
    #OpenRouter API를 통해 LLM에 질의
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

def run_experiment():
    #전체 QA 세트와 모델에 대해 실험 실행
    results = []

    for query, (gold_answer, correct_id) in QA_SET.items():
        for prompt_type in ["vanilla", "task", "cot"]:
            for model in MODELS:
                doc_ids = sample_documents(correct_id, k=2)
                prompt = build_prompt(prompt_type, query, doc_ids)
                prediction = call_model(model, prompt)
            
                results.append({
                    "query": query,
                    "model": model,
                    "prompt_type": prompt_type,
                    "ref_docs": ", ".join(doc_ids),
                    "real_answer": gold_answer,
                    "prediction": prediction,
                })

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    # 실험 실행 및 결과 저장
    df_results = run_experiment()
    print(df_results)
    df_results.to_csv("rag_experiment_results.csv", index=False)
    print("실험이 완료되었습니다. 결과는 'rag_experiment_results.csv'에 저장되었습니다.")
