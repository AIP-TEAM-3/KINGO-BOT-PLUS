import pandas as pd
import torch
from bert_score import score as bert_score
from openai import OpenAI

client = OpenAI(api_key="your-key")

CSV_IN  = "/content/Question2doc_Pair - Prompting_result.csv"
CSV_OUT = "Question2doc_pair-prompting_with_scores.csv"

df = pd.read_csv(CSV_IN)
df["long_answer"] = df["long_answer"].fillna("").astype(str)
df["prediction"]  = df["prediction"].fillna("").astype(str)


P, R, F1 = bert_score(
    cands      = df["prediction"].tolist(),
    refs       = df["long_answer"].tolist(),
    model_type = "klue/roberta-large",
    num_layers = 24,
    batch_size = 32,
    device     = "cuda" if torch.cuda.is_available() else "cpu",
)
df["bert_score"] = F1.cpu().numpy()

def gpt_yes_no(pred: str, ref: str) -> str:
    prompt = (
        "다음 두 한국어 문장에서, 정답 문장에 있는 모든 핵심 정보(날짜, 행사 일정)가 "
        "예측 문장에 정확히 포함되어 있으면 yes, "
        "틀리면 no만 출력해. "
        "추가적인 문장이나 단어는 절대 쓰지 마.\n\n"
        f"[예측 문장]\n{pred}\n\n[정답 문장]\n{ref}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=1,
        messages=[{"role": "user", "content": prompt}],
    )
    ans = resp.choices[0].message.content.strip().lower()
    return "yes" if ans.startswith("y") else "no"

df["gpt_same_answer"] = [
    gpt_yes_no(pred, ref) for pred, ref in zip(df["prediction"], df["long_answer"])
]

df.to_csv(CSV_OUT, index=False)
print(f"✅ 완료! 결과가 '{CSV_OUT}'에 저장되었습니다.")
