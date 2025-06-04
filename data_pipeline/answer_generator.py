import openai
from openai import OpenAI
import os

client = OpenAI(api_key="sk-")

TEXT_DIR = "./texts"
QUERY_DIR = "./querys"
OUTPUT_DIR = "./answers"

def generate_questions(text, query, filename, answer_num=5):
    prompt = f"""
다음은 대학교 학사 안내 문서입니다:

{text}

이 문서를 참고하여 다음 질문 리스트에서 상위 {answer_num}개를 이용하여 그에 매칭되는 답변을 생성해주세요.

{query}

질문은 모두 {filename} 파일로부터 나온 것으로 가정하고, 다음과 같은 형식으로 출력해 주세요:

질문내용<TAB>답변내용<TAB>{filename}

예:
제대복학 신청은 어디서 하나요?  제대 복학은 GLS에서 신청합니다. {filename}

출력 형식을 반드시 지켜줘. 다른 linenumber라던가 절대 추가하지마.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000
    )

    return response.choices[0].message.content

def save_questions(filename, questions):
    # 개별 파일 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_answers.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(questions)

    # ✅ 통합 파일에 덧붙이기
    all_path = os.path.join(OUTPUT_DIR, "all_answers.txt")
    with open(all_path, "a", encoding="utf-8") as f:
        f.write(questions.strip() + "\n")  # 줄바꿈 보장

def process_files(start: int, end: int):
    for i in range(start, end + 1):
        fname = f"{i}.txt"
        text_path = os.path.join(TEXT_DIR, fname)
        if not os.path.exists(text_path):
            print(f"❌ 파일 없음: {fname}")
            continue

        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"⚠️  파일 비어있음: {fname}")
                continue

        qname = f"{i}_questions.txt"
        query_path = os.path.join(QUERY_DIR, qname)
        if not os.path.exists(query_path):
            print(f"❌ 파일 없음: {qname}")
            continue

        with open(query_path, "r", encoding="utf-8") as q:
            query = q.read().strip()
            if not query:
                print(f"⚠️  파일 비어있음: {qname}")
                continue

        print(f"✅ Generating questions for {fname}...")
        answers = generate_questions(content, query, fname)
        save_questions(fname.replace(".txt", ""), answers)
        print(f"📁 저장 완료: answers/{fname.replace('.txt', '')}_answers.txt\n")

if __name__ == "__main__":
    # 👇 여기에서 시작~끝 파일 번호 지정 (예: 22~23)
    start_file = 8
    end_file = 179

    process_files(start=start_file, end=end_file)