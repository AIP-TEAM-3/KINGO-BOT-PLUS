import openai
from openai import OpenAI
import os

client = OpenAI(api_key="sk-")

TEXT_DIR = "./texts"
OUTPUT_DIR = "./outputs_accent1"
QUESTIONS_PER_FILE = 5
TONE = [
    "나 제대 복학하려고 하는데 언제 신청해?\n수강신청 마감 기간이 언제까지야?", 
    "제대 복학 신청 기간에 대해 알려줘\n수강신청 마감 기간 알려줘", "제대 복학 신청 기간은 언제인가요?\n수강신청 마감 기간은 언제인가요?", 
    "제대 복학 신청 기간을 알려주세요.\n수강신청 마감기간을 알려주세요."
]

def generate_questions(text, filename, tone, num_questions=50):
    prompt = f"""
다음은 대학교 학사 안내 문서입니다:

{text}

이 문서를 참고하여 학생들이 자연스럽게 물어볼 수 있는 질문을 {num_questions}개 생성해주세요.
질문은 모두 {filename} 파일로부터 나온 것으로 가정하고, 다음과 같은 형식으로 출력해 주세요:

질문내용<TAB>{filename}

예:
제대복학 신청은 어디서 하나요?	{filename}
제대 후 복학 신청 시 필요한 서류가 무엇인가요?	{filename}

질문 형식이나 표현은 다양하게 작성해 주세요. 같은 의미라도 문장이 다르면 괜찮습니다.
한 줄에 하나의 질문만 출력해 주세요.

말투 형식 : {tone}

말투 형식을 잘 지켜주세요. 출력 형식도 반드시 지켜주세요.
넘버링을 붙이지 말아주세요.
질문내용<TAB>{filename}
반드시 이 형식을 지키세요.
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
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_questions.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(questions)

    # 통합 파일에 덧붙이기
    all_path = os.path.join(OUTPUT_DIR, "all_questions.txt")
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

        print(f"Generating questions for {fname}...")
        questions = generate_questions(content, fname, tone=TONE[0])
        save_questions(fname.replace(".txt", ""), questions)
        print(f"📁 저장 완료: outputs/{fname.replace('.txt', '')}_questions.txt\n")

if __name__ == "__main__":
    # 👇 여기에서 시작~끝 파일 번호 지정 (예: 22~23)
    start_file = 8
    end_file = 179

    process_files(start=start_file, end=end_file)