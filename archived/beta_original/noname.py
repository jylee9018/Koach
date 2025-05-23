import os
import subprocess
import whisper
from openai import OpenAI
import shutil
from pydub import AudioSegment
import textgrid

# 환경 변수에서 OpenAI API 키 불러오기
api_key = os.getenv("OPENAI_API_KEY")

# === 파일 경로 설정 ===

# 원본 입력 파일
LEARNER_AUDIO = "input/learner.m4a"
NATIVE_AUDIO = "input/native.m4a"

# 변환된 WAV 파일
LEARNER_WAV = "wav/learner.wav"
NATIVE_WAV = "wav/native.wav"

# Whisper 전사 결과
LEARNER_TRANSCRIPT = "wav/learner.txt"
NATIVE_TRANSCRIPT = "wav/native.txt"

# 정답 스크립트
SCRIPT_PATH = "wav/script.txt"

# MFA 관련 파일
LEXICON_PATH = "models/korean_mfa.dict"
ACOUSTIC_MODEL = "models/korean_mfa.zip"
MFA_INPUT = "mfa_input"
MFA_OUTPUT = "aligned"

LEARNER_TEXTGRID = os.path.join(MFA_OUTPUT, "learner.TextGrid")
NATIVE_TEXTGRID = os.path.join(MFA_OUTPUT, "native.TextGrid")


# === 1. 오디오 변환 ===
def convert_audio(input_path, output_path):
    print(f"🎧 변환 중: {input_path} → {output_path}")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    print("✅ 오디오 변환 완료")


# === 2. Whisper 전사 ===
def transcribe_audio(wav_path, transcript_path):
    print(f"📝 Whisper 전사 중: {wav_path}")
    model = whisper.load_model("base")
    result = model.transcribe(wav_path, language="ko")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"📄 전사 결과: {result['text']}")
    return result["text"]


# === 3. MFA 정렬 ===
def run_mfa_alignment(wav_path, transcript_path, output_name):
    print(f"🔧 MFA 정렬 시작: {output_name}")
    os.makedirs(MFA_INPUT, exist_ok=True)
    os.makedirs(MFA_OUTPUT, exist_ok=True)

    # 파일 복사
    shutil.copy(wav_path, os.path.join(MFA_INPUT, f"{output_name}.wav"))
    shutil.copy(transcript_path, os.path.join(MFA_INPUT, f"{output_name}.txt"))

    # MFA 정렬 실행
    command = [
        "mfa",
        "align",
        MFA_INPUT,
        LEXICON_PATH,
        ACOUSTIC_MODEL,
        MFA_OUTPUT,
        "--clean",
        "--no_text_cleaning",
    ]
    subprocess.run(command, check=True)
    print("✅ MFA 정렬 완료")


# === 4. TextGrid 요약 ===
def summarize_textgrid(path):
    print(f"📊 TextGrid 요약 중: {path}")
    tg = textgrid.TextGrid.fromFile(path)
    summary = []
    for tier in tg.tiers:
        if tier.name.lower() in ["phones", "phoneme", "phone"]:
            for interval in tier:
                if interval.mark.strip():
                    summary.append(
                        f"{interval.mark}: {round(interval.minTime, 2)}s ~ {round(interval.maxTime, 2)}s"
                    )
    return "\n".join(summary)


# === 5. GPT 프롬프트 생성 ===
def generate_prompt(
    learner_text, native_text, script_text, learner_timing, native_timing
):
    prompt = f"""
다음은 한국어 학습자의 발화 정보와 원어민의 예시 발화 정보입니다.

# 학습자 발화 텍스트:
"{learner_text}"

# 원어민 발화 텍스트:
"{native_text}"

# 목표 스크립트:
"{script_text}"

# 학습자의 음소 정렬 정보:
{learner_timing}

# 원어민의 음소 정렬 정보:
{native_timing}

위 정보를 바탕으로 다음을 분석해줘:

1. 학습자의 발음에서 누락되거나 부정확한 단어나 음소는 무엇인가?
   - 구체적으로 제시해줘.

2. 원어민과 비교했을 때 어떤 **단어나 구절에서** 속도 차이가 있는가?  
   - 속도 정보를 제시할 때는 꼭 해당하는 **단어나 음소**를 함께 말해줘.

3. 더 자연스럽고 명확하게 발음하기 위한 팁을 간단히 
"""
    return prompt


# === 6. GPT 호출 ===
def get_feedback(prompt):
    print("🤖 GPT 피드백 생성 중...")
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 한국어 발음 평가 전문가입니다."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# === 7. 전체 실행 ===
def main():
    # 오디오 변환
    convert_audio(LEARNER_AUDIO, LEARNER_WAV)
    convert_audio(NATIVE_AUDIO, NATIVE_WAV)

    # Whisper 전사
    learner_text = transcribe_audio(LEARNER_WAV, LEARNER_TRANSCRIPT)
    native_text = transcribe_audio(NATIVE_WAV, NATIVE_TRANSCRIPT)

    # 스크립트 로딩
    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        script_text = f.read().strip()

    # MFA 정렬
    run_mfa_alignment(LEARNER_WAV, LEARNER_TRANSCRIPT, "learner")
    run_mfa_alignment(NATIVE_WAV, NATIVE_TRANSCRIPT, "native")

    # TextGrid 요약
    learner_timing = summarize_textgrid(LEARNER_TEXTGRID)
    native_timing = summarize_textgrid(NATIVE_TEXTGRID)

    # 프롬프트 생성
    prompt = generate_prompt(
        learner_text, native_text, script_text, learner_timing, native_timing
    )

    # GPT 피드백 생성
    feedback = get_feedback(prompt)

    # 출력
    print("\n📣 발음 피드백 결과:\n")
    print(feedback)


if __name__ == "__main__":
    main()
