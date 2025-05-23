from langchain_core.prompts import ChatPromptTemplate

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 한국어 발음 전문가입니다. 발음 특징을 분석하고 피드백을 제공해주세요.",
        ),
        (
            "human",
            """
    다음 발음 특징을 분석해주세요:
    Pitch: {pitch}
    Duration: {duration}
    Energy: {energy}
    
    분석 결과를 다음 형식으로 제공해주세요:
    1. 전반적인 발음 평가
    2. 주요 문제점
    3. 개선 방향
    """,
        ),
    ]
)

FEEDBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 친절한 한국어 발음 코치입니다. 학습자에게 도움이 되는 피드백을 제공해주세요.",
        ),
        (
            "human",
            """
    다음 분석 결과를 바탕으로 피드백을 생성해주세요:
    {analysis}
    
    다음 형식으로 피드백을 제공해주세요:
    1. 칭찬할 점
    2. 개선이 필요한 점
    3. 구체적인 연습 방법
    4. 예시 단어/문장
    """,
        ),
    ]
)

COMPARISON_FEEDBACK_PROMPT = """
다음은 학습자의 한국어 발음을 원어민 발음과 비교 분석한 결과입니다:

스크립트: {script}
전체 유사도: {similarity:.2f}
발견된 오류: {pronunciation_errors}

세부 분석:
- 억양(피치) 차이: {analysis_details['pitch']['difference']:.2f} Hz
- 억양 패턴 유사도: {analysis_details['pitch']['contour_similarity']:.2f}
- 발화 속도 비율(학습자/원어민): {analysis_details['duration']['pace_ratio']:.2f}
- 발화 강도 차이: {analysis_details['energy']['difference']:.2f}

이 분석 결과를 바탕으로 학습자에게 도움이 되는 피드백을 제공해주세요.
다음 형식으로 피드백을 생성해주세요:

1. 잘한 점
2. 개선이 필요한 발음 요소 (억양, 속도, 강세 등)
3. 구체적인 연습 방법 및 팁
4. 추천 연습 문장

답변은 한국어 학습자가 이해하기 쉽게 작성해주세요.
"""
