import streamlit as st
import requests
from io import BytesIO

st.title("Koach - 한국어 발음 교정 AI 코치")

# 스크립트 입력
script = st.text_area(
    "발음할 한국어 문장/단어를 입력하세요", placeholder="예: 안녕하세요, 반갑습니다."
)

# 파일 업로드 영역
st.subheader("음성 파일 업로드")

col1, col2 = st.columns(2)

with col1:
    st.write("학습자 발음")
    learner_audio = st.file_uploader(
        "학습자 음성 파일 (.wav, .mp3, .m4a, .aac)",
        type=["wav", "mp3", "m4a", "aac"],
        key="learner_audio",
    )

    if learner_audio:
        st.audio(learner_audio)

with col2:
    st.write("원어민 발음 (선택사항)")
    native_audio = st.file_uploader(
        "원어민 음성 파일 (.wav, .mp3, .m4a, .aac)",
        type=["wav", "mp3", "m4a", "aac"],
        key="native_audio",
    )

    if native_audio:
        st.audio(native_audio)

if script and learner_audio:
    if st.button("분석 시작"):
        with st.spinner("분석 중..."):
            # 디버깅 정보 추가
            st.write(f"학습자 파일 정보: {learner_audio.name}, {type(learner_audio)}")
            if native_audio:
                st.write(f"원어민 파일 정보: {native_audio.name}, {type(native_audio)}")

            # FastAPI 서버로 요청 보내기
            files = {}

            # 파일 위치 재설정(필수)
            learner_audio.seek(0)
            files["learner_audio"] = (
                learner_audio.name,
                learner_audio.getvalue(),
                "audio/wav",
            )

            if native_audio:
                native_audio.seek(0)
                files["native_audio"] = (
                    native_audio.name,
                    native_audio.getvalue(),
                    "audio/wav",
                )

            data = {"script": script}

            try:
                url = "http://localhost:8000/api/v1/analyze"
                st.write(f"요청 URL: {url}")

                # 디버깅을 위한 헤더 추가
                headers = {"Accept": "application/json"}

                response = requests.post(url, files=files, data=data, headers=headers)

                # 디버그 출력
                st.write(f"응답 상태 코드: {response.status_code}")
                st.write(f"응답 내용: {response.text[:200]}...")  # 처음 200자만 표시

                if response.status_code == 200:
                    result = response.json()

                    # 분석 결과 표시
                    st.subheader("📊 분석 결과")

                    # 유사도 점수
                    st.metric("발음 유사도", f"{result.get('similarity_score', 0):.2f}")

                    # 발음 오류
                    if errors := result.get("pronunciation_errors"):
                        st.write("🚫 발견된 오류:")
                        for error in errors:
                            st.write(f"- {error}")

                    # GPT 피드백
                    if feedback := result.get("gpt_result"):
                        st.subheader("💡 AI 피드백")
                        st.write(feedback)
                else:
                    st.error(
                        f"분석 중 오류가 발생했습니다. (상태 코드: {response.status_code})\n응답: {response.text}"
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    "서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인해주세요."
                )
            except Exception as e:
                st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
else:
    st.info("분석을 시작하려면 스크립트를 입력하고 학습자 음성 파일을 업로드하세요.")
