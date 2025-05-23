import streamlit as st
import os
import tempfile
import shutil
import time
from koach_s import Koach
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("streamlit-app")

# 페이지 설정
st.set_page_config(page_title="한국어 발음 코치", page_icon="🎤", layout="wide")

# 제목 및 소개
st.title("🎤 한국어 발음 코치")
st.markdown("### 한국어 발음을 분석하고 개인화된 피드백을 받아보세요")

# 필요한 디렉토리 생성
os.makedirs("mfa_input", exist_ok=True)
os.makedirs("mfa_output", exist_ok=True)
os.makedirs("lexicon", exist_ok=True)

# API 키 설정 (환경 변수에서 가져오기)
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# 메인 컨텐츠
tab1, tab2 = st.tabs(["📝 발음 분석", "ℹ️ 사용 방법"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("학습자 오디오")
        learner_audio = st.file_uploader(
            "학습자 오디오 파일을 업로드하세요",
            type=["wav", "mp3", "m4a", "ogg"],
            key="learner_audio",
        )

    with col2:
        st.subheader("원어민 오디오")
        native_audio = st.file_uploader(
            "원어민 오디오 파일을 업로드하세요",
            type=["wav", "mp3", "m4a", "ogg"],
            key="native_audio",
        )

    # 스크립트 입력
    st.subheader("스크립트 (선택사항)")
    script = st.text_area(
        "발음할 텍스트를 입력하세요 (없으면 원어민 발화에서 자동 추출)", height=100
    )

    # 분석 버튼
    analyze_button = st.button(
        "발음 분석하기", type="primary", use_container_width=True
    )

    # 분석 실행
    if analyze_button:
        if not learner_audio or not native_audio:
            st.error("학습자와 원어민 오디오 파일이 모두 필요합니다.")
        elif not openai_api_key:
            st.error("OpenAI API 키가 설정되지 않았습니다. 환경 변수로 설정해주세요.")
        else:
            # 임시 디렉토리 생성
            temp_dir = tempfile.mkdtemp()
            learner_path = os.path.join(temp_dir, "learner_audio")
            native_path = os.path.join(temp_dir, "native_audio")

            # 파일 업로더에서 파일 저장
            with open(learner_path, "wb") as f:
                f.write(learner_audio.getbuffer())

            with open(native_path, "wb") as f:
                f.write(native_audio.getbuffer())

            # 설정 준비 (간소화된 버전)
            config = {
                "whisper_model": "base",
                "openai_model": "gpt-4o",
                "mfa_input_dir": "mfa_input",
                "mfa_output_dir": "mfa_output",
                "lexicon_path": "models/korean_mfa.dict",
                "acoustic_model": "models/korean_mfa.zip",
            }

            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # 발음 코치 초기화
                status_text.text("발음 코치 초기화 중...")
                progress_bar.progress(10)
                koach = Koach(config)

                # 분석 실행
                status_text.text("오디오 변환 및 전사 중...")
                progress_bar.progress(30)

                # 분석 결과 (실제로는 한 번에 실행되지만 UI를 위해 단계별로 표시)
                time.sleep(1)  # UI 업데이트를 위한 지연
                status_text.text("음성 정렬 중...")
                progress_bar.progress(60)

                time.sleep(1)  # UI 업데이트를 위한 지연
                status_text.text("피드백 생성 중...")
                progress_bar.progress(80)

                # 최종 분석 실행
                result = koach.analyze_pronunciation(
                    learner_audio=learner_path,
                    native_audio=native_path,
                    script=script if script else None,
                )
                progress_bar.progress(100)
                status_text.text("분석 완료!")

                # 결과 표시
                if result["success"]:
                    st.success("발음 분석이 완료되었습니다!")

                    # 결과 컨테이너
                    result_container = st.container()

                    with result_container:
                        # 전사 텍스트 비교
                        st.subheader("📊 전사 결과 비교")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 학습자 발화")
                            st.info(result["learner_text"])

                        with col2:
                            st.markdown("#### 원어민 발화")
                            st.info(result["native_text"])

                        if result.get("script_text"):
                            st.markdown("#### 목표 스크립트")
                            st.success(result["script_text"])

                        # 피드백 표시
                        st.subheader("🔍 발음 피드백")
                        st.markdown(result["feedback"])

                        # 결과 저장 버튼
                        if st.button("결과 저장하기"):
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            filename = f"pronunciation_feedback_{timestamp}.txt"

                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(f"# 한국어 발음 피드백 ({timestamp})\n\n")
                                f.write(f"## 학습자 발화\n{result['learner_text']}\n\n")
                                f.write(f"## 원어민 발화\n{result['native_text']}\n\n")
                                if result.get("script_text"):
                                    f.write(
                                        f"## 목표 스크립트\n{result['script_text']}\n\n"
                                    )
                                f.write(f"## 피드백\n{result['feedback']}")

                            st.success(f"결과가 {filename}에 저장되었습니다.")
                else:
                    st.error(f"분석 실패: {result['error']}")

            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                logger.exception("분석 중 오류 발생")

            finally:
                # 임시 파일 정리
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"임시 파일 정리 중 오류: {e}")

with tab2:
    st.header("한국어 발음 코치 사용 방법")

    st.markdown(
        """
    ### 1. 오디오 준비
    - **학습자 오디오**: 발음을 평가받고 싶은 한국어 문장을 녹음한 파일을 업로드하세요.
    - **원어민 오디오**: 같은 문장을 발음한 원어민의 오디오 파일을 업로드하세요.
    
    ### 2. 스크립트 입력 (선택사항)
    - 발음하려는 텍스트를 입력하면 더 정확한 분석이 가능합니다.
    - 입력하지 않으면 원어민 발화에서 자동으로 추출합니다.
    
    ### 3. 분석 실행
    - "발음 분석하기" 버튼을 클릭하여 분석을 시작합니다.
    - 분석에는 약 1-2분이 소요됩니다.
    
    ### 4. 결과 확인
    - 학습자와 원어민의 발화 텍스트를 비교합니다.
    - 발음 피드백을 확인하고 개선점을 파악합니다.
    - 필요시 결과를 파일로 저장할 수 있습니다.
    """
    )

    st.info("💡 팁: 조용한 환경에서 녹음하면 더 정확한 분석 결과를 얻을 수 있습니다.")

    st.subheader("필요한 설정")
    st.markdown(
        """
    ### 환경 변수 설정
    - OpenAI API 키를 환경 변수로 설정해야 합니다:
      ```
      export OPENAI_API_KEY=your_api_key_here
      ```
    
    ### 필요한 파일
    - 한국어 발음 사전 파일(`lexicon/korean_lexicon.txt`)이 필요합니다.
    - 한국어 음향 모델(`korean_acoustic_model`)이 필요합니다.
    """
    )

# 푸터
st.markdown("---")
st.markdown("© 2025 한국어 발음 코치 | 문의: example@example.com")
