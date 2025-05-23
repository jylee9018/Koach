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
st.set_page_config(
    page_title="한국어 발음 코치",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 테마 설정 (세션 상태에 저장)
if "theme" not in st.session_state:
    # URL 쿼리 파라미터에서 테마 설정 가져오기
    query_params = st.query_params
    theme_param = (
        query_params.get("theme", ["light"])[0] if "theme" in query_params else "light"
    )
    st.session_state.theme = theme_param


# 테마 전환 함수
def toggle_theme():
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

    # URL 쿼리 파라미터 업데이트
    st.query_params["theme"] = st.session_state.theme


# CSS 스타일 적용
def get_css():
    if st.session_state.theme == "dark":
        return """
        <style>
            /* 다크 모드 스타일 */
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Noto Sans KR', sans-serif;
                color: #e0e0e0;
            }
            
            .stApp {
                background-color: #121212;
            }
            
            /* 헤더 스타일 */
            .main-header {
                background-color: #1e2a38;
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                border-left: 5px solid #4361ee;
            }
            
            /* 카드 스타일 */
            .card {
                background-color: #1e1e1e;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                margin-bottom: 1rem;
            }
            
            /* 업로더 스타일 */
            .upload-section {
                border: 2px dashed #444;
                border-radius: 10px;
                padding: 1.5rem;
                text-align: center;
                margin-bottom: 1rem;
                transition: all 0.3s;
                background-color: #252525;
            }
            
            .upload-section:hover {
                border-color: #4361ee;
            }
            
            /* 버튼 스타일 */
            .stButton>button {
                background-color: #4361ee;
                color: white;
                font-weight: 500;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                transition: all 0.3s;
            }
            
            .stButton>button:hover {
                background-color: #3a56d4;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            }
            
            /* 결과 컨테이너 */
            .result-container {
                background-color: #252525;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 2rem;
            }
            
            /* 피드백 섹션 */
            .feedback-section {
                background-color: #1e2a38;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 1rem;
                border-left: 5px solid #4361ee;
            }
            
            /* 진행 상태 */
            .progress-section {
                background-color: #2d2a1e;
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 5px solid #ffc107;
            }
            
            /* 텍스트 영역 */
            .transcript-box {
                background-color: #2a2a2a;
                border-radius: 8px;
                padding: 1rem;
                border: 1px solid #444;
                margin-bottom: 1rem;
                max-height: 200px;
                overflow-y: auto;
            }
            
            /* 푸터 */
            .footer {
                text-align: center;
                padding: 1rem;
                margin-top: 2rem;
                border-top: 1px solid #333;
                color: #888;
            }
            
            /* 탭 스타일 */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
                background-color: #1a1a1a;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 4rem;
                white-space: pre-wrap;
                background-color: #252525;
                border-radius: 5px 5px 0 0;
                padding: 1rem 2rem;
                font-weight: 500;
                color: #ccc;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #4361ee !important;
                color: white !important;
            }
            
            /* 테마 전환 버튼 */
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
                background-color: #333;
                color: white;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            }
            
            /* 정보 박스 */
            .stAlert {
                background-color: #1e2a38;
                color: #e0e0e0;
            }
            
            /* 텍스트 입력 */
            .stTextInput>div>div>input {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border-color: #444;
            }
            
            .stTextArea>div>div>textarea {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border-color: #444;
            }
            
            /* 선택 박스 */
            .stSelectbox>div>div {
                background-color: #2a2a2a;
                color: #e0e0e0;
            }
        </style>
        """
    else:
        return """
        <style>
            /* 라이트 모드 스타일 */
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Noto Sans KR', sans-serif;
            }
            
            /* 헤더 스타일 */
            .main-header {
                background-color: #f0f7ff;
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                border-left: 5px solid #4361ee;
            }
            
            /* 카드 스타일 */
            .card {
                background-color: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }
            
            /* 업로더 스타일 */
            .upload-section {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 1.5rem;
                text-align: center;
                margin-bottom: 1rem;
                transition: all 0.3s;
            }
            
            .upload-section:hover {
                border-color: #4361ee;
            }
            
            /* 버튼 스타일 */
            .stButton>button {
                background-color: #4361ee;
                color: white;
                font-weight: 500;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                transition: all 0.3s;
            }
            
            .stButton>button:hover {
                background-color: #3a56d4;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            
            /* 결과 컨테이너 */
            .result-container {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 2rem;
            }
            
            /* 피드백 섹션 */
            .feedback-section {
                background-color: #f0f7ff;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 1rem;
                border-left: 5px solid #4361ee;
            }
            
            /* 진행 상태 */
            .progress-section {
                background-color: #fff8e6;
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 5px solid #ffc107;
            }
            
            /* 텍스트 영역 */
            .transcript-box {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
                border: 1px solid #e9ecef;
                margin-bottom: 1rem;
                max-height: 200px;
                overflow-y: auto;
            }
            
            /* 푸터 */
            .footer {
                text-align: center;
                padding: 1rem;
                margin-top: 2rem;
                border-top: 1px solid #e9ecef;
                color: #6c757d;
            }
            
            /* 탭 스타일 */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 4rem;
                white-space: pre-wrap;
                background-color: #f8f9fa;
                border-radius: 5px 5px 0 0;
                padding: 1rem 2rem;
                font-weight: 500;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #4361ee !important;
                color: white !important;
            }
            
            /* 테마 전환 버튼 */
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
                background-color: #f8f9fa;
                color: #333;
                border: 1px solid #e9ecef;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
        </style>
        """


# CSS 적용
st.markdown(get_css(), unsafe_allow_html=True)

# 테마 전환 버튼 추가
theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
theme_button_html = f"""
<button class="theme-toggle" onclick="parent.postMessage({{theme: 'toggle'}}, '*')">
    {theme_icon}
</button>
<script>
    window.addEventListener('message', function(e) {{
        if (e.data.theme === 'toggle') {{
            const form = document.createElement('form');
            form.method = 'POST';
            form.action = '';

            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = 'theme_toggle';
            input.value = 'true';

            form.appendChild(input);
            document.body.appendChild(form);
            form.submit();
        }}
    }});
</script>
"""
st.markdown(theme_button_html, unsafe_allow_html=True)

# 테마 토글 처리
if st.button("테마 전환", key="theme_toggle_button", help="라이트/다크 모드 전환"):
    toggle_theme()
    st.rerun()

# 필요한 디렉토리 생성
os.makedirs("mfa_input", exist_ok=True)
os.makedirs("mfa_output", exist_ok=True)
os.makedirs("lexicon", exist_ok=True)

# 제목 및 소개
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🎤 한국어 발음 코치")
st.markdown("### 한국어 발음을 분석하고 개인화된 피드백을 받아보세요")
st.markdown("</div>", unsafe_allow_html=True)

# API 키 설정 (환경 변수에서 가져오기)
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# 메인 컨텐츠
tab1, tab2 = st.tabs(["📝 발음 분석", "ℹ️ 사용 방법"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("학습자 오디오")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        learner_audio = st.file_uploader(
            "학습자 오디오 파일을 업로드하세요",
            type=["wav", "mp3", "m4a", "ogg"],
            key="learner_audio",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("원어민 오디오")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        native_audio = st.file_uploader(
            "원어민 오디오 파일을 업로드하세요",
            type=["wav", "mp3", "m4a", "ogg"],
            key="native_audio",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # 스크립트 입력
    st.subheader("스크립트 (선택사항)")
    script = st.text_area(
        "발음할 텍스트를 입력하세요 (없으면 원어민 발화에서 자동 추출)", height=100
    )

    # 분석 버튼
    analyze_button = st.button(
        "🔍 발음 분석하기", type="primary", use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 분석 실행
    if analyze_button:
        if not learner_audio or not native_audio:
            st.error("❌ 학습자와 원어민 오디오 파일이 모두 필요합니다.")
        elif not openai_api_key:
            st.error(
                "❌ OpenAI API 키가 설정되지 않았습니다. 환경 변수로 설정해주세요."
            )
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

            # 설정
            config = {
                "whisper_model": "base",  # 기본 모델 사용
                "openai_model": "gpt-4o",  # GPT-4o 사용
                "mfa_input_dir": "mfa_input",
                "lexicon_path": "models/korean_mfa.dict",
                "acoustic_model": "models/korean_mfa.zip",
            }

            # 진행 상태 표시
            st.markdown('<div class="progress-section">', unsafe_allow_html=True)
            st.subheader("🔄 분석 진행 중...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

            try:
                # 발음 코치 초기화
                status_text.text("🔄 발음 코치 초기화 중...")
                progress_bar.progress(10)
                koach = Koach(config)

                # 분석 실행
                status_text.text("🔄 오디오 변환 및 전사 중...")
                progress_bar.progress(30)

                # 분석 결과 (실제로는 한 번에 실행되지만 UI를 위해 단계별로 표시)
                time.sleep(1)  # UI 업데이트를 위한 지연
                status_text.text("🔄 음성 정렬 중...")
                progress_bar.progress(60)

                time.sleep(1)  # UI 업데이트를 위한 지연
                status_text.text("🔄 피드백 생성 중...")
                progress_bar.progress(80)

                # 최종 분석 실행
                result = koach.analyze_pronunciation(
                    learner_audio=learner_path,
                    native_audio=native_path,
                    script=script if script else None,
                )
                progress_bar.progress(100)
                status_text.text("✅ 분석 완료!")

                # 결과 표시
                if result["success"]:
                    st.success("✅ 발음 분석이 완료되었습니다!")

                    # 결과 컨테이너
                    st.markdown(
                        '<div class="result-container">', unsafe_allow_html=True
                    )

                    # 전사 텍스트 비교
                    st.subheader("📊 전사 결과 비교")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 학습자 발화")
                        st.markdown(
                            '<div class="transcript-box">', unsafe_allow_html=True
                        )
                        st.info(result["learner_text"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        st.markdown("#### 원어민 발화")
                        st.markdown(
                            '<div class="transcript-box">', unsafe_allow_html=True
                        )
                        st.info(result["native_text"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    if result.get("script_text"):
                        st.markdown("#### 목표 스크립트")
                        st.markdown(
                            '<div class="transcript-box">', unsafe_allow_html=True
                        )
                        st.success(result["script_text"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    # 피드백 표시
                    st.markdown(
                        '<div class="feedback-section">', unsafe_allow_html=True
                    )
                    st.subheader("🔍 발음 피드백")
                    st.markdown(result["feedback"])
                    st.markdown("</div>", unsafe_allow_html=True)

                    # 결과 저장 버튼
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("💾 결과 저장하기", use_container_width=True):
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

                            st.success(f"✅ 결과가 {filename}에 저장되었습니다.")

                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"❌ 분석 실패: {result['error']}")

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                logger.exception("분석 중 오류 발생")

            finally:
                # 임시 파일 정리
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"임시 파일 정리 중 오류: {e}")

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📚 한국어 발음 코치 사용 방법")

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

    st.subheader("⚙️ 필요한 설정")
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
    st.markdown("</div>", unsafe_allow_html=True)

# 푸터
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2025 한국어 발음 코치 | 문의: example@example.com")
st.markdown("</div>", unsafe_allow_html=True)
