"""
Koach 한국어 발음 교정 시스템 - 중앙화된 설정 관리
"""

import os
from pathlib import Path
from typing import Dict, Any

# .env 파일 로드
try:
    from dotenv import load_dotenv
    # 프로젝트 루트의 .env 파일 로드
    _project_root = Path(__file__).parent.parent.parent
    env_path = _project_root / ".env"
    load_dotenv(env_path)
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않았습니다. .env 파일을 수동으로 로드해주세요.")

# =============================================================================
# 📁 기본 경로 설정 (Base Paths)
# =============================================================================

# 프로젝트 구조:
# Koach/
# ├── data/
# │   ├── input/                            # 입력 파일들
# │   └── output/                           # ✅ 최종 결과물 (JSON만)
# │       └── analysis_result.json
# │
# └── koach/
#     ├── temp/                             # ✅ 모든 중간 결과물
#     │   ├── wav/                          # WAV 변환 파일들
#     │   ├── normalized/                   # 정규화된 오디오 파일들
#     │   ├── mfa_input/                    # MFA 입력 파일들
#     │   ├── mfa_output/                   # MFA 출력 파일들 (TextGrid 파일들)
#     │   └── visualize/                    # 시각화 결과물들 (PNG)
#     │       ├── phoneme_analysis.png
#     │       ├── prosody_analysis.png
#     │       ├── comparison_analysis.png
#     │       └── prosody_comparison.png
#     ├── core/
#     ├── utils/
#     └── config/

# 절대 경로 계산
PROJECT_ROOT = Path(__file__).parent.parent.parent  # /Users/jlee/JDrvie/Dev/Koach
KOACH_ROOT = Path(__file__).parent.parent           # /Users/jlee/JDrvie/Dev/Koach/koach

# 주요 디렉토리들
DATA_ROOT = PROJECT_ROOT / "data"
INPUT_DIR = DATA_ROOT / "input"
OUTPUT_DIR = DATA_ROOT / "output"           
MODELS_DIR = KOACH_ROOT / "models"

TEMP_ROOT = KOACH_ROOT / "temp"
WAV_DIR = TEMP_ROOT / "wav"                         # 중간: WAV 파일들
NORMALIZED_DIR = TEMP_ROOT / "normalized"           # 중간: 정규화된 파일들
MFA_INPUT_DIR = TEMP_ROOT / "mfa_input"             # 중간: MFA 입력
MFA_OUTPUT_DIR = TEMP_ROOT / "mfa_output"           # 중간: TextGrid 파일들
VISUALIZE_DIR = TEMP_ROOT / "visualize"
KNOWLEDGE_DIR = KOACH_ROOT / "knowledge"

# 모델별 디렉토리들
WHISPER_MODEL_DIR = MODELS_DIR / "whisper"
MFA_LEXICON_PATH = MODELS_DIR / "korean_mfa.dict"          
MFA_ACOUSTIC_MODEL_PATH = MODELS_DIR / "korean_mfa.zip"  
FAISS_INDEX_PATH = MODELS_DIR / "faiss"
SENTENCE_TRANSFORMER_PATH = MODELS_DIR / "sentence_transformer"

# =============================================================================
# ⚙️ 설정 (Configuration)
# =============================================================================

CURRENT_CONFIG = {
    # 📁 경로 설정
    "learner_audio": str(INPUT_DIR / "learner.m4a"),
    "native_audio": str(INPUT_DIR / "native.m4a"),
    "output_dir": str(OUTPUT_DIR),
    "temp_dir": str(TEMP_ROOT),
    "wav_dir": str(WAV_DIR),
    "normalized_dir": str(NORMALIZED_DIR),
    "mfa_input_dir": str(MFA_INPUT_DIR),
    "mfa_output_dir": str(MFA_OUTPUT_DIR),
    "visualize_dir": str(VISUALIZE_DIR),
    "knowledge_dir": str(KNOWLEDGE_DIR),

    # 🎤 모델 설정
    "whisper_model": "base",
    "openai_model": "gpt-4o",
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "use_rag": True,
    
    # 📄 모델 파일 경로
    "lexicon_path": str(MFA_LEXICON_PATH),
    "acoustic_model": str(MFA_ACOUSTIC_MODEL_PATH),
    
    # 🔊 오디오 처리 설정
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
        "hop_length": 512,
        "frame_length": 2048,
        "top_db": 30,
    },
    
    # 🎙️ Whisper 설정
    "whisper": {
        "model_name": "base",
        "language": "ko",
        "task": "transcribe",
    },
    
    # 🔤 MFA 설정 (최적화 및 건너뛰기 옵션)
    "mfa": {
        "model_name": "korean",
        "num_jobs": 2,                    # CPU 코어 수에 맞게 조정
        "clean": True,
        "fast_mode": True,                # 빠른 정렬 모드
        "timeout": 120,                   # 2분 타임아웃
        "batch_processing": True,         # ✅ 배치 처리 활성화
        "skip_mfa": False,                # ✅ True로 설정하면 MFA 건너뛰기
        "no_text_cleaning": True,
        "speaker_mode": False,
        "lexicon_path": str(MFA_LEXICON_PATH),
        "acoustic_model": str(MFA_ACOUSTIC_MODEL_PATH),
    },

    # 📝 스크립트 관련 설정
    "script": {
        "skip_transcription_with_script": True,    # 스크립트 제공 시 음성 인식 건너뛰기
        "supported_extensions": [".txt", ".text"], # 지원하는 파일 확장자
        "encoding": "utf-8",                       # 파일 인코딩
        "auto_detect_file": True,                  # 자동 파일 감지 활성화
        "max_file_size": 1048576,                  # 최대 파일 크기 (1MB)
    },
    
    # 🔍 FAISS 설정
    "faiss": {
        "dimension": 384,
        "index_type": "L2",
    },
    
    # 🤖 Sentence Transformer 설정
    "sentence_transformer": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 32,
    },
    
    # 📊 시각화 설정
    "visualization": {
        "enabled": True,
        "dpi": 300,
        "figsize": (15, 10),
    },
    
    # 📝 로깅 설정
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
    },
}

# =============================================================================
# 📍 경로 딕셔너리들 (Path Dictionaries)
# =============================================================================

# core/koach.py에서 사용하는 PATHS
PATHS = {
    "learner_audio": INPUT_DIR / "learner.m4a",
    "native_audio": INPUT_DIR / "native.m4a",
    "learner_wav": WAV_DIR / "learner.wav",
    "native_wav": WAV_DIR / "native.wav",
    "learner_transcript": WAV_DIR / "learner.txt",
    "native_transcript": WAV_DIR / "native.txt",
    "script_path": WAV_DIR / "script.txt",
    "learner_textgrid": MFA_OUTPUT_DIR / "learner.TextGrid",
    "native_textgrid": MFA_OUTPUT_DIR / "native.TextGrid",
    "lexicon_path": MFA_LEXICON_PATH,
    "acoustic_model": MFA_ACOUSTIC_MODEL_PATH,
    "mfa_input": MFA_INPUT_DIR,
    "mfa_output": MFA_OUTPUT_DIR,
}

# main.py에서 사용하는 NEW_PATHS (문자열 버전)
NEW_PATHS = {
    "learner_audio": str(INPUT_DIR / "learner.m4a"),
    "native_audio": str(INPUT_DIR / "native.m4a"),
    "learner_wav": str(WAV_DIR / "learner.wav"),
    "native_wav": str(WAV_DIR / "native.wav"),
    "learner_normalized": str(NORMALIZED_DIR / "learner_normalized.wav"),
    "native_normalized": str(NORMALIZED_DIR / "native_normalized.wav"),
    "learner_transcript": str(WAV_DIR / "learner.txt"),
    "native_transcript": str(WAV_DIR / "native.txt"),
    "script_path": str(WAV_DIR / "script.txt"),
    "learner_textgrid": str(MFA_OUTPUT_DIR / "learner.TextGrid"),
    "native_textgrid": str(MFA_OUTPUT_DIR / "native.TextGrid"),
    "lexicon_path": str(MFA_LEXICON_PATH),
    "acoustic_model": str(MFA_ACOUSTIC_MODEL_PATH),
    "mfa_input": str(MFA_INPUT_DIR),
    "mfa_output": str(MFA_OUTPUT_DIR),
    "output_dir": str(OUTPUT_DIR),
    "wav_dir": str(WAV_DIR),
    "normalized_dir": str(NORMALIZED_DIR),
    "temp_dir": str(TEMP_ROOT),
    "visualize_dir": str(VISUALIZE_DIR),
    "knowledge_dir": str(KNOWLEDGE_DIR),
}

# =============================================================================
# 🔧 함수들 (Functions)
# =============================================================================

def create_directories() -> None:
    """필요한 모든 디렉토리 생성"""
    directories = [
        INPUT_DIR,
        OUTPUT_DIR,
        TEMP_ROOT,
        WAV_DIR,
        NORMALIZED_DIR,
        MFA_INPUT_DIR,
        MFA_OUTPUT_DIR,
        VISUALIZE_DIR,
        MODELS_DIR,
        KNOWLEDGE_DIR,
        WHISPER_MODEL_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 디렉토리 생성: {directory}")

def validate_environment() -> list:
    """환경 변수 검증"""
    errors = []
    
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    return errors

def update_config(user_config: Dict[str, Any]) -> None:
    """설정 업데이트"""
    CURRENT_CONFIG.update(user_config)

def get_config() -> Dict[str, Any]:
    """현재 설정 반환"""
    return CURRENT_CONFIG

# =============================================================================
# 🏗️ 호환성 변수들 (Compatibility Variables)
# =============================================================================

# 기존 코드 호환성을 위한 변수들
DEFAULT_CONFIG = CURRENT_CONFIG.copy()
MAIN_CONFIG = CURRENT_CONFIG

# 초기화
create_directories()