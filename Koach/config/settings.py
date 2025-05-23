import os
from pathlib import Path
from typing import Dict, Any

# 기본 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
WAV_DIR = TEMP_DIR / "wav"
MFA_INPUT_DIR = TEMP_DIR / "mfa_input"
MFA_OUTPUT_DIR = TEMP_DIR / "mfa_output"
ALIGNED_DIR = TEMP_DIR / "aligned"
MODELS_DIR = BASE_DIR / "models"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# Whisper 모델 경로 추가
WHISPER_MODEL_DIR = MODELS_DIR / "whisper"

# MFA 모델 경로 설정
MFA_LEXICON_PATH = MODELS_DIR / "korean_mfa.dict"
MFA_ACOUSTIC_MODEL_PATH = MODELS_DIR / "korean_mfa.zip"

# 모델 경로 설정
WHISPER_MODEL_PATH = MODELS_DIR / "whisper"
FAISS_INDEX_PATH = MODELS_DIR / "faiss"
SENTENCE_TRANSFORMER_PATH = MODELS_DIR / "sentence_transformer"

# 기본 설정
DEFAULT_CONFIG = {
    "temp_dir": str(TEMP_DIR),
    "input_dir": str(INPUT_DIR),
    "output_dir": str(OUTPUT_DIR),
    "aligned_dir": str(ALIGNED_DIR),
    "mfa_input": str(MFA_INPUT_DIR),
    "mfa_output": str(MFA_OUTPUT_DIR),
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "hop_length": 512,
        "frame_length": 2048,
        "top_db": 30,
    },
    "whisper": {
        "model_name": "base",
        "language": "ko",
        "task": "transcribe",
    },
    "mfa": {
        "model_name": "korean",
        "num_jobs": 4,
        "clean": True,
        "lexicon_path": str(MFA_LEXICON_PATH),
        "acoustic_model": str(MFA_ACOUSTIC_MODEL_PATH),
    },
    "faiss": {
        "dimension": 384,
        "index_type": "L2",
    },
    "sentence_transformer": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 32,
    },
    "visualization": {
        "dpi": 300,
        "figsize": (15, 10),
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
    },
}

# 파일 경로 설정
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


def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        TEMP_DIR,
        INPUT_DIR,
        OUTPUT_DIR,
        WAV_DIR,
        MFA_INPUT_DIR,
        MFA_OUTPUT_DIR,
        ALIGNED_DIR,
        MODELS_DIR,
        KNOWLEDGE_DIR,
        WHISPER_MODEL_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def load_from_env(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """환경 변수에서 설정 로드"""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # 오디오 설정
    if "AUDIO_SAMPLE_RATE" in os.environ:
        config["audio"]["sample_rate"] = int(os.environ["AUDIO_SAMPLE_RATE"])
    if "AUDIO_CHANNELS" in os.environ:
        config["audio"]["channels"] = int(os.environ["AUDIO_CHANNELS"])
    if "AUDIO_HOP_LENGTH" in os.environ:
        config["audio"]["hop_length"] = int(os.environ["AUDIO_HOP_LENGTH"])
    if "AUDIO_FRAME_LENGTH" in os.environ:
        config["audio"]["frame_length"] = int(os.environ["AUDIO_FRAME_LENGTH"])
    if "AUDIO_TOP_DB" in os.environ:
        config["audio"]["top_db"] = float(os.environ["AUDIO_TOP_DB"])

    # Whisper 설정
    if "WHISPER_MODEL_NAME" in os.environ:
        config["whisper"]["model_name"] = os.environ["WHISPER_MODEL_NAME"]
    if "WHISPER_LANGUAGE" in os.environ:
        config["whisper"]["language"] = os.environ["WHISPER_LANGUAGE"]
    if "WHISPER_TASK" in os.environ:
        config["whisper"]["task"] = os.environ["WHISPER_TASK"]

    # MFA 설정
    if "MFA_MODEL_NAME" in os.environ:
        config["mfa"]["model_name"] = os.environ["MFA_MODEL_NAME"]
    if "MFA_NUM_JOBS" in os.environ:
        config["mfa"]["num_jobs"] = int(os.environ["MFA_NUM_JOBS"])
    if "MFA_CLEAN" in os.environ:
        config["mfa"]["clean"] = os.environ["MFA_CLEAN"].lower() == "true"

    # FAISS 설정
    if "FAISS_DIMENSION" in os.environ:
        config["faiss"]["dimension"] = int(os.environ["FAISS_DIMENSION"])
    if "FAISS_INDEX_TYPE" in os.environ:
        config["faiss"]["index_type"] = os.environ["FAISS_INDEX_TYPE"]

    # Sentence Transformer 설정
    if "SENTENCE_TRANSFORMER_MODEL_NAME" in os.environ:
        config["sentence_transformer"]["model_name"] = os.environ[
            "SENTENCE_TRANSFORMER_MODEL_NAME"
        ]
    if "SENTENCE_TRANSFORMER_BATCH_SIZE" in os.environ:
        config["sentence_transformer"]["batch_size"] = int(
            os.environ["SENTENCE_TRANSFORMER_BATCH_SIZE"]
        )

    # 시각화 설정
    if "VISUALIZATION_DPI" in os.environ:
        config["visualization"]["dpi"] = int(os.environ["VISUALIZATION_DPI"])
    if "VISUALIZATION_FIGSIZE" in os.environ:
        width, height = map(int, os.environ["VISUALIZATION_FIGSIZE"].split(","))
        config["visualization"]["figsize"] = (width, height)

    # 로깅 설정
    if "LOG_LEVEL" in os.environ:
        config["logging"]["level"] = os.environ["LOG_LEVEL"]
    if "LOG_FORMAT" in os.environ:
        config["logging"]["format"] = os.environ["LOG_FORMAT"]
    if "LOG_DATE_FORMAT" in os.environ:
        config["logging"]["date_format"] = os.environ["LOG_DATE_FORMAT"]

    return config


# 현재 설정 로드
CURRENT_CONFIG = load_from_env()

# 디렉토리 생성
create_directories()

# export
__all__ = ["CURRENT_CONFIG", "PATHS"]
