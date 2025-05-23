from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()


class Settings(BaseSettings):
    # API 설정
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Koach - Korean Pronunciation Coach"

    # OpenAI 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4")

    # 오디오 설정
    AUDIO_UPLOAD_DIR: str = os.getenv("AUDIO_UPLOAD_DIR", "/tmp/koach/uploads")
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # 데이터베이스 설정
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        case_sensitive = True


# 전역 설정 객체 생성
settings = Settings()
