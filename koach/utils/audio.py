import os
import subprocess
import logging
from typing import Optional, Tuple, Dict, Any
import whisper
import numpy as np
import soundfile as sf

from config.settings import CURRENT_CONFIG, WHISPER_MODEL_DIR

logger = logging.getLogger("Koach")

# Whisper 모델 다운로드 위치 설정
os.environ["WHISPER_MODEL_DIR"] = str(WHISPER_MODEL_DIR)


def convert_audio(
    input_path: str,
    output_path: str,
    target_format: str = "wav",
    sample_rate: int = None,
    channels: int = None,
) -> bool:
    """오디오 파일을 지정된 형식으로 변환"""
    try:
        if sample_rate is None:
            sample_rate = CURRENT_CONFIG["audio"]["sample_rate"]
        if channels is None:
            channels = CURRENT_CONFIG["audio"]["channels"]

        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ffmpeg 명령어 구성
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-y",  # 기존 파일 덮어쓰기
            output_path,
        ]

        # 변환 실행
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"오디오 변환 실패: {result.stderr}")
            return False

        return True

    except Exception as e:
        logger.error(f"오디오 변환 중 오류 발생: {e}")
        return False


def transcribe_audio(
    audio_path: str,
    model_name: str = None,
    language: str = "ko",
) -> Dict[str, Any]:
    """Whisper 모델을 사용하여 오디오 파일을 텍스트로 변환"""
    try:
        if model_name is None:
            model_name = CURRENT_CONFIG["whisper"]["model_name"]

        # Whisper 모델 로드
        model = whisper.load_model(model_name)

        # 오디오 파일 로드 및 전처리
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # 오디오 특성 추출
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # 디코딩 옵션 설정
        decode_options = {
            "language": language,
            "task": "transcribe",
            "word_timestamps": True,  # 단어 단위 타임스탬프 활성화
        }

        # 텍스트 변환
        result = model.transcribe(audio_path, **decode_options)

        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
            "words": result.get("words", []),  # 단어 정보 추가
        }

    except Exception as e:
        logger.error(f"음성 인식 중 오류 발생: {e}")
        return {}


def extract_audio_segment(
    audio_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
) -> bool:
    """오디오 파일에서 특정 구간 추출"""
    try:
        # ffmpeg 명령어 구성
        command = [
            "ffmpeg",
            "-i",
            audio_path,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-y",  # 기존 파일 덮어쓰기
            output_path,
        ]

        # 구간 추출 실행
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"오디오 구간 추출 실패: {result.stderr}")
            return False

        return True

    except Exception as e:
        logger.error(f"오디오 구간 추출 중 오류 발생: {e}")
        return False


def normalize_audio(
    input_path: str,
    output_path: str,
    target_level: float = -23.0,  # LUFS
) -> bool:
    """오디오 파일의 볼륨 정규화"""
    try:
        # ffmpeg 명령어 구성
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-af",
            f"loudnorm=I={target_level}:LRA=11:TP=-1.5",
            "-y",  # 기존 파일 덮어쓰기
            output_path,
        ]

        # 정규화 실행
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"오디오 정규화 실패: {result.stderr}")
            return False

        return True

    except Exception as e:
        logger.error(f"오디오 정규화 중 오류 발생: {e}")
        return False


def get_audio_duration(audio_path: str) -> float:
    """오디오 파일의 길이(초) 반환"""
    try:
        # ffprobe 명령어 구성
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]

        # 길이 확인 실행
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"오디오 길이 확인 실패: {result.stderr}")
            return 0.0

        return float(result.stdout.strip())

    except Exception as e:
        logger.error(f"오디오 길이 확인 중 오류 발생: {e}")
        return 0.0


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    """오디오 파일의 상세 정보 반환"""
    try:
        # ffprobe 명령어 구성
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,channels,sample_rate,bit_rate",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            audio_path,
        ]

        # 정보 확인 실행
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"오디오 정보 확인 실패: {result.stderr}")
            return {}

        # JSON 파싱
        import json

        info = json.loads(result.stdout)

        # 정보 구성
        stream = info.get("streams", [{}])[0]
        format_info = info.get("format", {})

        return {
            "codec_name": stream.get("codec_name"),
            "channels": int(stream.get("channels", 0)),
            "sample_rate": int(stream.get("sample_rate", 0)),
            "bit_rate": int(stream.get("bit_rate", 0)),
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
        }

    except Exception as e:
        logger.error(f"오디오 정보 확인 중 오류 발생: {e}")
        return {}
