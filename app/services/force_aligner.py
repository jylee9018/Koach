import os
import json
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ForceAligner:
    """
    한국어 음성과 텍스트를 정렬하여 음소 단위 시간 정보를 추출하는 클래스
    Montreal Forced Aligner(MFA)를 사용합니다.
    """

    def __init__(
        self,
        model_path: str = None,
        dict_path: str = None,
        use_gentle_fallback: bool = True,
    ):
        """
        Args:
            model_path: 학습된 한국어 음향 모델 경로
            dict_path: 한국어 발음 사전 경로
            use_gentle_fallback: MFA 실패 시 Gentle을 대체로 사용할지 여부
        """
        # 환경 변수에서 경로 가져오기 (기본값은 MFA 기본 설치 경로)
        default_model_path = os.path.expanduser(
            "~/Documents/MFA/pretrained_models/acoustic/korean_mfa"
        )
        default_dict_path = os.path.expanduser(
            "~/Documents/MFA/pretrained_models/dictionary/korean_mfa.dict"
        )

        self.model_path = model_path or os.environ.get(
            "MFA_KOREAN_MODEL_PATH", default_model_path
        )
        self.dict_path = dict_path or os.environ.get(
            "MFA_KOREAN_DICT_PATH", default_dict_path
        )
        self.use_gentle_fallback = use_gentle_fallback

        # 모델 경로 로깅
        logger.info(f"Using MFA model path: {self.model_path}")
        logger.info(f"Using MFA dictionary path: {self.dict_path}")

        # 필요한 디렉토리 생성
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)

        # MFA 설치 확인
        self._check_mfa_installation()

        # Gentle 설치 확인 (fallback으로 사용하는 경우)
        if use_gentle_fallback:
            self._check_gentle_installation()

        # Gentle 경로 설정
        self.gentle_path = os.environ.get(
            "GENTLE_PATH", os.path.join(os.path.expanduser("~"), "gentle")
        )
        if not os.path.exists(self.gentle_path):
            logger.warning(
                f"Gentle not found at {self.gentle_path}. Set GENTLE_PATH env variable."
            )

    def _check_mfa_installation(self):
        """MFA가 설치되어 있는지 확인"""
        try:
            result = subprocess.run(["mfa", "version"], capture_output=True, text=True)
            logger.info(f"MFA version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.warning("MFA not found. Please install Montreal Forced Aligner.")
            logger.info(
                "Installation guide: https://montreal-forced-aligner.readthedocs.io/"
            )

    def _check_gentle_installation(self):
        """Gentle이 설치되어 있는지 확인"""
        try:
            # Gentle은 일반적으로 로컬에서 서버로 실행되므로 설치 경로만 확인
            gentle_path = os.environ.get("GENTLE_PATH", "/opt/gentle")
            if os.path.exists(gentle_path):
                logger.info(f"Gentle found at: {gentle_path}")
            else:
                logger.warning(
                    f"Gentle not found at {gentle_path}. Set GENTLE_PATH env variable."
                )
        except Exception as e:
            logger.warning(f"Error checking Gentle installation: {e}")

    async def align(self, audio_path: str, text: str) -> dict:
        """오디오와 텍스트를 정렬합니다."""
        logger.info(f"MFA version: {self.get_mfa_version()}")

        # 한국어 사전 및 모델 확인
        dict_path = self.dict_path
        model_path = self.model_path

        # 사전 파일 확인
        if not os.path.exists(dict_path):
            logger.warning(f"한국어 사전 파일이 없습니다: {dict_path}")
            # 가능한 경로 확인
            possible_dict_paths = [
                os.path.expanduser(
                    "~/Documents/MFA/pretrained_models/dictionary/korean_mfa.dict"
                ),
                os.path.expanduser("~/.mfa/dictionary/korean_mfa.dict"),
            ]
            for path in possible_dict_paths:
                if os.path.exists(path):
                    logger.info(f"사전 파일을 찾았습니다: {path}")
                    self.dict_path = dict_path = path
                    break
            else:
                logger.error("사전 파일을 찾을 수 없습니다. 다운로드를 시도하세요.")
                raise FileNotFoundError("한국어 사전 파일을 찾을 수 없습니다.")

        # 음향 모델 확인
        if not os.path.exists(model_path):
            logger.warning(f"한국어 음향 모델이 없습니다: {model_path}")
            # 가능한 경로 확인
            possible_model_paths = [
                os.path.expanduser(
                    "~/Documents/MFA/pretrained_models/acoustic/korean_mfa"
                ),
                os.path.expanduser("~/.mfa/acoustic/korean_mfa"),
            ]
            for path in possible_model_paths:
                if os.path.exists(path):
                    logger.info(f"음향 모델을 찾았습니다: {path}")
                    self.model_path = model_path = path
                    break
            else:
                logger.error("음향 모델을 찾을 수 없습니다. 다운로드를 시도하세요.")
                raise FileNotFoundError("한국어 음향 모델을 찾을 수 없습니다.")

        # MFA로 정렬 시도
        try:
            alignment_result = await self._align_with_mfa(audio_path, text)
            return alignment_result
        except Exception as mfa_error:
            logger.error(f"MFA alignment failed: {mfa_error}")

            # Gentle 백업 사용
            if self.use_gentle_fallback:
                logger.info("Trying with Gentle as fallback...")
                try:
                    alignment_result = await self._align_with_gentle(audio_path, text)
                    return alignment_result
                except Exception as gentle_error:
                    logger.error(f"Gentle alignment also failed: {gentle_error}")
                    raise Exception(
                        f"Both MFA and Gentle alignment failed: {mfa_error}, {gentle_error}"
                    )
            else:
                raise Exception(f"MFA alignment failed: {mfa_error}")

    async def _align_with_mfa(self, audio_path: str, text: str) -> Dict:
        """MFA를 사용한 정렬"""
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = os.path.join(temp_dir, "corpus")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(corpus_dir, exist_ok=True)

            # 오디오 파일 복사
            audio_filename = os.path.basename(audio_path)
            audio_name = os.path.splitext(audio_filename)[0]
            shutil.copy(audio_path, os.path.join(corpus_dir, audio_filename))

            # 텍스트 파일 생성
            text_path = os.path.join(corpus_dir, f"{audio_name}.lab")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            # MFA 실행
            cmd = [
                "mfa",
                "align",
                corpus_dir,
                self.dict_path,
                self.model_path,
                output_dir,
                "--clean",
                "--overwrite",
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode != 0:
                raise Exception(f"MFA failed: {process.stderr}")

            # TextGrid 파일 읽기
            textgrid_path = os.path.join(output_dir, f"{audio_name}.TextGrid")
            if not os.path.exists(textgrid_path):
                raise Exception(f"TextGrid file not found: {textgrid_path}")

            # TextGrid를 JSON으로 변환
            alignment_result = self._parse_textgrid(textgrid_path)
            return alignment_result

    async def _align_with_gentle(self, audio_path: str, text: str) -> Dict:
        """Gentle을 사용한 정렬 (MFA 실패 시 대체)"""
        # Gentle 서버 URL (로컬에서 실행 중이라고 가정)
        gentle_url = os.environ.get(
            "GENTLE_URL", "http://localhost:8765/transcriptions"
        )

        # 외부 프로세스로 Gentle 실행 (실제 구현에서는 HTTP 요청으로 대체 가능)
        gentle_path = os.environ.get("GENTLE_PATH", "/opt/gentle")
        align_script = os.path.join(gentle_path, "align.py")

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_text:
            temp_text.write(text.encode("utf-8"))
            temp_text_path = temp_text.name

        try:
            cmd = [
                "python",
                align_script,
                audio_path,
                temp_text_path,
                "--output",
                "alignment.json",
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode != 0:
                raise Exception(f"Gentle failed: {process.stderr}")

            # 결과 JSON 파일 읽기
            with open("alignment.json", "r", encoding="utf-8") as f:
                gentle_result = json.load(f)

            # Gentle 결과를 표준 형식으로 변환
            alignment_result = self._convert_gentle_result(gentle_result)
            return alignment_result
        finally:
            # 임시 파일 정리
            if os.path.exists(temp_text_path):
                os.remove(temp_text_path)
            if os.path.exists("alignment.json"):
                os.remove("alignment.json")

    def _parse_textgrid(self, textgrid_path: str) -> Dict:
        """TextGrid 파일 파싱"""
        try:
            # textgrid 라이브러리 사용
            import textgrid

            tg = textgrid.TextGrid.fromFile(textgrid_path)

            words = []
            phonemes = []

            # 단어 계층 처리
            word_tier = None
            for tier in tg.tiers:
                if tier.name.lower() == "words":
                    word_tier = tier
                    break

            if word_tier:
                for interval in word_tier:
                    if interval.mark:  # 비어있지 않은 간격만
                        words.append(
                            {
                                "word": interval.mark,
                                "start": interval.minTime,
                                "end": interval.maxTime,
                                "aligned": True,
                            }
                        )

            # 음소 계층 처리
            phone_tier = None
            for tier in tg.tiers:
                if tier.name.lower() == "phones":
                    phone_tier = tier
                    break

            if phone_tier:
                for interval in phone_tier:
                    if interval.mark:  # 비어있지 않은 간격만
                        # 특수 기호 제거
                        phoneme = interval.mark.split("_")[0]
                        phonemes.append(
                            {
                                "label": phoneme,
                                "start": interval.minTime,
                                "end": interval.maxTime,
                                "aligned": True,
                            }
                        )

            return {
                "words": words,
                "phonemes": phonemes,
                "success": True,
                "source": "mfa",
            }
        except Exception as e:
            import logging

            logging.error(f"TextGrid 파싱 오류: {e}")
            raise

    def _convert_gentle_result(self, gentle_result: Dict) -> Dict:
        """Gentle 결과를 표준 형식으로 변환"""
        words = []
        phonemes = []

        for word in gentle_result.get("words", []):
            word_entry = {
                "word": word.get("word", ""),
                "start": word.get("start", 0),
                "end": word.get("end", 0),
                "aligned": word.get("case") == "success",
            }
            words.append(word_entry)

            # Gentle은 단어 내 음소 정보도 제공
            for phone in word.get("phones", []):
                phoneme_entry = {
                    "label": phone.get("phone", "").split("_")[0],  # 특수 기호 제거
                    "start": phone.get("start", 0),
                    "end": phone.get("end", 0),
                    "aligned": True,  # Gentle에서는 음소 수준 정렬 성공 여부를 명시적으로 제공하지 않음
                }
                phonemes.append(phoneme_entry)

        return {
            "words": words,
            "phonemes": phonemes,
            "success": True,
            "source": "gentle",
        }

    def get_mfa_version(self):
        """MFA 버전 정보를 반환합니다."""
        try:
            # 최신 MFA 버전에서는 --version 대신 version 명령어 사용
            result = subprocess.run(["mfa", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()

            # 이전 버전 시도
            result = subprocess.run(
                ["mfa", "--version"], capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"MFA 버전 확인 실패: {e}")
            return "Unknown"
