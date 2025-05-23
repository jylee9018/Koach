import os
import subprocess
import whisper
from openai import OpenAI
import shutil
from pydub import AudioSegment
import textgrid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Koach")


class Koach:
    """한국어 발음 평가 및 피드백 시스템"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 파라미터 (없으면 기본값 사용)
        """
        # 기본 설정
        self.config = {
            # 파일 경로
            "learner_audio": "input/learner.m4a",
            "native_audio": "input/native.m4a",
            "output_dir": "output",
            "wav_dir": "wav",
            "mfa_input_dir": "mfa_input",
            "mfa_output_dir": "aligned",
            # 모델 경로
            "lexicon_path": "models/korean_mfa.dict",
            "acoustic_model": "models/korean_mfa.zip",
            # Whisper 모델 크기
            "whisper_model": "base",
            # OpenAI 모델
            "openai_model": "gpt-4o",
        }

        # 사용자 설정으로 업데이트
        if config:
            self.config.update(config)

        # OpenAI API 키 설정
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        # 파일 경로 설정
        self._setup_paths()

        # 디렉토리 생성
        self._create_directories()

    def _setup_paths(self):
        """파일 경로 설정"""
        # 입력 파일
        self.learner_audio = self.config["learner_audio"]
        self.native_audio = self.config["native_audio"]

        # 디렉토리
        self.wav_dir = self.config["wav_dir"]
        self.mfa_input = self.config["mfa_input_dir"]
        self.mfa_output = self.config["mfa_output_dir"]

        # 변환된 WAV 파일
        self.learner_wav = os.path.join(self.wav_dir, "learner.wav")
        self.native_wav = os.path.join(self.wav_dir, "native.wav")

        # Whisper 전사 결과
        self.learner_transcript = os.path.join(self.wav_dir, "learner.txt")
        self.native_transcript = os.path.join(self.wav_dir, "native.txt")

        # 정답 스크립트
        self.script_path = os.path.join(self.wav_dir, "script.txt")

        # MFA 관련 파일
        self.lexicon_path = self.config["lexicon_path"]
        self.acoustic_model = self.config["acoustic_model"]

        # TextGrid 파일
        self.learner_textgrid = os.path.join(self.mfa_output, "learner.TextGrid")
        self.native_textgrid = os.path.join(self.mfa_output, "native.TextGrid")

    def _create_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(self.wav_dir, exist_ok=True)
        os.makedirs(self.mfa_input, exist_ok=True)
        os.makedirs(self.mfa_output, exist_ok=True)

    def convert_audio(self, input_path: str, output_path: str) -> bool:
        """오디오 파일을 WAV 형식으로 변환

        Args:
            input_path: 입력 오디오 파일 경로
            output_path: 출력 WAV 파일 경로

        Returns:
            bool: 변환 성공 여부
        """
        try:
            logger.info(f"🎧 변환 중: {input_path} → {output_path}")
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_path, format="wav")
            logger.info("✅ 오디오 변환 완료")
            return True
        except Exception as e:
            logger.error(f"오디오 변환 실패: {e}")
            return False

    def transcribe_audio(self, wav_path: str, transcript_path: str) -> Optional[str]:
        """Whisper를 사용하여 오디오 파일 전사

        Args:
            wav_path: WAV 파일 경로
            transcript_path: 전사 결과 저장 경로

        Returns:
            Optional[str]: 전사 텍스트 (실패 시 None)
        """
        try:
            logger.info(f"📝 Whisper 전사 중: {wav_path}")
            model = whisper.load_model(self.config["whisper_model"])
            result = model.transcribe(wav_path, language="ko")

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            logger.info(f"📄 전사 결과: {result['text']}")
            return result["text"]
        except Exception as e:
            logger.error(f"전사 실패: {e}")
            return None

    # def transcribe_audio(self, wav_path: str, transcript_path: str) -> Optional[str]:
    #     """Google Cloud Speech-to-Text를 사용하여 오디오 파일 전사

    #     Args:
    #         wav_path: WAV 파일 경로
    #         transcript_path: 전사 결과 저장 경로

    #     Returns:
    #         Optional[str]: 전사 텍스트 (실패 시 None)
    #     """
    #     try:
    #         logger.info(f"📝 Google STT 전사 중: {wav_path}")

    #         # 인증 정보 파일 경로
    #         credentials_path = self.config.get(
    #             "google_credentials", "/Users/jlee/Keys/my-credentials.json"
    #         )

    #         # Google Cloud 클라이언트 초기화 (명시적 인증)
    #         from google.cloud import speech
    #         from google.oauth2 import service_account

    #         credentials = service_account.Credentials.from_service_account_file(
    #             credentials_path
    #         )
    #         client = speech.SpeechClient(credentials=credentials)

    #         # 오디오 파일 읽기
    #         with open(wav_path, "rb") as audio_file:
    #             content = audio_file.read()

    #         audio = speech.RecognitionAudio(content=content)
    #         config = speech.RecognitionConfig(
    #             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #             sample_rate_hertz=16000,
    #             language_code="ko-KR",
    #             enable_automatic_punctuation=True,
    #         )

    #         # 전사 요청
    #         response = client.recognize(config=config, audio=audio)

    #         # 결과 처리
    #         transcript = ""
    #         for result in response.results:
    #             transcript += result.alternatives[0].transcript

    #         # 결과 저장
    #         with open(transcript_path, "w", encoding="utf-8") as f:
    #             f.write(transcript)

    #         logger.info(f"📄 전사 결과: {transcript}")
    #         return transcript
    #     except Exception as e:
    #         logger.error(f"전사 실패: {e}")
    #         return None

    def run_mfa_alignment(self, wav_path: str, transcript_path: str, output_name: str) -> bool:
        """MFA를 사용하여 오디오와 텍스트 정렬

        Args:
            wav_path: WAV 파일 경로
            transcript_path: 전사 텍스트 파일 경로
            output_name: 출력 파일 이름 (확장자 제외)

        Returns:
            bool: 정렬 성공 여부
        """
        try:
            logger.info(f"🔧 MFA 정렬 시작: {output_name}")

            # 파일 복사
            shutil.copy(wav_path, os.path.join(self.mfa_input, f"{output_name}.wav"))
            shutil.copy(
                transcript_path, os.path.join(self.mfa_input, f"{output_name}.txt")
            )

            # MFA 정렬 실행
            command = [
     
                "align",
                self.mfa_input,
        lexicon_path,
                self.acoustic_model,
                self.mfa_output,
                "--clean",
                "--no_text_cleaning",
            ]

            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                logger.error(f"MFA 정렬 실패: {result.stderr}")
                return False

            logger.info("✅ MFA 정렬 완료")
            return True
        except Exception as e:
            logger.error(f"MFA 정렬 실패: {e}")
            return False

    def summarize_textgrid(self, path: str) -> Optional[str]:
       """TextGrid 파일에서 음소 정보 추출

        Args:
            path: TextGrid 파일 경로

        Returns:
            Optional[str]: 음소 정보 요약 (실패 시 None)
        """
        try:
            logger.info(f"📊 TextGrid 요약 중: {path}")
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
        except Exception as e:
            logger.error(f"TextGrid 요약 실패: {e}")
            return None

    def generate_prompt(
        self,
        learner_text: str,
        native_text: str,
        script_text: str,
        learner_timing: str,
        native_timing: str,
    ) -> str:
        """GPT 프롬프트 생성

        Args:
            learner_text: 학습자 발화 텍스트
            native_text: 원어민 발화 텍스트
            script_text: 목표 스크립트
            learner_timing: 학습자 음소 정렬 정보
            native_timing: 원어민 음소 정렬 정보

        Returns:
            str: GPT 프롬프트
        """
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
   - 구체적으로 제시

2. 학습자의 발음에서 부적절하게 띄어 읽은 단어나 음소는 무엇인가?  
   - 꼭 해당하는 **단어나 음소**를 함께 제시

3. 원어민과 비교했을 때 어떤 **단어나 구절에서** 속도 차이가 있는가?  
   - 속도 정보를 제시할 때는 꼭 해당하는 **단어나 음소**를 함께 제시

4. 더 자연스럽고 명확하게 발음하기 위한 팁을 간단히 제시
"""
        return prompt

    def get_feedback(self, prompt: str) -> Optional[str]:
        """OpenAI API를 사용하여 피드백 생성

        Args:
            prompt: GPT 프롬프트

        Returns:
            Optional[str]: 생성된 피드백 (실패 시 None)
        """
        try:
            logger.info("🤖 GPT 피드백 생성 중...")

            if not self.api_key:
                logger.error("OpenAI API 키가 설정되지 않았습니다.")
                return None

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.config["openai_model"],
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 친절한 한국어 발음 강사입니다. 학습자가 외국인임을 고려하여 쉬운 문법 용어로 설명해주세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"피드백 생성 실패: {e}")
            return None

    def analyze_pronunciation(
        self,
        learner_audio: Optional[str] = None,
        native_audio: Optional[str] = None,
        script: Optional[str] = None,
    ) -> Dict:
        """발음 분석 전체 파이프라인 실행

        Args:
            learner_audio: 학습자 오디오 파일 경로 (기본값 사용 시 None)
            native_audio: 원어민 오디오 파일 경로 (기본값 사용 시 None)
            script: 목표 스크립트 (없으면 파일에서 로드)

        Returns:
            Dict: 분석 결과 및 상태
        """
        result = {
            "success": False,
            "feedback": None,
            "error": None,
            "learner_text": None,
            "native_text": None,
            "script_text": None,
        }

        try:
            # 입력 파일 설정
            if learner_audio:
                self.learner_audio = learner_audio
            if native_audio:
                self.native_audio = native_audio

            # 1. 오디오 변환
            if not self.convert_audio(self.learner_audio, self.learner_wav):
                result["error"] = "학습자 오디오 변환 실패"
                return result

            if not self.convert_audio(self.native_audio, self.native_wav):
                result["error"] = "원어민 오디오 변환 실패"
                return result

            # 2. Whisper 전사
            learner_text = self.transcribe_audio(
                self.learner_wav, self.learner_transcript
            )
            if not learner_text:
                result["error"] = "학습자 오디오 전사 실패"
                return result

            native_text = self.transcribe_audio(self.native_wav, self.native_transcript)
            if not native_text:
                result["error"] = "원어민 오디오 전사 실패"
                return result

            # 결과에 전사 텍스트 저장
            result["learner_text"] = learner_text
            result["native_text"] = native_text

            # 3. 스크립트 로딩 또는 설정
            if script:
                script_text = script
                # 스크립트 파일에도 저장
                with open(self.script_path, "w", encoding="utf-8") as f:
                    f.write(script_text)
            else:
                try:
                    with open(self.script_path, "r", encoding="utf-8") as f:
                        script_text = f.read().strip()
                except FileNotFoundError:
                    # 스크립트 파일이 없으면 원어민 전사를 사용
                    script_text = native_text
                    with open(self.script_path, "w", encoding="utf-8") as f:
                        f.write(script_text)

            result["script_text"] = script_text

            # 4. MFA 정렬
            if not self.run_mfa_alignment(
                self.learner_wav, self.learner_transcript, "learner"
            ):
                result["error"] = "학습자 오디오 정렬 실패"
                return result

            if not self.run_mfa_alignment(
                self.native_wav, self.native_transcript, "native"
            ):
                result["error"] = "원어민 오디오 정렬 실패"
                return result

            # 5. TextGrid 요약
            learner_timing = self.summarize_textgrid(self.learner_textgrid)
            if not learner_timing:
                result["error"] = "학습자 TextGrid 요약 실패"
                return result

            native_timing = self.summarize_textgrid(self.native_textgrid)
            if not native_timing:
                result["error"] = "원어민 TextGrid 요약 실패"
                return result

            # 6. 프롬프트 생성
            prompt = self.generate_prompt(
                learner_text, native_text, script_text, learner_timing, native_timing
            )

            # 7. GPT 피드백 생성
            feedback = self.get_feedback(prompt)
            if not feedback:
                result["error"] = "피드백 생성 실패"
                return result

            # 성공 결과 반환
            result["success"] = True
            result["feedback"] = feedback

            return result

        except Exception as e:
            logger.error(f"발음 분석 실패: {e}")
            result["error"] = str(e)
            return result


def main():
    """메인 함수"""
    try:
        # 코치 초기화
        koach = Koach()

        # 발음 분석 실행
        result = koach.analyze_pronunciation()

        # 결과 출력
        if result["success"]:
            print("\n📣 발음 피드백 결과:\n")
            print(result["feedback"])
        else:
            print(f"\n❌ 오류 발생: {result['error']}")

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
