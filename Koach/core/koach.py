import os
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil
import subprocess
import librosa
import textgrid
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

from utils.audio import (
    convert_audio,
    transcribe_audio,
    extract_audio_segment,
    normalize_audio,
    get_audio_duration,
    get_audio_info,
)
from utils.text import (
    summarize_textgrid,
    extract_word_boundaries,
    align_text_with_audio,
    extract_phoneme_features,
    compare_phoneme_sequences,
)
from core.prosody import ProsodyAnalyzer
from core.knowledge_base import KnowledgeBase
from config.settings import CURRENT_CONFIG, PATHS

logger = logging.getLogger(__name__)


class Koach:
    """한국어 발음 교정 도우미"""

    def __init__(self, config: Optional[Dict] = None):
        """초기화

        Args:
            config: 사용자 설정
        """
        # 상위 폴더의 .env 파일 로드
        env_path = Path(__file__).parent.parent.parent / '.env'
        load_dotenv(env_path)
        
        # API 키 확인
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        # 출력 디렉토리 설정
        self.output_dir = Path(__file__).parent.parent / "output"
        self.output_dir.mkdir(exist_ok=True)

        # 임시 디렉토리 설정
        self.temp_dir = Path(__file__).parent.parent / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        # 기본 설정
        self.config = {
            # Whisper 모델 설정
            "whisper_model": "base",
            "language": "ko",
            # OpenAI 모델
            "openai_model": "gpt-4o",
            # RAG 설정
            "use_rag": True,
            "knowledge_dir": "knowledge",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        }

        # 사용자 설정으로 업데이트
        if config:
            self.config.update(config)

        # 파일 경로 설정 (settings.py의 PATHS 사용)
        self.learner_audio = PATHS["learner_audio"]
        self.native_audio = PATHS["native_audio"]
        self.learner_wav = PATHS["learner_wav"]
        self.native_wav = PATHS["native_wav"]
        self.learner_transcript = PATHS["learner_transcript"]
        self.native_transcript = PATHS["native_transcript"]
        self.script_path = PATHS["script_path"]
        self.learner_textgrid = PATHS["learner_textgrid"]
        self.native_textgrid = PATHS["native_textgrid"]
        self.lexicon_path = PATHS["lexicon_path"]
        self.acoustic_model = PATHS["acoustic_model"]
        self.mfa_input = PATHS["mfa_input"]
        self.mfa_output = PATHS["mfa_output"]

        # RAG 지식 베이스 초기화 (설정에 따라)
        if self.config["use_rag"]:
            self.knowledge_base = KnowledgeBase(
                knowledge_dir=self.config["knowledge_dir"],
                embedding_model=self.config["embedding_model"],
            )
        else:
            self.knowledge_base = None

        self.model_name = "base"
        self.prosody_analyzer = ProsodyAnalyzer()

    def analyze_pronunciation(
        self,
        learner_audio: Optional[str] = None,
        native_audio: Optional[str] = None,
        script: Optional[str] = None,
        visualize: bool = True,
    ) -> Dict:
        """발음 분석 전체 파이프라인 실행

        Args:
            learner_audio: 학습자 오디오 파일 경로
            native_audio: 원어민 오디오 파일 경로
            script: 스크립트 (선택사항)
            visualize: 시각화 여부

        Returns:
            Dict: 분석 결과
        """
        result = {
            "learner": {},
            "native": {},
            "comparison": {},
            "feedback": None
        }

        try:
            # 1. 오디오 파일 변환 (.m4a -> .wav)
            if learner_audio:
                self.learner_wav = self.convert_audio(learner_audio)
                if not self.learner_wav:
                    result["error"] = "학습자 오디오 변환 실패"
                    return result

            if native_audio:
                self.native_wav = self.convert_audio(native_audio)
                if not self.native_wav:
                    result["error"] = "원어민 오디오 변환 실패"
                    return result

            # 2. Whisper로 음성 인식
            if self.learner_wav:
                learner_result = transcribe_audio(self.learner_wav)
                if not learner_result:
                    result["error"] = "학습자 음성 인식 실패"
                    return result
                result["learner"]["transcription"] = learner_result

            if self.native_wav:
                native_result = transcribe_audio(self.native_wav)
                if not native_result:
                    result["error"] = "원어민 음성 인식 실패"
                    return result
                result["native"]["transcription"] = native_result

            # 3. MFA 정렬
            if self.learner_wav and learner_result:
                learner_timing = self.align_audio(
                    self.learner_wav,
                    learner_result["text"],
                    "learner"
                )
                if not learner_timing:
                    result["error"] = "학습자 정렬 실패"
                    return result
                result["learner"]["timing"] = learner_timing

            if self.native_wav and native_result:
                native_timing = self.align_audio(
                    self.native_wav,
                    native_result["text"],
                    "native"
                )
                if not native_timing:
                    result["error"] = "원어민 정렬 실패"
                    return result
                result["native"]["timing"] = native_timing

            # 4. 발음 문제점 추출
            if learner_result and native_result:
                issues = self.extract_pronunciation_issues(
                    learner_result,
                    native_result,
                    learner_timing,
                    native_timing
                )
                result["comparison"]["issues"] = issues

            # 5. 억양/강세 분석
            if self.learner_wav and self.native_wav:
                prosody_result = self.analyze_prosody(
                    self.learner_wav,
                    self.native_wav,
                    learner_text=learner_result["text"],
                    learner_timing=learner_timing,
                    visualize=visualize
                )
                if not prosody_result:
                    result["error"] = "억양/강세 분석 실패"
                    return result
                result["comparison"]["prosody"] = prosody_result

            # 6. LLM 피드백 생성
            if result["comparison"] and learner_result and native_result:
                # TextGrid 요약 생성
                learner_textgrid_path = self.mfa_output / "learner" / f"{Path(self.learner_wav).stem}.TextGrid"
                native_textgrid_path = self.mfa_output / "native" / f"{Path(self.native_wav).stem}.TextGrid"
                
                learner_timing_summary = self.summarize_textgrid(str(learner_textgrid_path))
                native_timing_summary = self.summarize_textgrid(str(native_textgrid_path))
                
                if learner_timing_summary and native_timing_summary:
                    # 상세한 프롬프트 생성
                    prompt = self.generate_detailed_prompt(
                        learner_result["text"],
                        native_result["text"],
                        native_result["text"],  # 스크립트로 원어민 텍스트 사용
                        learner_timing_summary,
                        native_timing_summary,
                        result["comparison"].get("prosody")
                    )
                    
                    # OpenAI API로 피드백 생성
                    detailed_feedback = self.get_feedback(prompt)
                    
                    if detailed_feedback:
                        result["feedback"] = {
                            "summary": "상세한 발음 분석이 완료되었습니다.",
                            "detailed_analysis": detailed_feedback,
                            "prompt_used": prompt
                        }
                    else:
                        # 대체 피드백
                        result["feedback"] = self._generate_simple_feedback(result)
                else:
                    # 대체 피드백
                    result["feedback"] = self._generate_simple_feedback(result)

            # 성공 시 메타데이터 추가
            result["success"] = True
            result["timestamp"] = __import__("datetime").datetime.now().isoformat()
            result["processing_info"] = {
                "whisper_model": self.config.get("whisper_model", "base"),
                "openai_model": self.config.get("openai_model", "gpt-4o"),
                "rag_enabled": self.config.get("use_rag", False),
                "visualization_enabled": visualize
            }
            
            return result

        except Exception as e:
            logger.error(f"발음 분석 실패: {e}")
            result["error"] = str(e)
            result["timestamp"] = __import__("datetime").datetime.now().isoformat()
            return result

    def run_mfa_alignment(
        self, wav_path: str, transcript_path: str, output_name: str
    ) -> bool:
        """MFA를 사용하여 오디오와 텍스트 정렬"""
        try:
            logger.info(f"🔧 MFA 정렬 시작: {output_name}")

            # MFA 입력 디렉토리 준비
            mfa_input_dir = self.mfa_input / output_name
            mfa_input_dir.mkdir(parents=True, exist_ok=True)

            # 파일 복사 (경로를 문자열로 변환하여 비교)
            target_wav = str(mfa_input_dir / f"{output_name}.wav")
            target_txt = str(mfa_input_dir / f"{output_name}.txt")

            if str(wav_path) != target_wav:
                shutil.copy(wav_path, target_wav)
            if str(transcript_path) != target_txt:
                shutil.copy(transcript_path, target_txt)

            # MFA 정렬 실행
            command = [
                "mfa",
                "align",
                str(mfa_input_dir),
                str(self.lexicon_path),
                str(self.acoustic_model),
                str(self.mfa_output),
                "--clean",
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logger.error(f"MFA 정렬 실패: {result.stderr}")
                return False

            logger.info("✅ MFA 정렬 완료")
            return True

        except Exception as e:
            logger.error(f"MFA 정렬 실패: {e}")
            return False

    def _analyze_phonemes(
        self,
        textgrid_path: str,
    ) -> Optional[Dict[str, Any]]:
        """음소 분석

        Args:
            textgrid_path: TextGrid 파일 경로

        Returns:
            Optional[Dict[str, Any]]: 음소 분석 결과
        """
        try:
            # TextGrid 요약
            summary = self.summarize_textgrid(textgrid_path)

            # 단어 경계 추출
            word_boundaries = extract_word_boundaries(textgrid_path)

            # 음소 특징 추출
            phoneme_features = extract_phoneme_features(textgrid_path)

            return {
                "summary": summary,
                "word_boundaries": word_boundaries,
                "phoneme_features": phoneme_features,
            }

        except Exception as e:
            logger.error(f"음소 분석 중 오류가 발생했습니다: {str(e)}")
            return None

    def _compare_with_reference(
        self,
        learner_audio: str,
        reference_audio: str,
        learner_textgrid: str,
    ) -> Optional[Dict[str, Any]]:
        """참조 오디오와 비교

        Args:
            learner_audio: 학습자 오디오 파일 경로
            reference_audio: 참조 오디오 파일 경로
            learner_textgrid: 학습자 TextGrid 파일 경로

        Returns:
            Optional[Dict[str, Any]]: 비교 결과
        """
        try:
            # 참조 오디오 처리
            ref_path = Path(reference_audio)
            ref_wav = self.temp_dir / f"{ref_path.stem}.wav"

            if not convert_audio(
                reference_audio,
                str(ref_wav),
                sample_rate=CURRENT_CONFIG["audio"]["sample_rate"],
                channels=CURRENT_CONFIG["audio"]["channels"],
            ):
                raise RuntimeError("참조 오디오 변환에 실패했습니다.")

            # 참조 오디오 MFA 정렬
            ref_mfa_output = self.aligned_dir / ref_path.stem
            if not self.run_mfa_alignment(str(ref_wav), str(ref_wav), ref_path.stem):
                raise RuntimeError("참조 오디오 MFA 정렬에 실패했습니다.")

            ref_textgrid = ref_mfa_output / f"{ref_path.stem}.TextGrid"

            # 음소 시퀀스 비교
            phoneme_comparison = compare_phoneme_sequences(
                learner_textgrid,
                str(ref_textgrid),
            )

            # 운율 비교
            prosody_comparison = self.prosody_analyzer.compare_prosody(
                learner_audio,
                str(ref_wav),
            )

            return {
                "phoneme_comparison": phoneme_comparison,
                "prosody_comparison": prosody_comparison,
            }

        except Exception as e:
            logger.error(f"참조 오디오 비교 중 오류가 발생했습니다: {str(e)}")
            return None

    def _generate_feedback(
        self,
        phoneme_analysis: Dict[str, Any],
        prosody_analysis: Dict[str, Any],
        comparison: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """피드백 생성

        Args:
            phoneme_analysis: 음소 분석 결과
            prosody_analysis: 운율 분석 결과
            comparison: 참조 오디오 비교 결과 (선택사항)

        Returns:
            List[Dict[str, Any]]: 피드백 목록
        """
        feedback = []

        # 음소 피드백
        if phoneme_analysis:
            # 음소 길이 피드백
            phoneme_features = phoneme_analysis["phoneme_features"]
            if phoneme_features["mean_duration"] < 0.05:
                feedback.append(
                    {
                        "type": "phoneme_duration",
                        "level": "warning",
                        "message": "음소 발음이 너무 짧습니다. 각 음소를 더 길게 발음해보세요.",
                    }
                )

            # 음소 간격 피드백
            if phoneme_features["mean_gap"] > 0.1:
                feedback.append(
                    {
                        "type": "phoneme_gap",
                        "level": "warning",
                        "message": "음소 사이의 간격이 너무 깁니다. 음소를 더 연속적으로 발음해보세요.",
                    }
                )

        # 운율 피드백
        if prosody_analysis:
            # 피치 피드백
            pitch_stats = prosody_analysis["pitch"]["statistics"]
            if pitch_stats["std"] < 10:
                feedback.append(
                    {
                        "type": "pitch_variation",
                        "level": "info",
                        "message": "음높이 변화가 적습니다. 더 다양한 음높이로 발음해보세요.",
                    }
                )

            # 강세 피드백
            if prosody_analysis["energy"]["stress_count"] < 2:
                feedback.append(
                    {
                        "type": "stress",
                        "level": "info",
                        "message": "단어 강세가 부족합니다. 중요한 단어에 더 강세를 주어 발음해보세요.",
                    }
                )

        # 참조 오디오 비교 피드백
        if comparison:
            # 음소 일치도 피드백
            phoneme_comparison = comparison["phoneme_comparison"]
            if phoneme_comparison["match_rate"] < 0.8:
                feedback.append(
                    {
                        "type": "phoneme_match",
                        "level": "warning",
                        "message": "음소 발음이 참조 발음과 많이 다릅니다. 각 음소의 발음을 더 정확하게 해보세요.",
                    }
                )

            # 운율 차이 피드백
            prosody_comparison = comparison["prosody_comparison"]
            if abs(prosody_comparison["pitch"]["mean_diff"]) > 20:
                feedback.append(
                    {
                        "type": "pitch_difference",
                        "level": "info",
                        "message": "전체적인 음높이가 참조 발음과 다릅니다. 음높이를 조절해보세요.",
                    }
                )

        return feedback

    def _visualize_results(
        self,
        learner_audio: str,
        reference_audio: Optional[str],
        learner_textgrid: str,
        phoneme_analysis: Dict[str, Any],
        prosody_analysis: Dict[str, Any],
        comparison: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """결과 시각화

        Args:
            learner_audio: 학습자 오디오 파일 경로
            reference_audio: 참조 오디오 파일 경로 (선택사항)
            learner_textgrid: 학습자 TextGrid 파일 경로
            phoneme_analysis: 음소 분석 결과
            prosody_analysis: 운율 분석 결과
            comparison: 참조 오디오 비교 결과 (선택사항)

        Returns:
            List[str]: 시각화 결과 파일 경로 목록
        """
        visualization_paths = []

        try:
            # 1. 음소 시각화
            phoneme_plot_path = self.output_dir / "phoneme_analysis.png"
            self._plot_phoneme_analysis(
                phoneme_analysis,
                str(phoneme_plot_path),
            )
            visualization_paths.append(str(phoneme_plot_path))

            # 2. 운율 시각화
            prosody_plot_path = self.output_dir / "prosody_analysis.png"
            self.prosody_analyzer.visualize_prosody(
                learner_audio,
                reference_audio,
                str(prosody_plot_path),
            )
            visualization_paths.append(str(prosody_plot_path))

            # 3. 비교 시각화 (참조 오디오가 있는 경우)
            if comparison:
                comparison_plot_path = self.output_dir / "comparison_analysis.png"
                self._plot_comparison_analysis(
                    comparison,
                    str(comparison_plot_path),
                )
                visualization_paths.append(str(comparison_plot_path))

            return visualization_paths

        except Exception as e:
            logger.error(f"시각화 중 오류가 발생했습니다: {str(e)}")
            return []

    def _plot_phoneme_analysis(
        self,
        phoneme_analysis: Dict[str, Any],
        output_path: str,
    ) -> None:
        """음소 분석 시각화

        Args:
            phoneme_analysis: 음소 분석 결과
            output_path: 출력 파일 경로
        """
        import matplotlib.pyplot as plt

        # 음소 길이 분포
        phoneme_features = phoneme_analysis["phoneme_features"]
        durations = [p["duration"] for p in phoneme_features["phonemes"]]

        plt.figure(figsize=CURRENT_CONFIG["visualization"]["figsize"])
        plt.hist(durations, bins=20)
        plt.title("음소 길이 분포")
        plt.xlabel("길이 (초)")
        plt.ylabel("빈도")
        plt.savefig(output_path, dpi=CURRENT_CONFIG["visualization"]["dpi"])
        plt.close()

    def _plot_comparison_analysis(
        self,
        comparison: Dict[str, Any],
        output_path: str,
    ) -> None:
        """비교 분석 시각화

        Args:
            comparison: 비교 분석 결과
            output_path: 출력 파일 경로
        """
        import matplotlib.pyplot as plt

        # 음소 일치도
        phoneme_comparison = comparison["phoneme_comparison"]
        match_rate = phoneme_comparison["match_rate"]

        # 운율 차이
        prosody_comparison = comparison["prosody_comparison"]
        pitch_diff = prosody_comparison["pitch"]["mean_diff"]
        energy_diff = prosody_comparison["energy"]["mean_diff"]

        # 시각화
        plt.figure(figsize=CURRENT_CONFIG["visualization"]["figsize"])

        # 음소 일치도
        plt.subplot(1, 3, 1)
        plt.bar(["일치도"], [match_rate * 100])
        plt.title("음소 일치도")
        plt.ylabel("일치도 (%)")
        plt.ylim(0, 100)

        # 피치 차이
        plt.subplot(1, 3, 2)
        plt.bar(["피치 차이"], [pitch_diff])
        plt.title("피치 차이")
        plt.ylabel("차이 (Hz)")

        # 에너지 차이
        plt.subplot(1, 3, 3)
        plt.bar(["에너지 차이"], [energy_diff])
        plt.title("에너지 차이")
        plt.ylabel("차이 (dB)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=CURRENT_CONFIG["visualization"]["dpi"])
        plt.close()

    def analyze_prosody(
        self,
        learner_audio: str,
        native_audio: str,
        learner_text: str = None,
        learner_timing: str = None,
        visualize: bool = True,
    ) -> Dict:
        """억양/강세 분석

        Args:
            learner_audio: 학습자 오디오 파일 경로
            native_audio: 원어민 오디오 파일 경로
            learner_text: 학습자 전사 텍스트
            learner_timing: 학습자 타이밍 정보
            visualize: 시각화 여부

        Returns:
            Dict: 분석 결과
        """
        try:
            # ProsodyAnalyzer 인스턴스 생성
            analyzer = ProsodyAnalyzer()

            # 원어민 오디오로 임계값 조정
            analyzer.adjust_thresholds(native_audio)

            # 억양/강세 비교 (learner_text와 learner_timing 제외)
            result = analyzer.compare_prosody(
                learner_audio=learner_audio, reference_audio=native_audio
            )

            # 시각화
            if visualize:
                output_path = self.output_dir / "prosody_comparison.png"
                analyzer.visualize_prosody(
                    learner_audio=learner_audio,
                    reference_audio=native_audio,
                    output_path=str(output_path),
                )
                result["visualization"] = str(output_path)

            return result

        except Exception as e:
            logger.error(f"억양/강세 분석 실패: {e}")
            return None

    def _extract_pitch(self, segment: np.ndarray, sr: int) -> float:
        """음소 구간의 피치 추출"""
        try:
            if len(segment) < 2:
                return 0.0

            # librosa의 피치 추출 함수 사용
            pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
            if len(pitches) > 0:
                return np.mean(pitches[magnitudes > np.median(magnitudes)])
            return 0.0
        except Exception as e:
            logger.error(f"피치 추출 중 오류 발생: {e}")
            return 0.0

    def _extract_pitch_contour(self, y: np.ndarray, sr: int) -> np.ndarray:
        """전체 오디오의 피치 윤곽선 추출"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            if len(pitches) > 0:
                return np.mean(pitches, axis=1)
            return np.array([])
        except Exception as e:
            logger.error(f"피치 윤곽선 추출 중 오류 발생: {e}")
            return np.array([])

    def visualize_prosody(
        self, prosody_result: Dict[str, Any], output_path: str
    ) -> None:
        """운율 분석 결과 시각화"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # 피치 윤곽선 데이터 추출
            pitch_contour = prosody_result.get("pitch_contour", [])
            if len(pitch_contour) == 0:  # 빈 리스트 체크
                logger.warning("피치 윤곽선 데이터가 없습니다.")
                return

            # 시각화
            plt.figure(figsize=(12, 6))

            # 피치 윤곽선 플롯
            plt.subplot(2, 1, 1)
            plt.plot(pitch_contour)
            plt.title("Pitch Contour")
            plt.xlabel("Time")
            plt.ylabel("Pitch (Hz)")

            # 음소별 특성 플롯
            plt.subplot(2, 1, 2)
            phoneme_features = prosody_result.get("phoneme_features", {})
            if phoneme_features:
                durations = [f["duration"] for f in phoneme_features.values()]
                energies = [f["energy"] for f in phoneme_features.values()]
                pitches = [f["pitch"] for f in phoneme_features.values()]

                x = np.arange(len(durations))
                width = 0.25

                plt.bar(x - width, durations, width, label="Duration")
                plt.bar(x, energies, width, label="Energy")
                plt.bar(x + width, pitches, width, label="Pitch")

                plt.title("Phoneme Features")
                plt.xlabel("Phoneme")
                plt.ylabel("Value")
                plt.legend()

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            logger.info(f"운율 분석 시각화 결과가 저장되었습니다: {output_path}")

        except Exception as e:
            logger.error(f"운율 분석 시각화 중 오류 발생: {e}")
            raise

    def get_feedback(self, prompt: str) -> Optional[str]:
        """OpenAI API를 사용하여 피드백 생성

        Args:
            prompt: GPT 프롬프트

        Returns:
            Optional[str]: 생성된 피드백 (실패 시 None)
        """
        try:
            logger.info("🤖 GPT 피드백 생성 중...")

            if not self.openai_api_key:
                logger.error("OpenAI API 키가 설정되지 않았습니다.")
                return None

            client = OpenAI(api_key=self.openai_api_key)
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

    def generate_prompt(
        self,
        learner_text: str,
        native_text: str,
        script_text: str,
        learner_timing: str,
        native_timing: str,
        prosody_feedback: Optional[List[str]] = None,
    ) -> str:
        """GPT 프롬프트 생성"""
        # 기본 프롬프트
        prompt = f"""
다음은 한국어 학습자의 발화 정보와 원어민의 예시 발화 정보입니다.

# 학습자 발화 텍스트:
"{learner_text}"

# 원어민 발화 텍스트:
"{native_text}"

# 목표 스크립트:
"{script_text}"

# 학습자의 음소 정렬 정보 (MFA 분석 결과):
{learner_timing}

# 원어민의 음소 정렬 정보 (MFA 분석 결과):
{native_timing}

위 정보를 바탕으로 다음을 분석해줘:

1. 학습자와 원어민의 발음 차이점:
   - 어떤 단어나 음소에서 차이가 나는지 구체적으로 제시
   - 원어민은 어떻게 발음하는지 함께 설명
   - 예시: "학습자는 'ㅓ'를 'ㅗ'처럼 발음했는데, 원어민은 'ㅓ'를 더 넓게 발음했습니다."

2. 학습자와 원어민의 발화 속도 차이:
   - 어떤 구절에서 속도 차이가 나는지 구체적으로 제시
   - 원어민의 발화 속도를 참고하여 개선 방향 제시
   - 예시: "원어민은 '안녕하세요'를 0.8초에 발음했는데, 학습자는 1.2초가 걸렸습니다."

3. 학습자와 원어민의 억양 패턴 차이:
   - 어떤 부분에서 억양이 다른지 구체적으로 제시
   - 원어민의 억양 패턴을 참고하여 개선 방향 제시
   - 예시: "원어민은 문장 끝에서 음높이가 내려가는데, 학습자는 올라갔습니다."

4. 구체적인 개선 방안:
   - 원어민 발화를 참고하여 각 문제점별 개선 방법 제시
   - 실제 발음 연습 방법 구체적으로 설명
   - 예시: "원어민처럼 발음하려면 입을 더 크게 벌리고 'ㅓ'를 발음해보세요."

5. 연습 전략:
   - 원어민 발화를 따라하는 구체적인 방법 제시
   - 어떤 부분부터 연습하면 좋을지 순서대로 설명
   - 예시: "먼저 '안녕하세요'의 '녕' 부분을 천천히 연습한 후, 전체 문장을 연습하세요."
"""

        # RAG가 활성화된 경우, 관련 지식 검색 및 추가
        if self.config["use_rag"] and self.knowledge_base:
            # 발음 문제점 추출
            issues = self.extract_pronunciation_issues(
                learner_text, native_text, learner_timing, native_timing
            )

            # 쿼리 생성
            query = f"한국어 발음: {' '.join([issue['type'] for issue in issues])}"

            # 관련 지식 검색
            relevant_docs = self.knowledge_base.search(query, top_k=3)

            if relevant_docs:
                prompt += "\n\n# 참고할 발음 지식:\n"
                for doc in relevant_docs:
                    prompt += f"- {doc['content']}\n"

                prompt += "\n위 참고 지식을 활용하여 학습자에게 더 구체적이고 도움이 되는 피드백을 제공해주세요."

        # 억양/강세 피드백 추가
        if prosody_feedback:
            prompt += "\n\n# 억양과 강세 분석 결과:\n"
            for feedback in prosody_feedback:
                prompt += f"- {feedback}\n"

            prompt += "\n위 억양과 강세 분석 결과를 참고하여, 원어민 발화와 비교했을 때 어떤 부분을 개선해야 하는지 구체적으로 설명해주세요."

        return prompt

    def extract_pronunciation_issues(
        self,
        learner_result: Dict,
        reference_result: Dict,
        learner_timing: Dict,
        reference_timing: Dict
    ) -> List[Dict]:
        """발음 문제점 추출

        Args:
            learner_result: 학습자 인식 결과
            reference_result: 원어민 인식 결과
            learner_timing: 학습자 타이밍 정보
            reference_timing: 원어민 타이밍 정보

        Returns:
            List[Dict]: 발음 문제점 목록
        """
        issues = []
        
        # 단어 단위 비교 (안전한 접근)
        learner_words = learner_result.get("words", [])
        reference_words = reference_result.get("words", [])
        
        if learner_words and reference_words:
            # 최소 길이만큼 비교
            min_length = min(len(learner_words), len(reference_words))
            for i in range(min_length):
                learner_word = learner_words[i]
                ref_word = reference_words[i]
                
                # 단어 객체가 딕셔너리이고 'word' 키가 있는지 확인
                if (isinstance(learner_word, dict) and 'word' in learner_word and 
                    isinstance(ref_word, dict) and 'word' in ref_word):
                    
                    if learner_word["word"] != ref_word["word"]:
                        issues.append({
                            "type": "word_mismatch",
                            "position": i,
                            "learner": learner_word["word"],
                            "reference": ref_word["word"],
                            "start": learner_word.get("start", 0),
                            "end": learner_word.get("end", 0)
                        })
        
        # 텍스트 레벨에서의 기본 비교 (단어 정보가 없는 경우)
        if not learner_words or not reference_words:
            learner_text = learner_result.get("text", "").strip()
            reference_text = reference_result.get("text", "").strip()
            
            if learner_text != reference_text:
                issues.append({
                    "type": "text_mismatch", 
                    "learner": learner_text,
                    "reference": reference_text
                })
        
        # 운율 비교
        if "prosody" in learner_result and "prosody" in reference_result:
            learner_prosody = learner_result["prosody"]
            reference_prosody = reference_result["prosody"]
            
            # 피치 차이
            if ("pitch" in learner_prosody and "pitch" in reference_prosody and
                "mean" in learner_prosody["pitch"] and "mean" in reference_prosody["pitch"]):
                
                pitch_diff = abs(learner_prosody["pitch"]["mean"] - reference_prosody["pitch"]["mean"])
                if pitch_diff > 20:  # 20Hz 이상 차이나면 문제로 판단
                    issues.append({
                        "type": "pitch_difference",
                        "learner_mean": learner_prosody["pitch"]["mean"],
                        "reference_mean": reference_prosody["pitch"]["mean"],
                        "difference": pitch_diff
                    })
            
            # 에너지 차이
            if ("energy" in learner_prosody and "energy" in reference_prosody and
                "mean" in learner_prosody["energy"] and "mean" in reference_prosody["energy"]):
                
                energy_diff = abs(learner_prosody["energy"]["mean"] - reference_prosody["energy"]["mean"])
                if energy_diff > 0.1:  # 0.1 이상 차이나면 문제로 판단
                    issues.append({
                        "type": "energy_difference",
                        "learner_mean": learner_prosody["energy"]["mean"],
                        "reference_mean": reference_prosody["energy"]["mean"],
                        "difference": energy_diff
                    })
        
        return issues

    def align_audio(self, audio_path: str, text: str, speaker_type: str) -> Optional[Dict]:
        """MFA를 사용하여 오디오와 텍스트 정렬

        Args:
            audio_path: 오디오 파일 경로
            text: 정렬할 텍스트
            speaker_type: 화자 타입 ("learner" 또는 "native")

        Returns:
            Optional[Dict]: 정렬 결과
        """
        try:
            # MFA 작업 디렉토리 설정
            mfa_input_dir = self.temp_dir / "mfa_input" / speaker_type
            mfa_input_dir.mkdir(parents=True, exist_ok=True)
            
            # 오디오 파일을 mfa_input 디렉토리로 복사
            audio_filename = Path(audio_path).name
            shutil.copy2(audio_path, mfa_input_dir / audio_filename)
            
            # 텍스트 파일 생성
            text_filename = Path(audio_path).stem + ".txt"
            with open(mfa_input_dir / text_filename, "w", encoding="utf-8") as f:
                f.write(text)

            # MFA 정렬 실행
            logger.info(f"🔧 MFA 정렬 시작: {speaker_type}")
            command = [
                "mfa",
                "align",
                str(mfa_input_dir),
                str(self.lexicon_path),
                str(self.acoustic_model),
                str(self.mfa_output / speaker_type),
                "--clean",
            ]
            subprocess.run(command, check=True)
            logger.info("✅ MFA 정렬 완료")

            # TextGrid 파일 읽기
            textgrid_path = self.mfa_output / speaker_type / f"{Path(audio_path).stem}.TextGrid"
            if not textgrid_path.exists():
                logger.error(f"TextGrid 파일을 찾을 수 없습니다: {textgrid_path}")
                return None

            # TextGrid 파싱
            with open(textgrid_path, "r", encoding="utf-8") as f:
                textgrid_content = f.read()

            # 단어 단위 타이밍 정보 추출
            timing_info = {}
            current_word = None
            for line in textgrid_content.split("\n"):
                if line.strip().startswith('text = "'):
                    current_word = line.strip()[8:-1]  # "text = " 제거
                elif line.strip().startswith('xmin = '):
                    start_time = float(line.strip()[7:])
                elif line.strip().startswith('xmax = '):
                    end_time = float(line.strip()[7:])
                    if current_word:
                        timing_info[current_word] = {
                            "start": start_time,
                            "end": end_time
                        }

            return timing_info

        except Exception as e:
            logger.error(f"MFA 정렬 중 오류가 발생했습니다: {str(e)}")
            return None

    def convert_audio(self, audio_path: str) -> Optional[str]:
        """오디오 파일 변환 (.m4a -> .wav)

        Args:
            audio_path: 입력 오디오 파일 경로

        Returns:
            Optional[str]: 변환된 wav 파일 경로
        """
        try:
            # 입력 파일 경로 확인
            input_path = Path(audio_path)
            if not input_path.exists():
                logger.error(f"입력 파일을 찾을 수 없습니다: {audio_path}")
                return None

            # 출력 파일 경로 설정
            output_dir = self.temp_dir / "wav"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{input_path.stem}.wav"

            # ffmpeg로 변환
            subprocess.run([
                "ffmpeg",
                "-i", str(input_path),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",  # 기존 파일 덮어쓰기
                str(output_path)
            ], check=True)

            logger.info(f"오디오 파일이 변환되었습니다: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"오디오 변환 중 오류가 발생했습니다: {str(e)}")
            return None

    def _generate_simple_feedback(self, result: Dict) -> Optional[Dict]:
        """간단한 피드백 생성
        
        Args:
            result: 분석 결과
            
        Returns:
            Optional[Dict]: 피드백 결과
        """
        try:
            feedback = {
                "summary": "발음 분석이 완료되었습니다.",
                "suggestions": []
            }
            
            # 발음 문제점이 있다면 피드백 추가
            if "issues" in result["comparison"]:
                issues = result["comparison"]["issues"]
                if issues:
                    feedback["suggestions"].append("발견된 발음 문제점들을 개선해보세요.")
                    for issue in issues[:3]:  # 최대 3개까지만
                        if issue["type"] == "word_mismatch":
                            feedback["suggestions"].append(
                                f"'{issue['learner']}'을(를) '{issue['reference']}'로 발음해보세요."
                            )
                        elif issue["type"] == "text_mismatch":
                            feedback["suggestions"].append(
                                "발음된 텍스트와 원본 텍스트가 다릅니다."
                            )
            
            # 운율 분석 결과가 있다면 피드백 추가
            if "prosody" in result["comparison"]:
                prosody = result["comparison"]["prosody"]
                if prosody:
                    feedback["suggestions"].append("억양과 강세를 원어민과 비교하여 개선해보세요.")
            
            return feedback
            
        except Exception as e:
            logger.error(f"피드백 생성 중 오류 발생: {e}")
            return None

    def summarize_textgrid(self, path: str) -> Optional[str]:
        """TextGrid 파일에서 음소 정보 추출

        Args:
            path: TextGrid 파일 경로

        Returns:
            Optional[str]: 음소 정보 요약 (실패 시 None)
        """
        try:
            logger.info(f"📊 TextGrid 요약 중: {path}")
            import textgrid
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

    def summarize_textgrid_compact(self, path: str) -> Optional[str]:
        """TextGrid 파일에서 핵심 음소 정보만 추출 (압축 버전)

        Args:
            path: TextGrid 파일 경로

        Returns:
            Optional[str]: 압축된 음소 정보 요약
        """
        try:
            logger.info(f"📊 TextGrid 압축 요약 중: {path}")
            import textgrid
            tg = textgrid.TextGrid.fromFile(path)
            
            # 중요한 음소만 필터링 (무음 구간 제외)
            important_phonemes = []
            
            for tier in tg.tiers:
                if tier.name.lower() in ["phones", "phoneme", "phone"]:
                    for interval in tier:
                        phoneme = interval.mark.strip()
                        # 무음 구간이나 침묵 구간 제외
                        if phoneme and phoneme not in ['', 'sil', 'sp', '<eps>']:
                            duration = round(interval.maxTime - interval.minTime, 2)
                            # 0.05초 이상인 음소만 포함 (너무 짧은 것들 제외)
                            if duration >= 0.05:
                                important_phonemes.append(f"{phoneme}({duration}s)")
            
            # 최대 20개의 핵심 음소만 반환
            if len(important_phonemes) > 20:
                # 앞쪽 10개, 뒤쪽 10개만 선택
                important_phonemes = important_phonemes[:10] + ['...'] + important_phonemes[-10:]
            
            return ", ".join(important_phonemes)
            
        except Exception as e:
            logger.error(f"TextGrid 압축 요약 실패: {e}")
            return None

    def extract_pronunciation_issues_detailed(
        self, learner_text: str, native_text: str, learner_timing: str
    ) -> List[str]:
        """발음 문제점 추출 (상세 분석)

        Args:
            learner_text: 학습자 텍스트
            native_text: 원어민 텍스트
            learner_timing: 학습자 타이밍 정보

        Returns:
            List[str]: 발견된 문제점 목록
        """
        issues = []

        # 1. 텍스트 길이 차이 (너무 크면 발음 누락 가능성)
        if len(native_text) > len(learner_text) * 1.2:
            issues.append("발음 누락 가능성 있음")

        # 2. 받침 관련 문제 검출
        learner_words = learner_text.split()
        native_words = native_text.split()

        for n_word in native_words:
            if len(n_word) >= 2 and n_word not in learner_text:
                # 받침이 있는 단어면 받침 관련 문제 가능성 추가
                if any(c in "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ" for c in n_word):
                    issues.append(f"'{n_word}' 단어 발음 문제 (받침 관련)")

        # 3. 공통 음소 문제 (음소 길이, 음색 등)
        phoneme_issues = set()
        for line in learner_timing.split(","):
            parts = line.split("(")
            if len(parts) >= 2:
                phoneme = parts[0].strip()

                # 모음 관련 문제 검출
                if phoneme in ["ㅓ", "ㅗ", "ㅜ", "ㅡ"]:
                    phoneme_issues.add(f"{phoneme} 모음 발음")

                # 경음/격음 관련 문제 검출
                if phoneme in ["ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ", "ㅋ", "ㅌ", "ㅍ", "ㅊ"]:
                    phoneme_issues.add(f"{phoneme} 자음 발음")

        issues.extend(list(phoneme_issues))
        return issues

    def generate_detailed_prompt(
        self,
        learner_text: str,
        native_text: str,
        script_text: str,
        learner_timing: str,
        native_timing: str,
        prosody_feedback: Optional[Dict] = None,
    ) -> str:
        """상세한 GPT 프롬프트 생성

        Args:
            learner_text: 학습자 발화 텍스트
            native_text: 원어민 발화 텍스트
            script_text: 목표 스크립트
            learner_timing: 학습자 음소 정렬 정보
            native_timing: 원어민 음소 정렬 정보
            prosody_feedback: 운율 분석 결과

        Returns:
            str: GPT 프롬프트
        """
        # 기본 프롬프트
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

        # 운율 분석 결과 추가
        if prosody_feedback:
            prompt += f"\n\n# 운율 및 억양 분석 결과:\n"
            if 'differences' in prosody_feedback:
                diff = prosody_feedback['differences']
                prompt += f"- 피치 평균 차이: {diff.get('pitch', {}).get('mean', 0):.2f}Hz\n"
                prompt += f"- 에너지 평균 차이: {diff.get('energy', {}).get('mean', 0):.3f}\n"
                prompt += f"- 말하기 속도 차이: {diff.get('time', {}).get('total_duration', 0):.2f}초\n"
            
            prompt += "\n위 운율 분석 결과를 참고하여 학습자의 억양과 강세에 대한 구체적인 피드백도 함께 제공해주세요."

        # RAG가 활성화된 경우, 관련 지식 검색 및 추가
        if self.config["use_rag"] and self.knowledge_base:
            # 발음 문제점 추출
            issues = self.extract_pronunciation_issues_detailed(
                learner_text, native_text, learner_timing
            )

            # 쿼리 생성
            query = f"한국어 발음: {' '.join(issues)}"

            # 관련 지식 검색
            relevant_docs = self.knowledge_base.search(query, top_k=3)

            if relevant_docs:
                prompt += "\n\n# 참고할 발음 지식:\n"
                for doc in relevant_docs:
                    prompt += f"- {doc['content']}\n"

                prompt += "\n위 참고 지식을 활용하여 학습자에게 더 구체적이고 도움이 되는 피드백을 제공해주세요."

        return prompt

    def generate_compact_prompt(
        self,
        learner_text: str,
        native_text: str,
        script_text: str,
        learner_timing: str,
        native_timing: str,
        prosody_feedback: Optional[Dict] = None,
    ) -> str:
        """압축된 GPT 프롬프트 생성 (토큰 절약형)

        Args:
            learner_text: 학습자 발화 텍스트
            native_text: 원어민 발화 텍스트
            script_text: 목표 스크립트
            learner_timing: 학습자 음소 정렬 정보 (압축형)
            native_timing: 원어민 음소 정렬 정보 (압축형)
            prosody_feedback: 운율 분석 결과

        Returns:
            str: 압축된 GPT 프롬프트
        """
        
        # 기본 정보만 포함한 간단한 프롬프트
        prompt = f"""한국어 발음 교정 요청:

학습자: "{learner_text}"
원어민: "{native_text}"
목표: "{script_text}"

핵심 음소 정보:
- 학습자: {learner_timing}
- 원어민: {native_timing}

분석 요청:
1. 잘못 발음된 단어/음소
2. 누락된 부분  
3. 속도 문제
4. 개선 방법

간결하고 구체적으로 답변해주세요."""

        # 운율 정보 추가 (압축형)
        if prosody_feedback and 'differences' in prosody_feedback:
            diff = prosody_feedback['differences']
            pitch_diff = diff.get('pitch', {}).get('mean', 0)
            energy_diff = diff.get('energy', {}).get('mean', 0)
            if abs(pitch_diff) > 20 or abs(energy_diff) > 0.1:
                prompt += f"\n\n운율 차이: 피치{pitch_diff:+.0f}Hz, 에너지{energy_diff:+.2f}"

        # RAG 지식 (최대 1개만)
        if self.config["use_rag"] and self.knowledge_base:
            issues = self.extract_pronunciation_issues_detailed(
                learner_text, native_text, learner_timing
            )
            if issues:
                query = f"한국어 발음: {issues[0]}"  # 첫 번째 이슈만 사용
                relevant_docs = self.knowledge_base.search(query, top_k=1)
                if relevant_docs:
                    prompt += f"\n\n참고: {relevant_docs[0]['content'][:200]}..."  # 200자만 사용

        return prompt
