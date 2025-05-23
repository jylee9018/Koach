from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
from dataclasses import dataclass
from .force_aligner import ForceAligner
from .audio_embedder import AudioEmbedder
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhonemeError:
    phoneme: str
    error_type: str
    start_time: float
    end_time: float


class PronunciationAnalyzer:
    def __init__(self, threshold_ms: int = 50):
        self.threshold_ms = threshold_ms

        # Forced Aligner 초기화
        self.aligner = ForceAligner()

        # 임베딩 추출기 초기화
        self.embedder = AudioEmbedder(model_type="wav2vec2")

    def detect_errors(self, alignment_result: Dict) -> List[PhonemeError]:
        """음소 단위 오류 탐지"""
        errors = []

        for phoneme in alignment_result.get("phonemes", []):
            # 정렬 실패한 경우 누락으로 판단
            if not phoneme.get("aligned", True):
                errors.append(
                    PhonemeError(
                        phoneme=phoneme["label"],
                        error_type="누락",
                        start_time=phoneme["start"],
                        end_time=phoneme["end"],
                    )
                )
                continue

            # 음소 길이가 임계값보다 짧으면 왜곡으로 판단
            duration = (phoneme["end"] - phoneme["start"]) * 1000  # ms로 변환
            if duration <= self.threshold_ms:
                errors.append(
                    PhonemeError(
                        phoneme=phoneme["label"],
                        error_type="왜곡",
                        start_time=phoneme["start"],
                        end_time=phoneme["end"],
                    )
                )

        return errors

    async def analyze(self, features: Dict) -> Dict:
        """음성 특징을 분석하여 발음 평가"""
        try:
            # features가 None인 경우 처리
            if features is None:
                return {
                    "errors": ["특징 추출에 실패했습니다."],
                    "similarity": 0.0,
                    "pitch": {"mean": 0.0, "variation": 0.0},
                }

            # 피치 데이터 안전하게 가져오기
            pitch = features.get("pitch", {"mean": 0.0, "std": 0.0})
            duration = features.get("duration", {"total": 0.0})
            energy = features.get("energy", {"rms": 0.0})

            # 유사도 점수 계산 (예시)
            similarity = min(
                1.0,
                max(
                    0.0,
                    0.5
                    + 0.2 * (pitch.get("mean", 0.0) / 200)
                    + 0.2 * (duration.get("total", 0.0) / 2)
                    + 0.1 * (energy.get("rms", 0.0) * 10),
                ),
            )

            # 발음 오류 분석 (예시)
            errors = []
            if pitch.get("std", 0.0) > 50:
                errors.append("억양 변화가 큼")
            if duration.get("total", 0.0) > 3:
                errors.append("발화 속도가 느림")
            if energy.get("rms", 0.0) < 0.1:
                errors.append("발화 강도가 약함")

            return {
                "errors": errors,
                "similarity": similarity,
                "pitch": {
                    "mean": pitch.get("mean", 0.0),
                    "variation": pitch.get("std", 0.0),
                },
            }
        except Exception as e:
            raise Exception(f"Pronunciation analysis failed: {str(e)}")

    async def analyze_with_alignment(self, audio_path: str, script: str) -> Dict:
        """Forced Alignment를 사용한 상세 발음 분석"""
        try:
            # 1. Forced Alignment 수행
            alignment_result = await self.aligner.align(audio_path, script)

            # 2. 음소 오류 탐지
            phoneme_errors = self.detect_errors(alignment_result)

            # 3. 임베딩 추출
            # 단어 임베딩 추출
            word_embeddings = await self.embedder.extract_word_embeddings(
                audio_path, alignment_result
            )

            # 음소 임베딩 추출
            phoneme_embeddings = await self.embedder.extract_phoneme_embeddings(
                audio_path, alignment_result
            )

            embeddings = {
                "word_embeddings": word_embeddings,
                "phoneme_embeddings": phoneme_embeddings,
            }

            # 4. 결과 구성
            result = {
                "alignment": alignment_result,
                "phoneme_errors": phoneme_errors,
                "word_count": len(alignment_result.get("words", [])),
                "error_count": len(phoneme_errors),
                "similarity": self._calculate_alignment_similarity(
                    alignment_result, phoneme_errors
                ),
                "embeddings": embeddings,
            }

            return result
        except Exception as e:
            logger.error(f"Alignment analysis failed: {e}")
            raise Exception(f"Alignment analysis failed: {str(e)}")

    async def analyze_comparison(self, comparison_result: Dict, script: str) -> Dict:
        """비교 결과를 분석하여 더 상세한 피드백 제공"""
        try:
            # 기존 비교 결과에서 필요한 정보 추출
            pitch_comparison = comparison_result.get("pitch_comparison", {})
            duration_comparison = comparison_result.get("duration_comparison", {})
            energy_comparison = comparison_result.get("energy_comparison", {})
            overall_similarity = comparison_result.get("overall_similarity", 0.0)

            # 분석 결과 구성
            analysis_details = {
                "pitch": {
                    "difference": pitch_comparison.get("mean_diff", 0.0),
                    "contour_similarity": pitch_comparison.get(
                        "contour_similarity", 0.0
                    ),
                },
                "duration": {"pace_ratio": duration_comparison.get("pace_ratio", 1.0)},
                "energy": {"difference": energy_comparison.get("rms_diff", 0.0)},
            }

            # 발음 오류 목록 (예시)
            pronunciation_errors = []

            # 억양 문제 확인
            if pitch_comparison.get("contour_similarity", 1.0) < 0.7:
                pronunciation_errors.append("억양 패턴이 원어민과 다름")

            # 속도 문제 확인
            pace_ratio = duration_comparison.get("pace_ratio", 1.0)
            if pace_ratio > 1.3:
                pronunciation_errors.append("발화 속도가 원어민보다 느림")
            elif pace_ratio < 0.7:
                pronunciation_errors.append("발화 속도가 원어민보다 빠름")

            # 강세 문제 확인
            if energy_comparison.get("energy_ratio", 1.0) < 0.7:
                pronunciation_errors.append("발화 강도가 원어민보다 약함")

            return {
                "similarity": overall_similarity,
                "pronunciation_errors": pronunciation_errors,
                "analysis_details": analysis_details,
                "script": script,
            }
        except Exception as e:
            logger.error(f"Comparison analysis failed: {e}")
            raise Exception(f"Comparison analysis failed: {str(e)}")

    def _calculate_alignment_similarity(
        self, alignment_result: Dict, phoneme_errors: List[PhonemeError]
    ) -> float:
        """정렬 결과와 오류를 기반으로 유사도 점수 계산"""
        # 전체 음소 수
        total_phonemes = len(alignment_result.get("phonemes", []))
        if total_phonemes == 0:
            return 0.0

        # 오류 음소 수
        error_phonemes = len(phoneme_errors)

        # 정렬 성공률 계산
        alignment_success_rate = sum(
            1 for p in alignment_result.get("phonemes", []) if p.get("aligned", True)
        ) / max(total_phonemes, 1)

        # 오류 비율 계산
        error_rate = error_phonemes / max(total_phonemes, 1)

        # 유사도 점수 계산 (오류가 적을수록, 정렬 성공률이 높을수록 높은 점수)
        similarity = (1.0 - error_rate) * 0.7 + alignment_success_rate * 0.3

        return max(0.0, min(1.0, similarity))
