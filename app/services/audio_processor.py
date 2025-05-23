import librosa
import parselmouth
import numpy as np
from typing import Dict, Tuple, Optional
import os
import logging
from .force_aligner import ForceAligner
from .audio_embedder import AudioEmbedder

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        # Forced Aligner 초기화
        self.aligner = ForceAligner()

        # 임베딩 추출기 초기화
        self.embedder = AudioEmbedder(model_type="wav2vec2")

    async def process_audio(self, audio_path: str) -> Dict:
        """오디오 파일 처리 및 특징 추출"""
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # 피치 추출
            pitch_values = self._extract_pitch(y, sr)
            pitch_mean = (
                np.mean(pitch_values[pitch_values > 0])
                if np.any(pitch_values > 0)
                else 0
            )
            pitch_std = (
                np.std(pitch_values[pitch_values > 0])
                if np.any(pitch_values > 0)
                else 0
            )

            # 지속시간 계산
            duration = len(y) / sr

            # 에너지/강도 계산
            rms = np.sqrt(np.mean(y**2))

            # 전체 임베딩 추출
            full_embedding = await self.embedder.extract_embeddings(audio_path)

            return {
                "pitch": {
                    "mean": float(pitch_mean),
                    "std": float(pitch_std),
                    "contour": pitch_values.tolist(),
                },
                "duration": {
                    "total": float(duration),
                },
                "energy": {
                    "rms": float(rms),
                },
                "embedding": {
                    "full": full_embedding.tolist(),
                },
            }
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise Exception(f"Audio processing failed: {str(e)}")

    async def process_audio_with_alignment(self, audio_path: str, script: str) -> Dict:
        """오디오 파일 처리 및 정렬 기반 특징 추출"""
        try:
            # 기본 오디오 처리
            features = await self.process_audio(audio_path)

            # Forced Alignment 수행
            alignment_result = await self.aligner.align(audio_path, script)

            # 단어 및 음소 임베딩 추출
            word_embeddings = await self.embedder.extract_word_embeddings(
                audio_path, alignment_result
            )

            phoneme_embeddings = await self.embedder.extract_phoneme_embeddings(
                audio_path, alignment_result
            )

            # 결과 통합
            features["alignment"] = alignment_result
            features["embedding"]["words"] = {
                k: v.tolist() for k, v in word_embeddings.items()
            }
            features["embedding"]["phonemes"] = {
                k: v.tolist() for k, v in phoneme_embeddings.items()
            }

            return features
        except Exception as e:
            logger.error(f"Audio processing with alignment failed: {e}")
            raise Exception(f"Audio processing with alignment failed: {str(e)}")

    async def compare_audios(
        self, learner_features: Dict, native_features: Dict
    ) -> Dict:
        """학습자와 원어민 발음 비교 분석"""
        try:
            # 피치 비교
            pitch_diff = {
                "mean_diff": abs(
                    learner_features["pitch"]["mean"] - native_features["pitch"]["mean"]
                ),
                "std_diff": abs(
                    learner_features["pitch"]["std"] - native_features["pitch"]["std"]
                ),
                "contour_similarity": self._calculate_contour_similarity(
                    learner_features["pitch"]["contour"],
                    native_features["pitch"]["contour"],
                ),
            }

            # 지속시간 비교
            duration_diff = {
                "total_diff": abs(
                    learner_features["duration"]["total"]
                    - native_features["duration"]["total"]
                ),
                "pace_ratio": learner_features["duration"]["total"]
                / max(native_features["duration"]["total"], 0.001),
            }

            # 에너지/강도 비교
            energy_diff = {
                "rms_diff": abs(
                    learner_features["energy"]["rms"] - native_features["energy"]["rms"]
                ),
                "energy_ratio": learner_features["energy"]["rms"]
                / max(native_features["energy"]["rms"], 0.001),
            }

            # 유사도 점수 계산
            similarity = self._calculate_overall_similarity(
                pitch_diff, duration_diff, energy_diff
            )

            # 임베딩 유사도 계산 (있는 경우)
            embedding_similarity = 0.0
            if (
                "embedding" in learner_features
                and "embedding" in native_features
                and "full" in learner_features["embedding"]
                and "full" in native_features["embedding"]
            ):
                embedding_similarity = self._calculate_embedding_similarity(
                    np.array(learner_features["embedding"]["full"]),
                    np.array(native_features["embedding"]["full"]),
                )

            return {
                "pitch_comparison": pitch_diff,
                "duration_comparison": duration_diff,
                "energy_comparison": energy_diff,
                "embedding_similarity": float(embedding_similarity),
                "overall_similarity": similarity,
            }
        except Exception as e:
            raise Exception(f"Audio comparison failed: {str(e)}")

    def _extract_pitch(self, y: np.ndarray, sr: int) -> np.ndarray:
        """피치(F0) 추출"""
        # librosa의 피치 추출 함수 사용
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # 각 프레임에서 가장 강한 피치 선택
        pitch_values = []
        for t in range(magnitudes.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_values.append(pitch)

        return np.array(pitch_values)

    def _calculate_contour_similarity(self, contour1, contour2) -> float:
        """피치 윤곽선 유사도 계산"""
        # 길이 정규화 (DTW 등 더 정교한 방법으로 대체 가능)
        # 간단한 MVP를 위해 길이가 다른 경우 더 짧은 쪽에 맞춤
        min_len = min(len(contour1), len(contour2))
        c1 = np.array(contour1[:min_len])
        c2 = np.array(contour2[:min_len])

        # 0이 아닌 값만 고려 (무성음 제외)
        mask = (c1 > 0) & (c2 > 0)
        if not np.any(mask):
            return 0.0

        c1_valid = c1[mask]
        c2_valid = c2[mask]

        # 정규화된 오차 계산
        if len(c1_valid) == 0:
            return 0.0

        error = np.mean(np.abs(c1_valid - c2_valid) / c2_valid)
        similarity = max(0.0, 1.0 - error)

        return float(similarity)

    def _calculate_overall_similarity(
        self, pitch_diff, duration_diff, energy_diff
    ) -> float:
        """전체 유사도 점수 계산"""
        # 각 특성별 가중치 설정
        pitch_weight = 0.5  # 억양이 가장 중요
        duration_weight = 0.3  # 발화 속도/리듬
        energy_weight = 0.2  # 강세

        # 각 특성별 유사도 계산 (0~1, 높을수록 유사)
        pitch_sim = (
            max(0.0, 1.0 - (pitch_diff["mean_diff"] / 100)) * 0.3
            + pitch_diff["contour_similarity"] * 0.7
        )

        duration_sim = max(0.0, 1.0 - abs(duration_diff["pace_ratio"] - 1.0))

        energy_sim = max(0.0, 1.0 - abs(energy_diff["energy_ratio"] - 1.0))

        # 가중 평균
        overall_sim = (
            pitch_sim * pitch_weight
            + duration_sim * duration_weight
            + energy_sim * energy_weight
        )

        return float(overall_sim)

    def _calculate_embedding_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """임베딩 벡터 간 코사인 유사도 계산"""
        # 벡터 정규화
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 코사인 유사도 계산
        cos_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # 유사도 범위를 0~1로 조정
        similarity = (cos_sim + 1) / 2

        return float(similarity)

    async def extract_features(self, audio_path: str) -> Dict:
        """오디오 파일에서 특징 추출"""
        try:
            if not audio_path or not os.path.exists(audio_path):
                raise ValueError(f"오디오 파일이 존재하지 않습니다: {audio_path}")

            # 오디오 파일 로딩
            y, sr = librosa.load(audio_path, sr=None)

            # 피치 추출 (Fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitches = pitches[magnitudes > 0.1]
            pitches = pitches[pitches > 0]  # 0보다 큰 값만 유지

            pitch_mean = float(np.mean(pitches)) if len(pitches) > 0 else 0.0
            pitch_std = float(np.std(pitches)) if len(pitches) > 0 else 0.0
            pitch_contour = pitches.tolist() if len(pitches) > 0 else []

            # 발화 길이
            duration = float(len(y) / sr)

            # 에너지/강도
            rms = float(np.sqrt(np.mean(y**2)))

            return {
                "pitch": {
                    "mean": pitch_mean,
                    "std": pitch_std,
                    "contour": (
                        pitch_contour[:100]
                        if len(pitch_contour) > 100
                        else pitch_contour
                    ),  # 크기 제한
                },
                "duration": {
                    "total": duration,
                    "speech_ratio": 0.8,  # 간단한 예시 값
                },
                "energy": {
                    "rms": rms,
                    "max": float(np.max(np.abs(y))),
                },
            }
        except Exception as e:
            raise Exception(f"특징 추출 실패: {str(e)}")

    def _analyze_duration(
        self, y: np.ndarray, sr: int, alignment_result: Dict = None
    ) -> Dict:
        """지속시간 분석"""
        # 전체 오디오 길이
        total_duration = len(y) / sr

        # 발화 속도 계산 (음절 수 / 초)
        syllable_rate = 0

        if alignment_result and "words" in alignment_result:
            # 단어 수
            word_count = len(alignment_result["words"])

            # 대략적인 음절 수 계산 (한국어 특성 고려)
            syllable_count = 0
            for word in alignment_result["words"]:
                # 한국어 단어의 음절 수 추정 (초성+중성+종성 조합)
                word_text = word.get("word", "")
                syllable_count += len(word_text)

            if total_duration > 0:
                syllable_rate = syllable_count / total_duration

        return {
            "total": float(total_duration),
            "syllable_rate": float(syllable_rate),
            "speech_ratio": 0.8,  # 실제로는 VAD(Voice Activity Detection)로 계산해야 함
        }
