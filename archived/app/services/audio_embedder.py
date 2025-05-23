import torch
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple, Union
import logging
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    WhisperProcessor,
    WhisperModel,
    Wav2Vec2FeatureExtractor,
)

logger = logging.getLogger(__name__)


class AudioEmbedder:
    """
    음성 임베딩을 추출하는 클래스
    Wav2Vec2, Whisper 등의 모델을 지원합니다.
    """

    def __init__(self, model_type="wav2vec2"):
        """오디오 임베딩 추출기 초기화"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        if model_type == "wav2vec2":
            model_name = "facebook/wav2vec2-large-xlsr-53"
            # 프로세서 대신 특성 추출기만 로드
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_name
            )
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    async def extract_embeddings(self, audio_path: str) -> np.ndarray:
        """오디오 파일에서 임베딩 추출"""
        try:
            # 오디오 로드
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

            # 특성 추출
            inputs = self.feature_extractor(
                speech_array, sampling_rate=sampling_rate, return_tensors="pt"
            ).to(self.device)

            # 임베딩 추출
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu().numpy().squeeze()
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}")
            raise

    async def extract_word_embeddings(
        self, audio_path: str, alignment_result: Dict, sampling_rate: int = 16000
    ) -> Dict[str, np.ndarray]:
        """
        정렬 결과를 사용하여 단어별 임베딩 추출

        Args:
            audio_path: 오디오 파일 경로
            alignment_result: 정렬 결과 (ForceAligner에서 반환된 형식)
            sampling_rate: 샘플링 레이트

        Returns:
            Dict[str, np.ndarray]: 단어별 임베딩 벡터
        """
        # 오디오 로드
        audio_array, sr = librosa.load(audio_path, sr=sampling_rate)

        word_embeddings = {}

        for word_info in alignment_result.get("words", []):
            word = word_info.get("word")
            start = word_info.get("start")
            end = word_info.get("end")

            if word and start is not None and end is not None:
                # 시간을 샘플 인덱스로 변환
                start_idx = int(start * sr)
                end_idx = int(end * sr)

                # 단어 구간 오디오 추출
                if start_idx < end_idx and end_idx <= len(audio_array):
                    word_audio = audio_array[start_idx:end_idx]

                    # 임베딩 추출
                    word_embedding = await self._extract_from_array(word_audio, sr)
                    word_embeddings[word] = word_embedding

        return word_embeddings

    async def extract_phoneme_embeddings(
        self, audio_path: str, alignment_result: Dict, sampling_rate: int = 16000
    ) -> Dict[str, np.ndarray]:
        """
        정렬 결과를 사용하여 음소별 임베딩 추출

        Args:
            audio_path: 오디오 파일 경로
            alignment_result: 정렬 결과 (ForceAligner에서 반환된 형식)
            sampling_rate: 샘플링 레이트

        Returns:
            Dict[str, np.ndarray]: 음소별 임베딩 벡터
        """
        # 오디오 로드
        audio_array, sr = librosa.load(audio_path, sr=sampling_rate)

        phoneme_embeddings = {}

        for phoneme_info in alignment_result.get("phonemes", []):
            phoneme = phoneme_info.get("label")
            start = phoneme_info.get("start")
            end = phoneme_info.get("end")

            if phoneme and start is not None and end is not None:
                # 시간을 샘플 인덱스로 변환
                start_idx = int(start * sr)
                end_idx = int(end * sr)

                # 음소 구간 오디오 추출
                if start_idx < end_idx and end_idx <= len(audio_array):
                    phoneme_audio = audio_array[start_idx:end_idx]

                    # 임베딩 추출
                    phoneme_embedding = await self._extract_from_array(
                        phoneme_audio, sr
                    )

                    # 같은 음소가 여러 번 나올 수 있으므로 키를 고유하게 만듦
                    key = f"{phoneme}_{start:.3f}_{end:.3f}"
                    phoneme_embeddings[key] = phoneme_embedding

        return phoneme_embeddings

    async def _extract_from_array(
        self, audio_array: np.ndarray, sampling_rate: int = 16000
    ) -> np.ndarray:
        """
        오디오 배열에서 임베딩 추출
        """
        # 오디오 길이가 너무 짧은 경우 처리
        min_samples = int(0.1 * sampling_rate)  # 최소 100ms
        if len(audio_array) < min_samples:
            audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))

        # 특성 추출
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=16000,  # 최대 1초로 제한
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 마지막 은닉 상태 사용
            embeddings = outputs.last_hidden_state

            # 평균 임베딩 계산
            mean_embedding = torch.mean(embeddings, dim=1).squeeze().cpu().numpy()

            return mean_embedding
