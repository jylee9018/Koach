import librosa
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import textgrid
from typing import Dict, Optional, List, Any
import logging
from datetime import datetime
import os
from pathlib import Path

from config.settings import CURRENT_CONFIG

logger = logging.getLogger("Koach")


class ProsodyAnalyzer:
    def __init__(self):
        """운율 분석기 초기화"""
        self.sample_rate = CURRENT_CONFIG["audio"]["sample_rate"]
        self.hop_length = CURRENT_CONFIG["audio"]["hop_length"]
        self.frame_length = CURRENT_CONFIG["audio"]["frame_length"]
        self.pitch_threshold = 0.5
        self.energy_threshold = 0.5
        self.silence_threshold = 0.1

    def analyze_audio(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """오디오 파일 분석

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            Optional[Dict[str, Any]]: 분석 결과
        """
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=None)

            # 피치 분석
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > self.pitch_threshold]
            
            # 에너지 분석
            energy = librosa.feature.rms(y=y)[0]
            
            # 무음 구간 분석
            is_silence = energy < self.silence_threshold
            silence_duration = np.sum(is_silence) * (len(y) / sr / len(energy))

            return {
                "pitch": {
                    "values": pitch_values.tolist(),
                    "statistics": {
                        "mean": float(np.mean(pitch_values)),
                        "std": float(np.std(pitch_values)),
                        "range": float(np.ptp(pitch_values))
                    }
                },
                "energy": {
                    "values": energy.tolist(),
                    "statistics": {
                        "mean": float(np.mean(energy)),
                        "std": float(np.std(energy)),
                        "range": float(np.ptp(energy)),
                        "min": float(np.min(energy))
                    }
                },
                "time": {
                    "total_duration": float(len(y) / sr),
                    "speech_duration": float(len(y) / sr - silence_duration),
                    "silence_duration": float(silence_duration)
                }
            }

        except Exception as e:
            logger.error(f"오디오 분석 중 오류가 발생했습니다: {str(e)}")
            return None

    def analyze_pitch(
        self,
        y: np.ndarray,
        textgrid_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """피치 분석

        Args:
            y: 오디오 신호
            textgrid_path: TextGrid 파일 경로 (선택사항)

        Returns:
            Dict[str, Any]: 피치 분석 결과
        """
        # 피치 추출
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )

        # 피치 통계 계산
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_stats = {
            "mean": float(np.mean(pitch_values)),
            "std": float(np.std(pitch_values)),
            "min": float(np.min(pitch_values)),
            "max": float(np.max(pitch_values)),
            "range": float(np.max(pitch_values) - np.min(pitch_values)),
        }

        # 구간별 분석 (TextGrid가 있는 경우)
        segment_analysis = None
        if textgrid_path:
            segment_analysis = self._analyze_pitch_segments(
                pitches,
                magnitudes,
                textgrid_path,
            )

        return {
            "statistics": pitch_stats,
            "segments": segment_analysis,
        }

    def analyze_energy(
        self,
        y: np.ndarray,
        textgrid_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """에너지 분석

        Args:
            y: 오디오 신호
            textgrid_path: TextGrid 파일 경로 (선택사항)

        Returns:
            Dict[str, Any]: 에너지 분석 결과
        """
        # 에너지 추출
        energy = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        # 에너지 통계 계산
        energy_stats = {
            "mean": float(np.mean(energy)),
            "std": float(np.std(energy)),
            "min": float(np.min(energy)),
            "max": float(np.max(energy)),
            "range": float(np.max(energy) - np.min(energy)),
        }

        # 강세 감지
        peaks, _ = librosa.find_peaks(
            energy,
            height=np.mean(energy) + np.std(energy),
            distance=int(self.sample_rate / self.hop_length),  # 최소 1초 간격
        )

        # 구간별 분석 (TextGrid가 있는 경우)
        segment_analysis = None
        if textgrid_path:
            segment_analysis = self._analyze_energy_segments(
                energy,
                textgrid_path,
            )

        return {
            "statistics": energy_stats,
            "stress_count": len(peaks),
            "stress_times": [
                float(t) for t in peaks * self.hop_length / self.sample_rate
            ],
            "segments": segment_analysis,
        }

    def analyze_time_domain(
        self,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """시간 영역 분석

        Args:
            y: 오디오 신호

        Returns:
            Dict[str, Any]: 시간 영역 분석 결과
        """
        # 음성 구간 감지
        intervals = librosa.effects.split(
            y,
            top_db=CURRENT_CONFIG["audio"]["top_db"],
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        # 구간 길이 계산
        durations = []
        for start, end in intervals:
            duration = (end - start) * self.hop_length / self.sample_rate
            durations.append(float(duration))

        # 통계 계산
        total_duration = float(len(y) / self.sample_rate)
        speech_duration = float(sum(durations))
        silence_duration = total_duration - speech_duration

        return {
            "total_duration": total_duration,
            "speech_duration": speech_duration,
            "silence_duration": silence_duration,
            "speech_ratio": speech_duration / total_duration,
            "num_segments": len(intervals),
            "segment_durations": durations,
        }

    def analyze_sentence_structure(
        self,
        textgrid_path: str,
    ) -> Optional[Dict[str, Any]]:
        """문장 구조 분석

        Args:
            textgrid_path: TextGrid 파일 경로

        Returns:
            Optional[Dict[str, Any]]: 문장 구조 분석 결과
        """
        try:
            import textgrid

            # TextGrid 로드
            tg = textgrid.TextGrid.fromFile(textgrid_path)

            # 단어 정보 추출
            word_tier = tg.getFirst("words")
            words = []
            for interval in word_tier:
                if interval.mark:
                    words.append(
                        {
                            "text": interval.mark,
                            "start": float(interval.minTime),
                            "end": float(interval.maxTime),
                            "duration": float(interval.maxTime - interval.minTime),
                        }
                    )

            # 음절 정보 추출
            syllable_tier = tg.getFirst("syllables")
            syllables = []
            for interval in syllable_tier:
                if interval.mark:
                    syllables.append(
                        {
                            "text": interval.mark,
                            "start": float(interval.minTime),
                            "end": float(interval.maxTime),
                            "duration": float(interval.maxTime - interval.minTime),
                        }
                    )

            # 통계 계산
            word_durations = [w["duration"] for w in words]
            syllable_durations = [s["duration"] for s in syllables]

            return {
                "word_count": len(words),
                "syllable_count": len(syllables),
                "words_per_syllable": len(words) / len(syllables) if syllables else 0,
                "word_durations": {
                    "mean": float(np.mean(word_durations)),
                    "std": float(np.std(word_durations)),
                    "min": float(np.min(word_durations)),
                    "max": float(np.max(word_durations)),
                },
                "syllable_durations": {
                    "mean": float(np.mean(syllable_durations)),
                    "std": float(np.std(syllable_durations)),
                    "min": float(np.min(syllable_durations)),
                    "max": float(np.max(syllable_durations)),
                },
                "words": words,
                "syllables": syllables,
            }

        except Exception as e:
            logger.error(f"문장 구조 분석 중 오류가 발생했습니다: {str(e)}")
            return None

    def compare_prosody(
        self,
        learner_audio: str,
        reference_audio: str,
        learner_text: Optional[str] = None,
        learner_timing: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """학습자와 원어민의 운율 비교

        Args:
            learner_audio: 학습자 오디오 파일 경로
            reference_audio: 원어민 오디오 파일 경로
            learner_text: 학습자 전사 텍스트 (선택사항)
            learner_timing: 학습자 타이밍 정보 (선택사항)

        Returns:
            Optional[Dict[str, Any]]: 비교 결과
        """
        try:
            # 학습자 오디오 분석
            learner_result = self.analyze_audio(learner_audio)
            if not learner_result:
                return None

            # 원어민 오디오 분석
            reference_result = self.analyze_audio(reference_audio)
            if not reference_result:
                return None

            # 피치 비교
            pitch_diff = {
                "mean": learner_result["pitch"]["statistics"]["mean"]
                - reference_result["pitch"]["statistics"]["mean"],
                "std": learner_result["pitch"]["statistics"]["std"]
                - reference_result["pitch"]["statistics"]["std"],
                "range": learner_result["pitch"]["statistics"]["range"]
                - reference_result["pitch"]["statistics"]["range"],
            }

            # 에너지 비교
            energy_diff = {
                "mean": learner_result["energy"]["statistics"]["mean"]
                - reference_result["energy"]["statistics"]["mean"],
                "std": learner_result["energy"]["statistics"]["std"]
                - reference_result["energy"]["statistics"]["std"],
                "range": learner_result["energy"]["statistics"]["range"]
                - reference_result["energy"]["statistics"]["range"],
            }

            # 시간 영역 비교
            time_diff = {
                "total_duration": learner_result["time"]["total_duration"]
                - reference_result["time"]["total_duration"],
                "speech_duration": learner_result["time"]["speech_duration"]
                - reference_result["time"]["speech_duration"],
                "silence_duration": learner_result["time"]["silence_duration"]
                - reference_result["time"]["silence_duration"],
            }

            return {
                "learner": learner_result,
                "reference": reference_result,
                "differences": {
                    "pitch": pitch_diff,
                    "energy": energy_diff,
                    "time": time_diff,
                },
            }

        except Exception as e:
            logger.error(f"운율 비교 중 오류가 발생했습니다: {str(e)}")
            return None

    def visualize_prosody(
        self, prosody_result: Dict[str, Any], output_path: str = None
    ) -> None:
        """운율 분석 결과 시각화"""
        try:
            # ✅ output_path가 None이면 기본 경로 사용하지 않고 에러 방지
            if output_path is None:
                logger.warning("시각화 출력 경로가 지정되지 않았습니다.")
                return

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # 피치 분석 시각화
            pitch_data = prosody_result.get("pitch", {})
            if pitch_data:
                pitch_contour = pitch_data.get("contour", [])
                times = range(len(pitch_contour))
                axes[0].plot(times, pitch_contour, label="Pitch")
                axes[0].set_title("Pitch Analysis")
                axes[0].set_ylabel("Frequency (Hz)")
                axes[0].legend()

            # 에너지 분석 시각화
            energy_data = prosody_result.get("energy", {})
            if energy_data:
                energy_contour = energy_data.get("contour", [])
                times = range(len(energy_contour))
                axes[1].plot(times, energy_contour, label="Energy", color="orange")
                axes[1].set_title("Energy Analysis")
                axes[1].set_ylabel("Energy")
                axes[1].legend()

            # 타이밍 분석 시각화
            timing_data = prosody_result.get("timing", {})
            if timing_data:
                durations = timing_data.get("segment_durations", [])
                segments = range(len(durations))
                axes[2].bar(segments, durations, label="Segment Duration", color="green")
                axes[2].set_title("Timing Analysis")
                axes[2].set_xlabel("Segments")
                axes[2].set_ylabel("Duration (s)")
                axes[2].legend()

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"📈 운율 시각화 저장: {output_path}")

        except Exception as e:
            logger.error(f"운율 시각화 실패: {e}")

    def _analyze_pitch_segments(
        self,
        pitches: np.ndarray,
        magnitudes: np.ndarray,
        textgrid_path: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """구간별 피치 분석

        Args:
            pitches: 피치 배열
            magnitudes: 크기 배열
            textgrid_path: TextGrid 파일 경로

        Returns:
            Optional[List[Dict[str, Any]]]: 구간별 분석 결과
        """
        try:
            import textgrid

            # TextGrid 로드
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            word_tier = tg.getFirst("words")

            segments = []
            for interval in word_tier:
                if not interval.mark:
                    continue

                # 구간에 해당하는 프레임 인덱스 계산
                start_frame = int(interval.minTime * self.sample_rate / self.hop_length)
                end_frame = int(interval.maxTime * self.sample_rate / self.hop_length)

                # 구간의 피치 추출
                segment_pitches = pitches[:, start_frame:end_frame]
                segment_magnitudes = magnitudes[:, start_frame:end_frame]

                # 유효한 피치 값만 선택
                valid_pitches = segment_pitches[
                    segment_magnitudes > np.median(segment_magnitudes)
                ]

                if len(valid_pitches) > 0:
                    segments.append(
                        {
                            "text": interval.mark,
                            "start": float(interval.minTime),
                            "end": float(interval.maxTime),
                            "duration": float(interval.maxTime - interval.minTime),
                            "mean_pitch": float(np.mean(valid_pitches)),
                            "std_pitch": float(np.std(valid_pitches)),
                            "min_pitch": float(np.min(valid_pitches)),
                            "max_pitch": float(np.max(valid_pitches)),
                        }
                    )

            return segments

        except Exception as e:
            logger.error(f"구간별 피치 분석 중 오류가 발생했습니다: {str(e)}")
            return None

    def _analyze_energy_segments(
        self,
        energy: np.ndarray,
        textgrid_path: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """구간별 에너지 분석

        Args:
            energy: 에너지 배열
            textgrid_path: TextGrid 파일 경로

        Returns:
            Optional[List[Dict[str, Any]]]: 구간별 분석 결과
        """
        try:
            import textgrid

            # TextGrid 로드
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            word_tier = tg.getFirst("words")

            segments = []
            for interval in word_tier:
                if not interval.mark:
                    continue

                # 구간에 해당하는 프레임 인덱스 계산
                start_frame = int(interval.minTime * self.sample_rate / self.hop_length)
                end_frame = int(interval.maxTime * self.sample_rate / self.hop_length)

                # 구간의 에너지 추출
                segment_energy = energy[start_frame:end_frame]

                if len(segment_energy) > 0:
                    segments.append(
                        {
                            "text": interval.mark,
                            "start": float(interval.minTime),
                            "end": float(interval.maxTime),
                            "duration": float(interval.maxTime - interval.minTime),
                            "mean_energy": float(np.mean(segment_energy)),
                            "std_energy": float(np.std(segment_energy)),
                            "min_energy": float(np.min(segment_energy)),
                            "max_energy": float(np.max(segment_energy)),
                        }
                    )

            return segments

        except Exception as e:
            logger.error(f"구간별 에너지 분석 중 오류가 발생했습니다: {str(e)}")
            return None

    def adjust_thresholds(self, reference_audio: str) -> None:
        """원어민 오디오를 기반으로 임계값 조정

        Args:
            reference_audio: 원어민 오디오 파일 경로
        """
        try:
            # 원어민 오디오 분석
            reference_result = self.analyze_audio(reference_audio)
            if not reference_result:
                return

            # 피치 임계값 조정
            pitch_stats = reference_result["pitch"]["statistics"]
            self.pitch_threshold = pitch_stats["mean"] * 0.1  # 평균의 10%를 임계값으로 설정

            # 에너지 임계값 조정
            energy_stats = reference_result["energy"]["statistics"]
            self.energy_threshold = energy_stats["mean"] * 0.1  # 평균의 10%를 임계값으로 설정

            # 무음 임계값 조정
            self.silence_threshold = energy_stats["min"] * 1.5  # 최소 에너지의 1.5배를 임계값으로 설정

        except Exception as e:
            logger.error(f"임계값 조정 중 오류가 발생했습니다: {str(e)}")
