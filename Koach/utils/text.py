import re
import logging
from typing import Dict, List, Optional
import textgrid
import numpy as np

logger = logging.getLogger("Koach")


def summarize_textgrid(textgrid_path: str) -> Optional[Dict]:
    """TextGrid 파일에서 음소 정보 추출 및 요약"""
    try:
        # TextGrid 파일 로드
        tg = textgrid.TextGrid.fromFile(textgrid_path)

        # 음소 정보 추출
        phonemes = []
        for tier in tg.tiers:
            if tier.name.lower() in ["phones", "phoneme", "phone"]:
                for interval in tier:
                    if interval.mark.strip():
                        phonemes.append(
                            {
                                "text": interval.mark,
                                "start": interval.minTime,
                                "end": interval.maxTime,
                                "duration": interval.maxTime - interval.minTime,
                            }
                        )

        if not phonemes:
            logger.warning("TextGrid 파일에서 음소 정보를 찾을 수 없습니다.")
            return None

        # 음소 통계 계산
        durations = [p["duration"] for p in phonemes]
        total_duration = sum(durations)
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)

        # 결과 반환
        return {
            "phonemes": phonemes,
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "std_duration": std_duration,
            "num_phonemes": len(phonemes),
        }

    except Exception as e:
        logger.error(f"TextGrid 파일 분석 중 오류 발생: {e}")
        return None


def extract_word_boundaries(textgrid_path: str) -> Optional[List[Dict]]:
    """TextGrid 파일에서 단어 경계 정보 추출"""
    try:
        # TextGrid 파일 로드
        tg = textgrid.TextGrid.fromFile(textgrid_path)

        # 단어 정보 추출
        words = []
        for tier in tg.tiers:
            if tier.name.lower() in ["words", "word"]:
                for interval in tier:
                    if interval.mark.strip():
                        words.append(
                            {
                                "text": interval.mark,
                                "start": interval.minTime,
                                "end": interval.maxTime,
                                "duration": interval.maxTime - interval.minTime,
                            }
                        )

        if not words:
            logger.warning("TextGrid 파일에서 단어 정보를 찾을 수 없습니다.")
            return None

        return words

    except Exception as e:
        logger.error(f"단어 경계 추출 중 오류 발생: {e}")
        return None


def align_text_with_audio(
    text: str, phonemes: List[Dict], words: List[Dict]
) -> Optional[Dict]:
    """텍스트를 음소와 단어 정보와 정렬"""
    try:
        # 텍스트 전처리
        text = text.strip()

        # 단어 단위로 분리
        text_words = text.split()

        if len(text_words) != len(words):
            logger.warning("텍스트와 단어 정보의 개수가 일치하지 않습니다.")
            return None

        # 정렬된 결과 생성
        aligned_result = {"text": text, "words": []}

        # 각 단어에 대해 음소 정보 매핑
        for word, word_info in zip(text_words, words):
            word_phonemes = [
                p
                for p in phonemes
                if p["start"] >= word_info["start"] and p["end"] <= word_info["end"]
            ]

            aligned_result["words"].append(
                {
                    "text": word,
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "duration": word_info["duration"],
                    "phonemes": word_phonemes,
                }
            )

        return aligned_result

    except Exception as e:
        logger.error(f"텍스트 정렬 중 오류 발생: {e}")
        return None


def extract_phoneme_features(phonemes: List[Dict]) -> Optional[Dict]:
    """음소 특성 추출"""
    try:
        if not phonemes:
            return None

        # 음소별 지속 시간
        durations = [p["duration"] for p in phonemes]

        # 음소별 시작/종료 시간
        starts = [p["start"] for p in phonemes]
        ends = [p["end"] for p in phonemes]

        # 음소 간 간격
        gaps = [starts[i] - ends[i - 1] for i in range(1, len(starts))]

        # 통계 계산
        features = {
            "total_duration": sum(durations),
            "avg_duration": np.mean(durations),
            "std_duration": np.std(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_gap": np.mean(gaps) if gaps else 0,
            "std_gap": np.std(gaps) if gaps else 0,
            "num_phonemes": len(phonemes),
        }

        return features

    except Exception as e:
        logger.error(f"음소 특성 추출 중 오류 발생: {e}")
        return None


def compare_phoneme_sequences(
    reference_phonemes: List[Dict], target_phonemes: List[Dict]
) -> Optional[Dict]:
    """두 음소 시퀀스 비교"""
    try:
        if not reference_phonemes or not target_phonemes:
            return None

        # 음소 텍스트 추출
        ref_texts = [p["text"] for p in reference_phonemes]
        target_texts = [p["text"] for p in target_phonemes]

        # 음소 지속 시간 추출
        ref_durations = [p["duration"] for p in reference_phonemes]
        target_durations = [p["duration"] for p in target_phonemes]

        # 비교 결과
        result = {
            "reference_length": len(reference_phonemes),
            "target_length": len(target_phonemes),
            "duration_ratio": sum(target_durations) / sum(ref_durations),
            "phoneme_matches": sum(
                1 for r, t in zip(ref_texts, target_texts) if r == t
            ),
            "duration_diffs": (
                [t - r for r, t in zip(ref_durations, target_durations)]
                if len(ref_durations) == len(target_durations)
                else []
            ),
        }

        # 일치율 계산
        result["match_rate"] = result["phoneme_matches"] / max(
            len(ref_texts), len(target_texts)
        )

        return result

    except Exception as e:
        logger.error(f"음소 시퀀스 비교 중 오류 발생: {e}")
        return None
