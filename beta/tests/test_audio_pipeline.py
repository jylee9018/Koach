import os
import sys
import asyncio
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from app.services.audio_processor import AudioProcessor
from app.services.force_aligner import ForceAligner
from app.services.audio_embedder import AudioEmbedder
from app.services.pronunciation_analyzer import PronunciationAnalyzer


async def test_audio_pipeline(audio_path: str, script: str):
    """오디오 처리 파이프라인 테스트"""
    logger.info(f"테스트 시작: {audio_path}")

    # 1. 오디오 프로세서 테스트
    logger.info("1. 오디오 프로세서 테스트")
    processor = AudioProcessor()
    features = await processor.process_audio(audio_path)
    logger.info(f"기본 피처 추출 완료: {list(features.keys())}")

    # 2. Forced Aligner 테스트
    logger.info("2. Forced Aligner 테스트")
    aligner = ForceAligner()
    try:
        alignment_result = await aligner.align(audio_path, script)
        logger.info(
            f"정렬 성공: {len(alignment_result.get('words', []))} 단어, {len(alignment_result.get('phonemes', []))} 음소"
        )
    except Exception as e:
        logger.error(f"정렬 실패: {e}")
        alignment_result = None

    # 3. 정렬 기반 피처 추출 테스트
    if alignment_result:
        logger.info("3. 정렬 기반 피처 추출 테스트")
        features_with_alignment = await processor.process_audio_with_alignment(
            audio_path, script
        )
        logger.info(f"정렬 기반 피처 추출 완료: {list(features_with_alignment.keys())}")

    # 4. 임베딩 추출 테스트
    logger.info("4. 임베딩 추출 테스트")
    embedder = AudioEmbedder()
    try:
        embedding = await embedder.extract_embeddings(audio_path)
        logger.info(f"임베딩 추출 완료: 차원={embedding.shape}")

        if alignment_result:
            word_embeddings = await embedder.extract_word_embeddings(
                audio_path, alignment_result
            )
            logger.info(f"단어 임베딩 추출 완료: {len(word_embeddings)} 단어")

            phoneme_embeddings = await embedder.extract_phoneme_embeddings(
                audio_path, alignment_result
            )
            logger.info(f"음소 임베딩 추출 완료: {len(phoneme_embeddings)} 음소")
    except Exception as e:
        logger.error(f"임베딩 추출 실패: {e}")

    # 5. 발음 분석 테스트
    logger.info("5. 발음 분석 테스트")
    analyzer = PronunciationAnalyzer()
    try:
        if alignment_result:
            analysis_result = await analyzer.analyze_with_alignment(audio_path, script)
            logger.info(
                f"발음 분석 완료: {analysis_result.get('similarity', 0):.2f} 유사도, {len(analysis_result.get('phoneme_errors', []))} 오류"
            )
    except Exception as e:
        logger.error(f"발음 분석 실패: {e}")

    logger.info("테스트 완료")


if __name__ == "__main__":
    # 테스트할 오디오 파일과 스크립트
    test_audio = "tests/2025-01_Sa.m4a"
    test_script = "영화 '부산행'을 봤어요. 이 영화는 연상호 감독의 좀비 영화예요. 이 영화는 연상호 감독이 처음으로 만든 실사 영화예요. 이 감독은 이 영화를 만들기 전에는 항상 애니메이션을 만들었어요. 그 애니메이션 영화들이 너무 좋아서, 이번에 새로운 영화에 대해서 기대를 많이 했어요. 저는 원래 좀비 영화를 좋아하지 않아요. 그런데 이 영화는 정말 재미있었어요."

    # 테스트 실행
    asyncio.run(test_audio_pipeline(test_audio, test_script))
