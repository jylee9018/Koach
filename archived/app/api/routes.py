from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from ..core.graph import koach_graph
from pydub import AudioSegment
import os
import logging
import json

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("api")

router = APIRouter(tags=["pronunciation"])


@router.post("/analyze")
async def analyze_pronunciation(
    learner_audio: UploadFile = File(...),
    native_audio: UploadFile = File(None),
    script: str = Form(...),
):
    try:
        # 학습자 오디오 저장
        learner_input_path = f"/tmp/learner_{learner_audio.filename}"
        learner_wav_path = f"/tmp/learner_converted_{learner_audio.filename}.wav"

        with open(learner_input_path, "wb") as buffer:
            buffer.write(await learner_audio.read())

        # 학습자 오디오 변환
        learner_ext = learner_audio.filename.split(".")[-1].lower()
        if learner_ext in ["m4a", "aac"]:
            audio_segment = AudioSegment.from_file(
                learner_input_path, format=learner_ext
            )
            audio_segment.export(learner_wav_path, format="wav")
            learner_path = learner_wav_path
        else:
            learner_path = learner_input_path

        # 원어민 오디오 처리 (제공된 경우)
        native_path = None
        native_input_path = None
        native_wav_path = None

        if native_audio:
            native_input_path = f"/tmp/native_{native_audio.filename}"
            native_wav_path = f"/tmp/native_converted_{native_audio.filename}.wav"

            with open(native_input_path, "wb") as buffer:
                buffer.write(await native_audio.read())

            # 원어민 오디오 변환
            native_ext = native_audio.filename.split(".")[-1].lower()
            if native_ext in ["m4a", "aac"]:
                audio_segment = AudioSegment.from_file(
                    native_input_path, format=native_ext
                )
                audio_segment.export(native_wav_path, format="wav")
                native_path = native_wav_path
            else:
                native_path = native_input_path

        # 초기 상태 설정
        initial_state = {
            "audio_path": learner_path,
            "script": script,
        }

        if native_path:
            initial_state["native_audio_path"] = native_path

        # 그래프 실행
        result = await koach_graph.ainvoke(initial_state)

        # 임시 파일 정리
        for path in [
            learner_input_path,
            learner_wav_path,
            native_input_path,
            native_wav_path,
        ]:
            if path and os.path.exists(path):
                os.remove(path)

        # 결과 반환
        return {
            "similarity_score": result.get("similarity_score", 0.0),
            "feedback": result.get("gpt_result", ""),
            "pronunciation_errors": result.get("pronunciation_errors", []),
            "phoneme_errors": result.get("phoneme_errors", []),
            "alignment": result.get("analysis_result", {}).get("alignment", {}),
        }
    except Exception as e:
        logger.error(f"Error in analyze_pronunciation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/align")
async def align_audio_text(
    audio: UploadFile = File(...),
    script: str = Form(...),
):
    """음성과 텍스트를 정렬하여 음소 단위 시간 정보 추출"""
    try:
        # 오디오 저장
        input_path = f"/tmp/align_{audio.filename}"
        wav_path = f"/tmp/align_converted_{audio.filename}.wav"

        with open(input_path, "wb") as buffer:
            buffer.write(await audio.read())

        # 오디오 변환
        audio_ext = audio.filename.split(".")[-1].lower()
        if audio_ext in ["m4a", "aac"]:
            audio_segment = AudioSegment.from_file(input_path, format=audio_ext)
            audio_segment.export(wav_path, format="wav")
            audio_path = wav_path
        else:
            audio_path = input_path

        # 초기 상태 설정
        initial_state = {
            "audio_path": audio_path,
            "script": script,
        }

        # 그래프의 extract_features 노드만 실행
        from ..services.force_aligner import ForceAligner

        aligner = ForceAligner()
        alignment_result = await aligner.align(audio_path, script)

        # 임시 파일 정리
        for path in [input_path, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)

        # 결과 반환
        return {
            "alignment": alignment_result,
        }
    except Exception as e:
        logger.error(f"Error in align_audio_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-embeddings")
async def extract_audio_embeddings(
    audio: UploadFile = File(...),
    script: str = Form(None),
):
    """음성에서 임베딩 추출"""
    try:
        # 오디오 저장
        input_path = f"/tmp/embed_{audio.filename}"
        wav_path = f"/tmp/embed_converted_{audio.filename}.wav"

        with open(input_path, "wb") as buffer:
            buffer.write(await audio.read())

        # 오디오 변환
        audio_ext = audio.filename.split(".")[-1].lower()
        if audio_ext in ["m4a", "aac"]:
            audio_segment = AudioSegment.from_file(input_path, format=audio_ext)
            audio_segment.export(wav_path, format="wav")
            audio_path = wav_path
        else:
            audio_path = input_path

        # 임베딩 추출
        from ..services.audio_embedder import AudioEmbedder

        embedder = AudioEmbedder()

        # 전체 임베딩 추출
        full_embedding = await embedder.extract_embeddings(audio_path)

        # 스크립트가 제공된 경우 단어/음소 임베딩도 추출
        word_embeddings = {}
        phoneme_embeddings = {}

        if script:
            # 정렬 수행
            from ..services.force_aligner import ForceAligner

            aligner = ForceAligner()
            alignment_result = await aligner.align(audio_path, script)

            # 단어 및 음소 임베딩 추출
            word_embeddings = await embedder.extract_word_embeddings(
                audio_path, alignment_result
            )
            phoneme_embeddings = await embedder.extract_phoneme_embeddings(
                audio_path, alignment_result
            )

        # 임시 파일 정리
        for path in [input_path, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)

        # 결과 반환 (임베딩은 크기가 크므로 차원 정보만 반환)
        return {
            "full_embedding_shape": full_embedding.shape,
            "word_embeddings_count": len(word_embeddings),
            "phoneme_embeddings_count": len(phoneme_embeddings),
            # 필요한 경우 실제 임베딩 값도 반환 가능
            # "full_embedding": full_embedding.tolist(),
        }
    except Exception as e:
        logger.error(f"Error in extract_audio_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_pronunciations(
    learner_audio: UploadFile = File(...),
    native_audio: UploadFile = File(...),
    script: str = Form(...),
):
    """학습자와 원어민 발음 비교"""
    try:
        # 학습자 오디오 저장 및 변환
        learner_input_path = f"/tmp/compare_learner_{learner_audio.filename}"
        learner_wav_path = (
            f"/tmp/compare_learner_converted_{learner_audio.filename}.wav"
        )

        with open(learner_input_path, "wb") as buffer:
            buffer.write(await learner_audio.read())

        learner_ext = learner_audio.filename.split(".")[-1].lower()
        if learner_ext in ["m4a", "aac"]:
            audio_segment = AudioSegment.from_file(
                learner_input_path, format=learner_ext
            )
            audio_segment.export(learner_wav_path, format="wav")
            learner_path = learner_wav_path
        else:
            learner_path = learner_input_path

        # 원어민 오디오 저장 및 변환
        native_input_path = f"/tmp/compare_native_{native_audio.filename}"
        native_wav_path = f"/tmp/compare_native_converted_{native_audio.filename}.wav"

        with open(native_input_path, "wb") as buffer:
            buffer.write(await native_audio.read())

        native_ext = native_audio.filename.split(".")[-1].lower()
        if native_ext in ["m4a", "aac"]:
            audio_segment = AudioSegment.from_file(native_input_path, format=native_ext)
            audio_segment.export(native_wav_path, format="wav")
            native_path = native_wav_path
        else:
            native_path = native_input_path

        # 초기 상태 설정
        initial_state = {
            "audio_path": learner_path,
            "native_audio_path": native_path,
            "script": script,
        }

        # 그래프 실행
        result = await koach_graph.ainvoke(initial_state)

        # 임시 파일 정리
        for path in [
            learner_input_path,
            learner_wav_path,
            native_input_path,
            native_wav_path,
        ]:
            if path and os.path.exists(path):
                os.remove(path)

        # 결과 반환
        return {
            "similarity_score": result.get("similarity_score", 0.0),
            "feedback": result.get("gpt_result", ""),
            "pronunciation_errors": result.get("pronunciation_errors", []),
            "phoneme_errors": result.get("phoneme_errors", []),
            "comparison_result": result.get("comparison_result", {}),
            "alignment": result.get("analysis_result", {}).get("alignment", {}),
        }
    except Exception as e:
        logger.error(f"Error in compare_pronunciations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
