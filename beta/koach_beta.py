#!/usr/bin/env python3
"""
Koach Beta - 한국어 발음 평가 및 피드백 시스템 (단일 파일 버전)

이 파일은 모든 기능이 포함된 단일 실행 파일입니다.
별도의 모듈 설치 없이 바로 실행할 수 있습니다.

필수 패키지:
    pip install openai whisper-openai pydub textgrid sentence-transformers faiss-cpu numpy

사용법:
    python koach_beta.py [learner_audio] [native_audio] [script_text]
    
예시:
    python koach_beta.py input/learner.m4a input/native.m4a "안녕하세요"
    python koach_beta.py  # 기본 파일 사용

환경 변수:
    export OPENAI_API_KEY="your_openai_api_key"
"""

import os
import sys
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
import json
from sentence_transformers import SentenceTransformer
import faiss
import torch

# =============================================================================
# 베타 버전 설정 (Beta Configuration)
# =============================================================================

BETA_CONFIG = {
    # 입력 파일 경로 (beta 폴더 기준)
    "learner_audio": "input/learner.m4a",
    "native_audio": "input/native.m4a",
    
    # 출력 디렉토리  
    "output_dir": "output",
    "wav_dir": "output/wav",
    "mfa_input_dir": "output/mfa_input",
    "mfa_output_dir": "output/aligned",
    
    # 모델 파일 경로 (상위 폴더의 models 디렉토리)
    "lexicon_path": "../models/korean_mfa.dict", 
    "acoustic_model": "../models/korean_mfa.zip",
    
    # AI 모델 설정
    "whisper_model": "base",  # tiny, base, small, medium, large
    "openai_model": "gpt-4o",
    
    # RAG 설정
    "use_rag": True,
    "knowledge_dir": "knowledge", 
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
}

# =============================================================================
# 로깅 설정 (Logging Configuration)  
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("KoachBeta")


# =============================================================================
# 지식 베이스 클래스 (Knowledge Base)
# =============================================================================

class KnowledgeBase:
    """한국어 발음 지식 베이스"""

    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Args:
            knowledge_dir: 지식 베이스 디렉토리
            embedding_model: 임베딩 모델명
        """
        self.knowledge_dir = knowledge_dir
        os.makedirs(knowledge_dir, exist_ok=True)

        # 문서 저장소
        self.documents = []
        self.document_ids = []

        # 임베딩 모델 로드
        self.model = SentenceTransformer(embedding_model)

        # FAISS 인덱스 초기화
        self.index = None

        # 기본 지식 로드
        self.load_knowledge()

    def load_knowledge(self):
        """지식 베이스 문서 로드"""
        logger.info("📚 지식 베이스 로드 중...")

        # 지식 디렉토리 내 모든 JSON 파일 로드
        file_paths = list(Path(self.knowledge_dir).glob("*.json"))

        if not file_paths:
            logger.warning("지식 베이스가 비어 있습니다. 기본 지식을 생성합니다.")
            self.create_default_knowledge()
            file_paths = list(Path(self.knowledge_dir).glob("*.json"))

        # 문서 로드 및 임베딩 생성
        documents = []
        document_ids = []

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data:
                    if "content" in item and "id" in item:
                        documents.append(item["content"])
                        document_ids.append(item["id"])
            except Exception as e:
                logger.error(f"파일 로드 실패 ({file_path}): {e}")

        # 문서 저장
        self.documents = documents
        self.document_ids = document_ids

        # 임베딩 생성 및 인덱스 구축
        self.build_index()

        logger.info(f"✅ {len(self.documents)}개 문서 로드 완료")

    def create_default_knowledge(self):
        """기본 지식 생성"""
        logger.info("기본 한국어 발음 지식 생성 중...")

        # 한국어 발음 기본 지식
        basic_knowledge = [
            {
                "id": "consonants_basic",
                "content": "한국어 자음(초성): ㄱ(g/k), ㄴ(n), ㄷ(d/t), ㄹ(r/l), ㅁ(m), ㅂ(b/p), ㅅ(s), ㅇ(silent/ng), ㅈ(j), ㅊ(ch), ㅋ(k), ㅌ(t), ㅍ(p), ㅎ(h). 각 자음은 위치와 발성 방법에 따라 구분된다.",
            },
            {
                "id": "vowels_basic",
                "content": "한국어 모음: ㅏ(a), ㅓ(eo), ㅗ(o), ㅜ(u), ㅡ(eu), ㅣ(i), ㅐ(ae), ㅔ(e), ㅚ(oe), ㅟ(wi). 입 모양과 혀의 위치가 발음에 중요하다.",
            },
            {
                "id": "final_consonants",
                "content": "한국어 받침(종성): 받침은 단어 끝에 오는 자음으로, 발음이 약화되거나 변형될 수 있다. 주요 받침 발음: ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅇ. 특히 ㄱ, ㄷ, ㅂ은 단어 끝에서 불파음으로 발음한다.",
            },
            {
                "id": "pronunciation_rules",
                "content": "한국어 발음 규칙: 1) 연음 현상: 받침이 다음 음절의 첫 소리로 넘어가는 현상 2) 자음 동화: 인접한 자음끼리 서로 영향을 주어 비슷한 소리로 변하는 현상 3) 구개음화: ㄷ, ㅌ이 ㅣ 모음 앞에서 ㅈ, ㅊ으로 변하는 현상 4) 경음화: 특정 환경에서 평음이 경음으로 변하는 현상",
            },
            {
                "id": "common_foreigner_errors",
                "content": "외국인 학습자들의 흔한 발음 실수: 1) ㅓ와 ㅗ 구별 어려움 2) ㄹ 발음 (영어 L/R과 다름) 3) 받침 발음 누락 4) 경음(ㄲ,ㄸ,ㅃ,ㅆ,ㅉ)과 평음 구별 어려움 5) 장단음 구별 부족 6) 연음 규칙 적용 실패",
            },
            {
                "id": "rhythm_intonation",
                "content": "한국어 리듬과 억양: 한국어는 음절 단위의 리듬을 가진다. 문장의 의미에 따라 억양 패턴이 달라진다. 의문문은 끝이 올라가고, 평서문은 내려간다. 감정 표현에 따라 억양의 굴곡이 커질 수 있다.",
            },
            {
                "id": "practice_techniques",
                "content": "발음 연습 기법: 1) 입 모양 거울로 확인하며 연습 2) 녹음하고 원어민과 비교 3) 느리게 발음한 후 점차 속도 높이기 4) 최소대립쌍(비슷한 소리 단어) 연습 5) 혀 위치 의식하며 발음하기 6) 과장된 발음으로 시작해 자연스럽게 조절",
            },
            {
                "id": "tense_consonants",
                "content": "경음(된소리) 발음: ㄲ, ㄸ, ㅃ, ㅆ, ㅉ는 성대를 긴장시키고 강하게 발음한다. 평음과 비교: '가다(gada)'와 '까다(kkada)', '바다(bada)'와 '빠다(ppada)'의 차이 인식하기.",
            },
            {
                "id": "aspirated_consonants",
                "content": "격음(거센소리) 발음: ㅋ, ㅌ, ㅍ, ㅊ는 강한 공기를 내보내며 발음한다. '카메라(kamera)', '타다(tada)', '파다(pada)', '차다(chada)'에서 거센 소리를 느낄 수 있다.",
            },
            {
                "id": "linking_sounds",
                "content": "연음 현상: 받침이 있는 단어 뒤에 모음으로 시작하는 조사나 어미가 오면, 받침은 다음 음절의 초성처럼 발음된다. 예: '꽃이(꼬치)', '밥을(바블)', '책은(채근)'",
            },
        ]

        # 지식 저장
        with open(
            os.path.join(self.knowledge_dir, "basic_pronunciation.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(basic_knowledge, f, ensure_ascii=False, indent=2)

        logger.info("✅ 기본 지식 생성 완료")

    def build_index(self):
        """FAISS 인덱스 구축"""
        if not self.documents:
            logger.warning("인덱스를 구축할 문서가 없습니다.")
            return

        # 문서 임베딩 생성
        embeddings = self.model.encode(self.documents)

        # FAISS 인덱스 생성
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

        logger.info(f"✅ {len(self.documents)}개 문서에 대한 인덱스 구축 완료")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """질의에 관련된 지식 검색

        Args:
            query: 검색 질의
            top_k: 반환할 문서 수

        Returns:
            List[Dict]: 검색된 문서 리스트
        """
        if not self.index or not self.documents:
            logger.warning("검색할 인덱스가 없습니다.")
            return []

        # 질의 임베딩
        query_embedding = self.model.encode([query])

        # 유사도 검색
        scores, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            k=min(top_k, len(self.documents)),
        )

        # 결과 정리
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append(
                    {
                        "id": self.document_ids[idx],
                        "content": self.documents[idx],
                        "score": float(scores[0][i]),
                    }
                )

        return results

    def add_document(self, doc_id: str, content: str):
        """새 문서 추가 (실시간 추가용)"""
        self.documents.append(content)
        self.document_ids.append(doc_id)
        
        # 인덱스 재구축 필요
        self.build_index()


# =============================================================================
# 메인 발음 분석 클래스 (Main Pronunciation Analysis)
# =============================================================================

class Koach:
    """한국어 발음 평가 및 피드백 시스템"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 파라미터 (없으면 기본값 사용)
        """
        # 베타 버전 기본 설정 사용
        self.config = BETA_CONFIG.copy()

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

        # RAG 지식 베이스 초기화 (설정에 따라)
        if self.config["use_rag"]:
            self.knowledge_base = KnowledgeBase(
                knowledge_dir=self.config["knowledge_dir"],
                embedding_model=self.config["embedding_model"],
            )
        else:
            self.knowledge_base = None

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
        os.makedirs(self.config["output_dir"], exist_ok=True)
        if self.config["use_rag"]:
            os.makedirs(self.config["knowledge_dir"], exist_ok=True)

    def convert_audio(self, input_path: str, output_path: str) -> bool:
        """오디오 파일을 WAV 형식으로 변환"""
        try:
            logger.info(f"🎧 변환 중: {input_path} → {output_path}")
            
            if not os.path.exists(input_path):
                logger.error(f"입력 파일이 존재하지 않습니다: {input_path}")
                return False
                
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_path, format="wav")
            logger.info("✅ 오디오 변환 완료")
            return True
        except Exception as e:
            logger.error(f"오디오 변환 실패: {e}")
            return False

    def transcribe_audio(self, wav_path: str, transcript_path: str) -> Optional[str]:
        """Whisper를 사용하여 오디오 파일 전사"""
        try:
            logger.info(f"📝 Whisper 전사 중: {wav_path}")
            model = whisper.load_model(self.config["whisper_model"])
            result = model.transcribe(wav_path, language="ko", word_timestamps=True)

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            logger.info(f"📄 전사 결과: {result['text']}")
            return result["text"]
        except Exception as e:
            logger.error(f"전사 실패: {e}")
            return None

    def run_mfa_alignment(
        self, wav_path: str, transcript_path: str, output_name: str
    ) -> bool:
        """MFA를 사용하여 오디오와 텍스트 정렬"""
        try:
            logger.info(f"🔧 MFA 정렬 시작: {output_name}")

            # MFA 모델 파일 확인
            if not os.path.exists(self.lexicon_path):
                logger.error(f"MFA 사전 파일이 없습니다: {self.lexicon_path}")
                logger.info("Korean MFA 모델을 다운로드하세요:")
                logger.info("mfa download acoustic korean_mfa")
                logger.info("mfa download dictionary korean_mfa")
                return False

            if not os.path.exists(self.acoustic_model):
                logger.error(f"MFA 음성 모델이 없습니다: {self.acoustic_model}")
                return False

            # 파일 복사
            shutil.copy(wav_path, os.path.join(self.mfa_input, f"{output_name}.wav"))
            shutil.copy(
                transcript_path, os.path.join(self.mfa_input, f"{output_name}.txt")
            )

            # MFA 정렬 실행
            command = [
                "mfa",
                "align",
                self.mfa_input,
                self.lexicon_path,
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
        """TextGrid 파일에서 압축된 음소 정보 추출"""
        try:
            logger.info(f"📊 TextGrid 요약 중: {path}")
            tg = textgrid.TextGrid.fromFile(path)
            
            # 간단한 요약 형태로 변경 (토큰 수 절약)
            phonemes = []
            
            for tier in tg.tiers:
                if hasattr(tier, 'intervals'):
                    for interval in tier.intervals:
                        if interval.mark and interval.mark.strip():
                            duration = round(interval.maxTime - interval.minTime, 2)
                            phonemes.append(f"{interval.mark}({duration}s)")
            
            summary = " | ".join(phonemes)
            logger.info(f"📝 음소 요약: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.error(f"TextGrid 요약 실패: {e}")
            return None

    def extract_pronunciation_issues(
        self, learner_text: str, native_text: str, learner_timing: str
    ) -> List[str]:
        """발음 문제점 추출 (간소화 버전)"""
        issues = []
        
        # 기본적인 텍스트 비교
        if learner_text != native_text:
            issues.append(f"텍스트 차이: '{learner_text}' vs '{native_text}'")
        
        # 타이밍 정보가 있으면 추가
        if learner_timing:
            issues.append(f"타이밍: {learner_timing[:200]}...")
            
        return issues

    def generate_prompt(
        self,
        learner_text: str,
        native_text: str,
        script_text: str,
        learner_timing: str,
        native_timing: str,
    ) -> str:
        """GPT용 발음 분석 프롬프트 생성 (압축된 버전)"""
        
        # RAG 검색으로 관련 지식 가져오기
        rag_context = ""
        if self.knowledge_base:
            query = f"한국어 발음 {script_text} 교정 피드백"
            search_results = self.knowledge_base.search(query, top_k=2)
            
            if search_results:
                rag_context = "\n참고 지식:\n"
                for result in search_results:
                    rag_context += f"- {result['content'][:200]}...\n"

        prompt = f"""당신은 한국어 발음 교정 전문가입니다.

**학습 목표 문장**: {script_text}

**분석 데이터**:
- 학습자 발음: {learner_text}
- 원어민 발음: {native_text}
- 학습자 타이밍: {learner_timing[:300] if learner_timing else 'N/A'}...
- 원어민 타이밍: {native_timing[:300] if native_timing else 'N/A'}...

{rag_context}

**요청사항**:
1. 학습자와 원어민 발음의 주요 차이점 분석
2. 구체적인 발음 교정 포인트 제시
3. 실제적인 연습 방법 제안

**응답 형식**:
## 📊 발음 분석
[주요 차이점]

## 🎯 교정 포인트  
[구체적인 교정사항]

## 💡 연습 방법
[실용적인 연습법]

간결하고 실용적으로 답변해주세요."""

        return prompt

    def get_feedback(self, prompt: str) -> Optional[str]:
        """OpenAI GPT를 사용하여 발음 피드백 생성"""
        if not self.api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            return None

        try:
            logger.info("🤖 GPT 피드백 생성 중...")
            
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.config["openai_model"],
                messages=[
                    {"role": "system", "content": "당신은 한국어 발음 교정 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            feedback = response.choices[0].message.content
            logger.info("✅ GPT 피드백 생성 완료")
            return feedback
            
        except Exception as e:
            logger.error(f"GPT 피드백 생성 실패: {e}")
            return None

    def analyze_pronunciation(
        self,
        learner_audio: Optional[str] = None,
        native_audio: Optional[str] = None,
        script: Optional[str] = None,
    ) -> Dict:
        """전체 발음 분석 프로세스 실행"""
        
        result = {
            "status": "started",
            "learner_audio": learner_audio or self.learner_audio,
            "native_audio": native_audio or self.native_audio,
            "script": script,
            "steps": {},
            "errors": []
        }

        try:
            # 1. 오디오 파일 변환
            logger.info("🎯 1단계: 오디오 파일 변환")
            
            learner_path = learner_audio or self.learner_audio
            native_path = native_audio or self.native_audio
            
            if not self.convert_audio(learner_path, self.learner_wav):
                result["errors"].append("학습자 오디오 변환 실패")
                return result
                
            if not self.convert_audio(native_path, self.native_wav):
                result["errors"].append("원어민 오디오 변환 실패") 
                return result
                
            result["steps"]["audio_conversion"] = "성공"

            # 2. 음성 인식
            logger.info("🎯 2단계: 음성 인식")
            
            learner_text = self.transcribe_audio(self.learner_wav, self.learner_transcript)
            native_text = self.transcribe_audio(self.native_wav, self.native_transcript)
            
            if not learner_text or not native_text:
                result["errors"].append("음성 인식 실패")
                return result
                
            result["learner_text"] = learner_text
            result["native_text"] = native_text
            result["steps"]["speech_recognition"] = "성공"

            # 3. 스크립트 저장
            if script:
                with open(self.script_path, "w", encoding="utf-8") as f:
                    f.write(script)
                result["script_text"] = script

            # 4. MFA 정렬
            logger.info("🎯 3단계: MFA 정렬")
            
            learner_aligned = self.run_mfa_alignment(
                self.learner_wav, self.learner_transcript, "learner"
            )
            native_aligned = self.run_mfa_alignment(
                self.native_wav, self.native_transcript, "native"
            )
            
            if not learner_aligned or not native_aligned:
                logger.warning("MFA 정렬 실패, 기본 분석으로 진행")
                result["steps"]["mfa_alignment"] = "실패"
                learner_timing = ""
                native_timing = ""
            else:
                result["steps"]["mfa_alignment"] = "성공"
                
                # TextGrid 요약
                learner_timing = self.summarize_textgrid(self.learner_textgrid) or ""
                native_timing = self.summarize_textgrid(self.native_textgrid) or ""

            result["learner_timing"] = learner_timing
            result["native_timing"] = native_timing

            # 5. 발음 문제점 추출
            logger.info("🎯 4단계: 발음 분석")
            
            pronunciation_issues = self.extract_pronunciation_issues(
                learner_text, native_text, learner_timing
            )
            result["pronunciation_issues"] = pronunciation_issues

            # 6. GPT 피드백 생성
            logger.info("🎯 5단계: GPT 피드백 생성")
            
            prompt = self.generate_prompt(
                learner_text, native_text, script or "알 수 없음",
                learner_timing, native_timing
            )
            
            gpt_feedback = self.get_feedback(prompt)
            result["gpt_feedback"] = gpt_feedback
            result["prompt_used"] = prompt

            if gpt_feedback:
                result["steps"]["gpt_feedback"] = "성공"
            else:
                result["steps"]["gpt_feedback"] = "실패"

            result["status"] = "완료"
            logger.info("🎉 전체 분석 프로세스 완료!")

        except Exception as e:
            logger.error(f"분석 프로세스 중 오류: {e}")
            result["errors"].append(f"예기치 않은 오류: {str(e)}")
            result["status"] = "실패"

        return result


# =============================================================================
# 메인 실행 함수 (Main Execution)
# =============================================================================

def print_usage():
    """사용법 출력"""
    print("""
🎤 Koach Beta - 한국어 발음 분석 시스템

사용법:
    python koach_beta.py [learner_audio] [native_audio] [script_text]
    python koach_beta.py  # 기본 파일 사용

예시:
    python koach_beta.py input/learner.m4a input/native.m4a "안녕하세요"
    python koach_beta.py input/my_voice.wav input/teacher.wav "한국어 발음 연습"

필수 설정:
    export OPENAI_API_KEY="your_openai_api_key"

필수 패키지:
    pip install openai whisper-openai pydub textgrid sentence-transformers faiss-cpu numpy

MFA 모델 (선택사항):
    mfa download acoustic korean_mfa
    mfa download dictionary korean_mfa
    """)

def check_requirements():
    """필수 요구사항 확인"""
    errors = []
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
    # 필수 디렉토리 확인 및 생성
    required_dirs = ["input", "output", "knowledge"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"📁 {dir_name} 디렉토리 생성됨")
    
    return errors

def main():
    """메인 실행 함수"""
    print("🚀 Koach Beta - 한국어 발음 분석 시스템 시작")
    print("=" * 60)
    
    # 명령행 인수 처리
    if len(sys.argv) == 4:
        learner_audio = sys.argv[1]
        native_audio = sys.argv[2] 
        script_text = sys.argv[3]
        
        # 설정 업데이트
        config = BETA_CONFIG.copy()
        config["learner_audio"] = learner_audio
        config["native_audio"] = native_audio
        
        logger.info(f"📁 입력 파일: {learner_audio}, {native_audio}")
        logger.info(f"📝 스크립트: {script_text}")
        
    elif len(sys.argv) == 1:
        # 기본 설정 사용
        config = BETA_CONFIG.copy()
        script_text = "안녕하세요"  # 기본 스크립트
        
        logger.info("📁 기본 파일 사용")
        logger.info(f"📝 기본 스크립트: {script_text}")
        
    else:
        print_usage()
        return

    # 요구사항 확인
    requirement_errors = check_requirements()
    if requirement_errors:
        for error in requirement_errors:
            print(error)
        print("\n설정을 완료한 후 다시 실행해주세요.")
        return
    
    try:
        # Koach 인스턴스 생성
        logger.info("🔧 Koach 시스템 초기화 중...")
        koach = Koach(config)
        
        # 발음 분석 실행
        logger.info("🎯 발음 분석 시작...")
        result = koach.analyze_pronunciation(
            learner_audio=config.get("learner_audio"),
            native_audio=config.get("native_audio"), 
            script=script_text
        )
        
        if result and result.get("status") == "완료":
            # 터미널 결과 출력
            print("\n" + "="*60)
            print("🎯 발음 분석 결과")
            print("="*60)
            
            if result.get("gpt_feedback"):
                print("\n📋 GPT 발음 피드백:")
                print("-" * 50)
                print(result["gpt_feedback"])
            else:
                print("\n⚠️ GPT 피드백 생성에 실패했습니다.")
                
            # 기본 분석 정보
            print(f"\n📊 기본 분석 정보:")
            print(f"  학습자 발음: {result.get('learner_text', 'N/A')}")
            print(f"  원어민 발음: {result.get('native_text', 'N/A')}")
            print(f"  목표 스크립트: {result.get('script_text', script_text)}")
            
            # 처리 단계 상태
            print(f"\n🔧 처리 단계 상태:")
            for step, status in result.get("steps", {}).items():
                status_icon = "✅" if status == "성공" else "❌" if status == "실패" else "⚠️"
                print(f"  {status_icon} {step}: {status}")
            
            # JSON 결과 저장
            output_file = os.path.join(config["output_dir"], "analysis_result.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 상세 결과가 {output_file}에 저장되었습니다.")
            print("\n✅ 분석 완료!")
            
        else:
            print("\n❌ 분석 실패")
            if result and result.get("errors"):
                print("오류 내용:")
                for error in result["errors"]:
                    print(f"  - {error}")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"❌ 예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n👋 Koach Beta 종료")

if __name__ == "__main__":
    main()
