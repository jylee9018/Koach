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
import json
from sentence_transformers import SentenceTransformer
import faiss
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Koach")


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
        """새 문서 추가

        Args:
            doc_id: 문서 ID
            content: 문서 내용
        """
        # 문서 추가
        self.documents.append(content)
        self.document_ids.append(doc_id)

        # 인덱스 재구축
        self.build_index()


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
            # RAG 설정
            "use_rag": True,
            "knowledge_dir": "knowledge",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
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

        # RAG 지식 베이스 초기화 (설정에 따라)
        if self.config["use_rag"]:
            self.knowledge_base = KnowledgeBase(
                knowledge_dir=self.config["knowledge_dir"],
                embedding_model=self.config["embedding_model"],
            )
        else:
            self.knowledge_base = None

    # [기존 메소드들은 동일하게 유지]

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
        if self.config["use_rag"]:
            os.makedirs(self.config["knowledge_dir"], exist_ok=True)

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

    def run_mfa_alignment(
        self, wav_path: str, transcript_path: str, output_name: str
    ) -> bool:
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

    def extract_pronunciation_issues(
        self, learner_text: str, native_text: str, learner_timing: str
    ) -> List[str]:
        """발음 문제점 추출

        Args:
            learner_text: 학습자 텍스트
            native_text: 원어민 텍스트
            learner_timing: 학습자 타이밍 정보

        Returns:
            List[str]: 발견된 문제점 목록
        """
        # 간단한 휴리스틱으로 문제 추출
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
        for line in learner_timing.split("\n"):
            parts = line.split(":")
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

        # RAG가 활성화된 경우, 관련 지식 검색 및 추가
        if self.config["use_rag"] and self.knowledge_base:
            # 발음 문제점 추출
            issues = self.extract_pronunciation_issues(
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

            # RAG 관련 정보 출력 (활성화된 경우)
            if koach.config["use_rag"] and koach.knowledge_base:
                print("\n📚 사용된 지식 베이스 정보:")
                print(f"- 문서 수: {len(koach.knowledge_base.documents)}")
                print(f"- 임베딩 모델: {koach.config['embedding_model']}")
        else:
            print(f"\n❌ 오류 발생: {result['error']}")

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
