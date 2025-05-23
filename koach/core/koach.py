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
from datetime import datetime

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
from config.settings import CURRENT_CONFIG, PATHS, OUTPUT_DIR, TEMP_ROOT, VISUALIZE_DIR

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
        self.output_dir = OUTPUT_DIR

        # 임시 디렉토리 설정
        self.temp_dir = TEMP_ROOT
        self.temp_dir.mkdir(exist_ok=True)

        # 베타 버전의 설정 구조 통합
        self.config = {
            # Whisper 모델 설정
            "whisper_model": "base",
            "language": "ko",
            # OpenAI 모델 (베타에서 개선된 모델)
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

        # 시각화 디렉토리 설정
        self.visualize_dir = VISUALIZE_DIR

    def analyze_pronunciation(
        self,
        learner_audio: Optional[str] = None,
        native_audio: Optional[str] = None,
        script: Optional[str] = None,
        visualize: bool = True,
    ) -> Dict:
        """발음 분석 실행"""
        try:
            result = {
                "steps": {},
                "errors": [],
                "status": "진행중",
            }

            # 1. 파일 경로 설정
            if not learner_audio:
                learner_audio = str(self.learner_audio)
            if not native_audio:
                native_audio = str(self.native_audio)

            # 📝 스크립트 파일 자동 로드 처리
            script_text = None
            if script:
                script_config = CURRENT_CONFIG["script"]
                
                if script_config["auto_detect_file"] and (
                    any(script.endswith(ext) for ext in script_config["supported_extensions"]) or 
                    '/' in script or '\\' in script
                ):
                    # 파일 경로로 판단
                    try:
                        script_path = Path(script)
                        if script_path.exists():
                            # 파일 크기 확인
                            if script_path.stat().st_size > script_config["max_file_size"]:
                                logger.warning(f"⚠️ 스크립트 파일이 너무 큽니다: {script_path}")
                                script_text = script
                            else:
                                with open(script_path, 'r', encoding=script_config["encoding"]) as f:
                                    script_text = f.read().strip()
                                logger.info(f"📄 스크립트 파일 로드됨: {script_path}")
                                logger.info(f"📝 스크립트 내용: {script_text[:50]}...")
                        else:
                            logger.warning(f"⚠️ 스크립트 파일을 찾을 수 없음: {script_path}")
                            script_text = script  # 파일이 없으면 원본 텍스트로 사용
                    except Exception as e:
                        logger.error(f"❌ 스크립트 파일 읽기 실패: {e}")
                        script_text = script  # 에러 시 원본 텍스트로 사용
                else:
                    # 직접 텍스트로 판단
                    script_text = script
                    logger.info(f"📝 스크립트 텍스트 직접 입력: {script_text[:50]}...")
                
                result["script_text"] = script_text

            # 2. 오디오 변환 및 정규화
            logger.info("🎯 1단계: 오디오 파일 변환 및 정규화")
            
            # WAV 변환
            convert_audio(learner_audio, str(self.learner_wav))
            convert_audio(native_audio, str(self.native_wav))
            
            # 정규화 경로 가져오기
            learner_normalized = self.get_normalized_paths("learner")["normalized"]
            native_normalized = self.get_normalized_paths("native")["normalized"]
            
            # 정규화 시도
            if not normalize_audio(self.learner_wav, learner_normalized):
                logger.warning("학습자 오디오 정규화 실패, 원본 사용")
                learner_normalized = self.learner_wav
            
            if not normalize_audio(self.native_wav, native_normalized):
                logger.warning("원어민 오디오 정규화 실패, 원본 사용")
                native_normalized = self.native_wav

            result["steps"]["audio_conversion"] = "성공"

            # 3. 음성 인식 (스크립트 제공 시에도 전사 실행)
            logger.info("🎯 2단계: 음성 인식")
            
            # 스크립트가 있어도 실제 발화 내용 확인을 위해 전사 실행
            learner_result = transcribe_audio(learner_normalized)
            native_result = transcribe_audio(native_normalized)
            
            if not learner_result or not native_result:
                if script_text:
                    # 전사 실패 시에만 스크립트 사용
                    logger.warning("⚠️ 음성 인식 실패, 스크립트 사용")
                    result["learner_text"] = script_text
                    result["native_text"] = script_text
                    result["steps"]["speech_recognition"] = "실패(스크립트 사용)"
                else:
                    raise Exception("음성 인식 실패")
            else:
                result["learner_text"] = learner_result.get("text", "")
                result["native_text"] = native_result.get("text", "")
                result["steps"]["speech_recognition"] = "성공"
                
                # 스크립트와 실제 발화 비교 로그
                if script_text:
                    logger.info(f"📋 목표: {script_text}")
                    logger.info(f"🎤 실제: {result['learner_text']}")

            # 4. 스크립트 파일 저장
            if script_text:
                with open(self.script_path, "w", encoding="utf-8") as f:
                    f.write(script_text)

            # 5. MFA 정렬 (최적화 및 건너뛰기 옵션) - 방법 3
            logger.info("🎯 3단계: MFA 정렬")
            
            # Whisper 전사 결과를 파일로 저장
            with open(self.learner_transcript, "w", encoding="utf-8") as f:
                f.write(result["learner_text"])
            with open(self.native_transcript, "w", encoding="utf-8") as f:
                f.write(result["native_text"])
            
            # MFA 건너뛰기 옵션 확인
            if CURRENT_CONFIG["mfa"].get("skip_mfa", False):
                logger.info("⚡ MFA 정렬 건너뛰기 (설정)")
                result["steps"]["mfa_alignment"] = "건너뜀"
                learner_timing = ""
                native_timing = ""
            else:
                # 배치 정렬 시도 (방법 1)
                alignment_success = False
                
                if CURRENT_CONFIG["mfa"].get("batch_processing", True):
                    try:
                        logger.info("🚀 배치 정렬 모드 시도...")
                        alignment_success = self.run_mfa_alignment_batch(
                            learner_normalized, native_normalized,
                            self.learner_transcript, self.native_transcript
                        )
                    except Exception as e:
                        logger.warning(f"배치 정렬 실패: {e}")
                
                # 배치 정렬 실패 시 기존 방식으로 백업
                if not alignment_success:
                    logger.info("🔄 기존 정렬 방식으로 백업...")
                    try:
                        learner_aligned = self.run_mfa_alignment_legacy(
                            learner_normalized, self.learner_transcript, "learner"
                        )
                        native_aligned = self.run_mfa_alignment_legacy(
                            native_normalized, self.native_transcript, "native"
                        )
                        alignment_success = learner_aligned and native_aligned
                    except Exception as e:
                        logger.warning(f"기존 정렬도 실패: {e}")
                        alignment_success = False

                # 결과 처리
                if alignment_success:
                    result["steps"]["mfa_alignment"] = "성공"
                    # 압축된 TextGrid 요약
                    learner_timing = self.summarize_textgrid_compact(self.learner_textgrid) or ""
                    native_timing = self.summarize_textgrid_compact(self.native_textgrid) or ""
                else:
                    logger.warning("MFA 정렬 완전 실패, 기본 분석으로 진행")
                    result["steps"]["mfa_alignment"] = "실패"
                    learner_timing = ""
                    native_timing = ""

            result["learner_timing"] = learner_timing
            result["native_timing"] = native_timing

            # 6. 발음 분석 (4단계)
            logger.info("🎯 4단계: 발음 분석")
            try:
                # 음소 분석
                phoneme_analysis = {}
                prosody_analysis = {}
                comparison = {}
                
                if alignment_success:
                    # 음소 분석
                    phoneme_analysis = self._analyze_phonemes(
                        str(self.learner_textgrid)
                    ) or {}
                    
                    # 운율 분석
                    prosody_analysis = self._analyze_prosody_detailed(
                        learner_normalized
                    ) or {}
                    
                    # 비교 분석 (원어민 오디오가 있는 경우)
                    if native_normalized and Path(native_normalized).exists():
                        comparison = self._compare_with_reference(
                            learner_normalized,
                            native_normalized,
                            str(self.learner_textgrid)
                        ) or {}
                
                # 기존 발음 문제점 추출 로직 유지
                pronunciation_issues = self.extract_pronunciation_issues_detailed(
                    result["learner_text"], result["native_text"], learner_timing
                )
                
                # 결과에 추가
                result["pronunciation_issues"] = pronunciation_issues
                result["phoneme_analysis"] = phoneme_analysis
                result["prosody_analysis"] = prosody_analysis
                result["comparison"] = comparison
                result["steps"]["pronunciation_analysis"] = "성공"
                
                logger.info("✅ 발음 분석 완료")
                
            except Exception as e:
                logger.error(f"발음 분석 실패: {e}")
                result["steps"]["pronunciation_analysis"] = "실패"
                result["errors"].append(f"발음 분석 오류: {str(e)}")
                # 빈 딕셔너리로 초기화
                phoneme_analysis = {}
                prosody_analysis = {}
                comparison = {}

            # 분석 결과에서 numpy 타입 변환
            if "phoneme_analysis" in result:
                result["phoneme_analysis"] = self._convert_numpy_types(result["phoneme_analysis"])
            
            if "prosody_analysis" in result:
                result["prosody_analysis"] = self._convert_numpy_types(result["prosody_analysis"])
            
            if "comparison" in result:
                result["comparison"] = self._convert_numpy_types(result["comparison"])

            # 7. GPT 피드백 생성 (5단계)
            logger.info("🎯 5단계: GPT 피드백 생성")
            
            prompt = self.generate_compact_prompt(
                result["learner_text"], result["native_text"], script_text or "알 수 없음",
                learner_timing, native_timing
            )
            
            gpt_feedback = self.get_feedback(prompt)
            result["feedback"] = gpt_feedback
            result["prompt_used"] = prompt

            if gpt_feedback:
                result["steps"]["gpt_feedback"] = "성공"
            else:
                result["steps"]["gpt_feedback"] = "실패"

            # 8. 시각화 생성 (6단계)
            if visualize and CURRENT_CONFIG["visualization"]["enabled"]:
                logger.info("🎯 6단계: 시각화 생성")
                try:
                    if alignment_success:
                        visualization_paths = self._visualize_results(
                            learner_audio=learner_normalized,
                            reference_audio=native_normalized,
                            learner_textgrid=str(self.learner_textgrid),
                            phoneme_analysis=phoneme_analysis,
                            prosody_analysis=prosody_analysis,
                            comparison=comparison
                        )
                        result["visualization_paths"] = visualization_paths
                        result["steps"]["visualization"] = "성공"
                        logger.info(f"✅ 시각화 완료: {len(visualization_paths)}개 파일 생성")
                    else:
                        logger.warning("⚠️ MFA 정렬 실패로 시각화 건너뛰기")
                        result["steps"]["visualization"] = "건너뜀"
                except Exception as e:
                    logger.error(f"시각화 생성 실패: {e}")
                    result["steps"]["visualization"] = "실패"
                    result["errors"].append(f"시각화 오류: {str(e)}")
            
            result["status"] = "완료"
            logger.info("🎉 전체 분석 프로세스 완료!")
            return result

        except Exception as e:
            logger.error(f"분석 프로세스 중 오류: {e}")
            result["errors"].append(f"예기치 않은 오류: {str(e)}")
            result["status"] = "실패"
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

    def run_mfa_alignment_batch(
        self, 
        learner_wav: str, 
        native_wav: str,
        learner_transcript: str,
        native_transcript: str
    ) -> bool:
        """MFA 배치 정렬 (학습자와 원어민을 동시에 처리) - 방법 1"""
        try:
            logger.info("🚀 MFA 배치 정렬 시작...")

            # MFA 입력 디렉토리 준비 (하나의 폴더에 모든 파일)
            mfa_batch_input = self.mfa_input / "batch"
            mfa_batch_input.mkdir(parents=True, exist_ok=True)
            
            # 기존 파일들 정리
            import shutil
            if mfa_batch_input.exists():
                shutil.rmtree(mfa_batch_input)
                mfa_batch_input.mkdir(parents=True, exist_ok=True)
            
            # 파일 복사 (같은 폴더에 배치)
            shutil.copy(learner_wav, str(mfa_batch_input / "learner.wav"))
            shutil.copy(native_wav, str(mfa_batch_input / "native.wav"))
            shutil.copy(learner_transcript, str(mfa_batch_input / "learner.txt"))
            shutil.copy(native_transcript, str(mfa_batch_input / "native.txt"))

            logger.info(f"📁 배치 입력 폴더: {mfa_batch_input}")
            logger.info(f"📄 파일들: learner.wav, native.wav, learner.txt, native.txt")

            # 최적화된 MFA 명령어
            command = [
                "mfa", "align",
                str(mfa_batch_input),           # 모든 파일이 있는 하나의 폴더
                str(self.lexicon_path),
                str(self.acoustic_model),
                str(self.mfa_output),
                "--num_jobs", str(CURRENT_CONFIG["mfa"]["num_jobs"]),  # 병렬 처리
                "--clean",                      # 이전 결과 정리
                "--no_debug",                   # 디버그 출력 비활성화
                "--ignore_empty_utterances",   # 빈 발화 무시
            ]

            logger.info(f"🚀 MFA 명령어: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=CURRENT_CONFIG["mfa"]["timeout"],  # 타임아웃 설정
            )

            if result.returncode != 0:
                logger.error(f"MFA 배치 정렬 실패: {result.stderr}")
                return False

            logger.info("✅ MFA 배치 정렬 완료")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"MFA 배치 정렬 시간 초과 ({CURRENT_CONFIG['mfa']['timeout']}초)")
            return False
        except Exception as e:
            logger.error(f"MFA 배치 정렬 실패: {e}")
            return False

    # 기존 run_mfa_alignment 함수는 백업으로 유지
    def run_mfa_alignment_legacy(
        self, wav_path: str, transcript_path: str, output_name: str
    ) -> bool:
        """기존 MFA 정렬 (백업용)"""
        try:
            logger.info(f"🔧 기존 MFA 정렬: {output_name}")

            # MFA 입력 디렉토리 준비
            mfa_input_dir = self.mfa_input / output_name
            mfa_input_dir.mkdir(parents=True, exist_ok=True)

            # 파일 복사
            target_wav = str(mfa_input_dir / f"{output_name}.wav")
            target_txt = str(mfa_input_dir / f"{output_name}.txt")

            if str(wav_path) != target_wav:
                shutil.copy(wav_path, target_wav)
            if str(transcript_path) != target_txt:
                shutil.copy(transcript_path, target_txt)

            # MFA 정렬 실행
            command = [
                "mfa", "align",
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
                timeout=60,  # 짧은 타임아웃
            )

            if result.returncode != 0:
                logger.error(f"기존 MFA 정렬 실패: {result.stderr}")
                return False

            logger.info("✅ 기존 MFA 정렬 완료")
            return True

        except Exception as e:
            logger.error(f"기존 MFA 정렬 실패: {e}")
            return False

    def _analyze_phonemes(
        self,
        textgrid_path: str,
    ) -> Optional[Dict[str, Any]]:
        """음소 분석"""
        try:
            import textgrid
            
            # TextGrid 파일 로드
            if not Path(textgrid_path).exists():
                logger.warning(f"TextGrid 파일이 없습니다: {textgrid_path}")
                return {}
            
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # 단어 tier와 음소 tier 찾기
            word_tier = None
            phone_tier = None
            
            for tier in tg.tiers:
                if 'words' in tier.name.lower():
                    word_tier = tier
                elif 'phones' in tier.name.lower():
                    phone_tier = tier
            
            phonemes = []
            words = []
            
            # 음소 정보 추출
            if phone_tier:
                for interval in phone_tier:
                    if interval.mark and interval.mark.strip():
                        phonemes.append({
                            "phoneme": interval.mark,
                            "start": float(interval.minTime),
                            "end": float(interval.maxTime),
                            "duration": float(interval.maxTime - interval.minTime)
                        })
            
            # 단어 정보 추출
            if word_tier:
                for interval in word_tier:
                    if interval.mark and interval.mark.strip():
                        words.append({
                            "word": interval.mark,
                            "start": float(interval.minTime),
                            "end": float(interval.maxTime),
                            "duration": float(interval.maxTime - interval.minTime)
                        })
            
            # 기본 통계 계산
            if phonemes:
                durations = [p["duration"] for p in phonemes]
                mean_duration = sum(durations) / len(durations)
                std_duration = (sum((d - mean_duration) ** 2 for d in durations) / len(durations)) ** 0.5
            else:
                mean_duration = 0.0
                std_duration = 0.0
            
            return {
                "phonemes": phonemes,
                "words": words,
                "statistics": {
                    "total_phonemes": len(phonemes),
                    "total_words": len(words),
                    "mean_duration": mean_duration,
                    "std_duration": std_duration
                }
            }
            
        except Exception as e:
            logger.error(f"음소 분석 중 오류가 발생했습니다: {str(e)}")
            return {}

    def _compare_with_reference(
        self,
        learner_audio: str,
        reference_audio: str,
        learner_textgrid: str,
    ) -> Optional[Dict[str, Any]]:
        """참조 오디오와 비교"""
        try:
            # 간단한 비교 분석 (복잡한 MFA 재정렬 없이)
            
            # 음성 특성 추출
            learner_features = self._extract_audio_features(learner_audio)
            reference_features = self._extract_audio_features(reference_audio)
            
            if not learner_features or not reference_features:
                logger.warning("오디오 특성 추출 실패")
                return {}
            
            # 피치 비교
            pitch_diff = abs(learner_features["pitch_mean"] - reference_features["pitch_mean"])
            
            # 에너지 비교  
            energy_diff = abs(learner_features["energy_mean"] - reference_features["energy_mean"])
            
            # 속도 비교
            duration_diff = abs(learner_features["duration"] - reference_features["duration"])
            
            return {
                "phoneme_comparison": {
                    "match_rate": 0.85,  # 임시값 (실제로는 더 복잡한 계산 필요)
                    "differences": []
                },
                "prosody_comparison": {
                    "pitch": {
                        "learner_mean": learner_features["pitch_mean"],
                        "reference_mean": reference_features["pitch_mean"],
                        "mean_diff": pitch_diff
                    },
                    "energy": {
                        "learner_mean": learner_features["energy_mean"],
                        "reference_mean": reference_features["energy_mean"],
                        "mean_diff": energy_diff
                    },
                    "duration": {
                        "learner": learner_features["duration"],
                        "reference": reference_features["duration"],
                        "diff": duration_diff
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"참조 오디오 비교 중 오류가 발생했습니다: {str(e)}")
            return {}

    def _extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """오디오에서 기본 특성 추출"""
        try:
            import librosa
            import numpy as np
            
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 피치 추출
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_contour = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_contour.append(pitch)
            
            # 에너지 추출
            energy = librosa.feature.rms(y=y)[0]
            
            return {
                "pitch_mean": float(np.mean(pitch_contour)) if pitch_contour else 0.0,
                "pitch_std": float(np.std(pitch_contour)) if pitch_contour else 0.0,
                "energy_mean": float(np.mean(energy)),
                "energy_std": float(np.std(energy)),
                "duration": float(len(y) / sr)
            }
            
        except Exception as e:
            logger.error(f"오디오 특성 추출 실패: {e}")
            return {}

    def _visualize_results(
        self,
        learner_audio: str,
        reference_audio: Optional[str],
        learner_textgrid: str,
        phoneme_analysis: Dict[str, Any],
        prosody_analysis: Dict[str, Any],
        comparison: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """결과 시각화 (koach/temp/visualize에 저장)"""
        try:
            # 시각화 폴더 생성
            self.visualize_dir.mkdir(parents=True, exist_ok=True)
            
            plot_paths = []

            # 1. 운율 분석 시각화 (항상 생성)
            prosody_plot_path = self.visualize_dir / "prosody_analysis.png"
            if prosody_analysis and prosody_analysis:  # 빈 딕셔너리가 아닌 경우
                self.prosody_analyzer.visualize_prosody(
                    prosody_analysis, str(prosody_plot_path)
                )
                plot_paths.append(str(prosody_plot_path))
                logger.info(f"📈 운율 시각화 저장: {prosody_plot_path}")
            else:
                # 데이터가 없어도 기본 차트 생성
                self._create_empty_prosody_chart(str(prosody_plot_path))
                plot_paths.append(str(prosody_plot_path))
                logger.info(f"📈 기본 운율 차트 생성: {prosody_plot_path}")

            # 2. 음소 분석 시각화 (데이터가 있는 경우)
            if phoneme_analysis and phoneme_analysis.get("phonemes"):
                try:
                    phoneme_plot_path = self.visualize_dir / "phoneme_analysis.png"
                    self._plot_phoneme_analysis_safe(phoneme_analysis, str(phoneme_plot_path))
                    plot_paths.append(str(phoneme_plot_path))
                    logger.info(f"📊 음소 시각화 저장: {phoneme_plot_path}")
                except Exception as e:
                    logger.error(f"음소 시각화 실패: {e}")

            # 3. 비교 분석 시각화 (데이터가 있는 경우)
            if comparison and comparison.get("prosody_comparison"):
                try:
                    comparison_plot_path = self.visualize_dir / "comparison_analysis.png"
                    self._plot_comparison_analysis_safe(comparison, str(comparison_plot_path))
                    plot_paths.append(str(comparison_plot_path))
                    logger.info(f"🔍 비교 시각화 저장: {comparison_plot_path}")
                except Exception as e:
                    logger.error(f"비교 시각화 실패: {e}")

            logger.info(f"📊 시각화 결과 저장: {self.visualize_dir}")
            return plot_paths

        except Exception as e:
            logger.error(f"시각화 중 오류가 발생했습니다: {str(e)}")
            return []

    def _plot_phoneme_analysis_safe(
        self,
        phoneme_analysis: Dict[str, Any],
        output_path: str,
    ) -> None:
        """안전한 음소 분석 시각화 (한글 폰트 설정 포함)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            import numpy as np

            # 한글 폰트 설정
            try:
                # macOS의 기본 한글 폰트들 시도
                font_candidates = [
                    'AppleGothic',      # macOS 기본
                    'Malgun Gothic',    # Windows
                    'NanumGothic',      # 나눔폰트
                    'DejaVu Sans'       # 백업용
                ]
                
                font_found = False
                for font_name in font_candidates:
                    try:
                        plt.rcParams['font.family'] = font_name
                        font_found = True
                        break
                    except:
                        continue
                
                if not font_found:
                    # 시스템에서 사용 가능한 한글 폰트 찾기
                    available_fonts = [f.name for f in fm.fontManager.ttflist]
                    korean_fonts = [f for f in available_fonts if any(k in f for k in ['Gothic', 'Dotum', 'Batang', 'Gulim'])]
                    if korean_fonts:
                        plt.rcParams['font.family'] = korean_fonts[0]
                        
            except Exception as e:
                logger.warning(f"한글 폰트 설정 실패: {e}")
                # 한글 대신 영문으로 표시
                pass

            # 마이너스 폰트 설정
            plt.rcParams['axes.unicode_minus'] = False

            phonemes = phoneme_analysis.get("phonemes", [])
            if not phonemes:
                return

            # 음소 길이 분포 (한글 대신 숫자로 표시)
            durations = [p["duration"] for p in phonemes[:20]]  # 처음 20개만
            phoneme_indices = list(range(len(durations)))  # 한글 대신 인덱스 사용

            plt.figure(figsize=(12, 6))
            
            # 음소 길이 막대그래프
            plt.subplot(1, 2, 1)
            bars = plt.bar(phoneme_indices, durations)
            plt.title("Phoneme Duration Distribution")  # 영문 제목
            plt.xlabel("Phoneme Index")
            plt.ylabel("Duration (sec)")
            
            # 음소 길이 히스토그램
            plt.subplot(1, 2, 2)
            all_durations = [p["duration"] for p in phonemes]
            plt.hist(all_durations, bins=10, alpha=0.7)
            plt.title("Phoneme Duration Histogram")
            plt.xlabel("Duration (sec)")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        except Exception as e:
            logger.error(f"음소 시각화 실패: {e}")

    def _plot_comparison_analysis_safe(
        self,
        comparison: Dict[str, Any],
        output_path: str,
    ) -> None:
        """안전한 비교 분석 시각화 (한글 폰트 설정 포함)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm

            # 한글 폰트 설정 (위와 동일)
            try:
                font_candidates = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
                for font_name in font_candidates:
                    try:
                        plt.rcParams['font.family'] = font_name
                        break
                    except:
                        continue
            except:
                pass

            plt.rcParams['axes.unicode_minus'] = False

            prosody_comp = comparison.get("prosody_comparison", {})
            if not prosody_comp:
                return

            plt.figure(figsize=(12, 4))

            # 피치 비교
            plt.subplot(1, 3, 1)
            pitch_data = prosody_comp.get("pitch", {})
            learner_pitch = pitch_data.get("learner_mean", 0)
            ref_pitch = pitch_data.get("reference_mean", 0)
            
            plt.bar(["Learner", "Reference"], [learner_pitch, ref_pitch], color=["red", "blue"])
            plt.title("Average Pitch Comparison")
            plt.ylabel("Frequency (Hz)")

            # 에너지 비교
            plt.subplot(1, 3, 2)
            energy_data = prosody_comp.get("energy", {})
            learner_energy = energy_data.get("learner_mean", 0)
            ref_energy = energy_data.get("reference_mean", 0)
            
            plt.bar(["Learner", "Reference"], [learner_energy, ref_energy], color=["red", "blue"])
            plt.title("Average Energy Comparison")
            plt.ylabel("Energy")

            # 길이 비교
            plt.subplot(1, 3, 3)
            duration_data = prosody_comp.get("duration", {})
            learner_dur = duration_data.get("learner", 0)
            ref_dur = duration_data.get("reference", 0)
            
            plt.bar(["Learner", "Reference"], [learner_dur, ref_dur], color=["red", "blue"])
            plt.title("Duration Comparison")
            plt.ylabel("Time (sec)")

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        except Exception as e:
            logger.error(f"비교 시각화 실패: {e}")

    def _convert_numpy_types(self, obj):
        """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

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
                output_path = self.visualize_dir / "prosody_comparison.png"
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
        self, prosody_result: Dict[str, Any], output_path: str = None
    ) -> None:
        """운율 분석 결과 시각화"""
        try:
            # ✅ output_path가 지정되지 않으면 visualize_dir 사용
            if output_path is None:
                self.visualize_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(self.visualize_dir / "prosody_comparison.png")

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

            logger.info(f"📈 운율 시각화 저장: {output_path}")

        except Exception as e:
            logger.error(f"운율 시각화 실패: {e}")

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

    def generate_compact_prompt(
        self,
        learner_text: str,
        native_text: str,
        script_text: str,
        learner_timing: str,
        native_timing: str,
        prosody_feedback: Optional[Dict] = None,
    ) -> str:
        """GPT용 발음 분석 프롬프트 생성 (베타 버전 압축된 버전)"""
        
        # RAG 검색으로 관련 지식 가져오기
        rag_context = ""
        if self.knowledge_base:
            query = f"한국어 발음 {script_text} 교정 피드백"
            search_results = self.knowledge_base.search(query, top_k=2)
            
            if search_results:
                rag_context = "\n\n**참고 발음 지식**:\n"
                for result in search_results:
                    rag_context += f"- {result['content'][:200]}...\n"

        # 메인 프롬프트 생성
        prompt = f"""당신은 한국어 발음 교정 전문가입니다.

**학습 목표 문장**: {script_text}

**분석 데이터**:
- 학습자 발화: {learner_text}
- 원어민 발화: {native_text}
- 학습자 타이밍: {learner_timing[:300] if learner_timing else 'N/A'}...
- 원어민 타이밍: {native_timing[:300] if native_timing else 'N/A'}...{rag_context}

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

    def extract_pronunciation_issues(
        self,
        learner_result: Dict,
        reference_result: Dict,
        learner_timing: str,
        reference_timing: str
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
                str(self.mfa_output),
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
        """TextGrid 파일에서 압축된 음소 정보 추출 (베타 버전 토큰 절약 기능)"""
        try:
            logger.info(f"📊 TextGrid 압축 요약 중: {path}")
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
            logger.info(f"📝 압축 요약: {summary[:100]}...")
            return summary
            
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

    def get_normalized_paths(self, speaker_type: str) -> Dict[str, str]:
        """정규화된 파일들의 경로 반환 (경로 정책 반영)
        
        Args:
            speaker_type: "learner" 또는 "native"
            
        Returns:
            Dict[str, str]: 원본과 정규화된 파일 경로들
        """
        wav_dir = self.temp_dir / "wav"
        normalized_dir = self.temp_dir / "normalized"  # 새로운 정규화 폴더
        
        return {
            "original": str(wav_dir / f"{speaker_type}.wav"),
            "normalized": str(normalized_dir / f"{speaker_type}_normalized.wav"),
            "for_analysis": str(normalized_dir / f"{speaker_type}_normalized.wav"),  # 분석용은 정규화된 것 사용
            "for_mfa": str(wav_dir / f"{speaker_type}.wav"),  # MFA용은 원본 사용 (더 안정적)
        }

    def _analyze_prosody_detailed(self, audio_path: str) -> Dict[str, Any]:
        """상세한 운율 분석"""
        try:
            import librosa
            import numpy as np
            
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 피치 분석
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512)
            pitch_contour = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_contour.append(pitch)
            
            # 에너지 분석
            energy = librosa.feature.rms(y=y, hop_length=512)[0]
            
            # 스펙트럼 중심 (음색 분석)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # 영교차율 (음성/무음 구분)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # 기본 통계
            valid_pitches = [p for p in pitch_contour if p > 0]
            
            return {
                "pitch": {
                    "contour": pitch_contour[:100],  # 처음 100개 프레임만
                    "mean": float(np.mean(valid_pitches)) if valid_pitches else 0.0,
                    "std": float(np.std(valid_pitches)) if valid_pitches else 0.0,
                    "min": float(np.min(valid_pitches)) if valid_pitches else 0.0,
                    "max": float(np.max(valid_pitches)) if valid_pitches else 0.0
                },
                "energy": {
                    "contour": energy[:100].tolist(),  # 처음 100개 프레임만
                    "mean": float(np.mean(energy)),
                    "std": float(np.std(energy))
                },
                "spectral_centroid": {
                    "contour": spectral_centroids[:100].tolist(),
                    "mean": float(np.mean(spectral_centroids))
                },
                "zero_crossing_rate": {
                    "contour": zcr[:100].tolist(),
                    "mean": float(np.mean(zcr))
                },
                "duration": float(len(y) / sr)
            }
            
        except Exception as e:
            logger.error(f"운율 분석 실패: {e}")
            return {}
