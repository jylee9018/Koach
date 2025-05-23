#!/usr/bin/env python3
"""
Koach - 한국어 발음 교정 도우미 (구조화된 메인 버전)

베타 버전의 사용자 친화적 기능들을 구조화된 버전에 통합한 메인 실행 파일입니다.

사용법:
    python main.py [learner_audio] [native_audio] [script_text]
    python main.py --help
    
예시:
    python main.py input/learner.m4a input/native.m4a "안녕하세요"
    python main.py --file input/learner.wav --reference input/native.wav --text "한국어"
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

# 프로젝트 모듈 import (경로 수정)
from core.koach import Koach
from config.settings import CURRENT_CONFIG, NEW_PATHS as PATHS, validate_environment, update_config

# =============================================================================
# 로깅 설정
# =============================================================================

logging.basicConfig(
    level=getattr(logging, CURRENT_CONFIG["logging"]["level"]),
    format=CURRENT_CONFIG["logging"]["format"],
    datefmt=CURRENT_CONFIG["logging"]["date_format"],
)
logger = logging.getLogger("KoachMain")

# =============================================================================
# CLI 인터페이스 (Command Line Interface)
# =============================================================================

def setup_argparse() -> argparse.ArgumentParser:
    """베타 버전 스타일의 향상된 명령행 인자 파서"""
    parser = argparse.ArgumentParser(
        description="🎤 Koach - 한국어 발음 교정 도우미",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py input/learner.m4a input/native.m4a "안녕하세요"
  python main.py --file input/my_voice.wav --reference input/teacher.wav
  python main.py --file input/speech.m4a --text "한국어 발음 연습"
  
환경 변수:
  OPENAI_API_KEY    OpenAI API 키 (필수)
  
더 자세한 정보는 README.md를 참조하세요.
        """
    )

    # 위치 인수 (베타 스타일)
    parser.add_argument(
        "learner_audio",
        nargs="?",
        help="학습자 음성 파일 경로"
    )
    
    parser.add_argument(
        "native_audio", 
        nargs="?",
        help="원어민 음성 파일 경로"
    )
    
    parser.add_argument(
        "script_text",
        nargs="?", 
        help="목표 발음 텍스트"
    )

    # 옵션 인수 (기존 스타일 호환)
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="학습자 음성 파일 경로 (위치 인수 대신 사용 가능)"
    )
    
    parser.add_argument(
        "--reference", "-r",
        type=str,
        help="원어민 참조 음성 파일 경로"
    )

    parser.add_argument(
        "--text", "-t",
        type=str,
        help="목표 발음 텍스트"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="결과 저장 디렉토리"
    )

    parser.add_argument(
        "--model-size", "-m",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper 모델 크기 (기본값: base)"
    )

    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="RAG 지식베이스 사용 안함"
    )

    parser.add_argument(
        "--no-visualization", "-nv",
        action="store_true",
        help="시각화 결과 생성 안함"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="최소한의 출력만 표시"
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Koach 1.0.0"
    )

    return parser

def parse_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    """명령행 인수를 파싱하고 설정을 생성합니다."""
    config = {}
    
    # 입력 파일 결정 (위치 인수 우선)
    learner_audio = args.learner_audio or args.file
    native_audio = args.native_audio or args.reference
    script_text = args.script_text or args.text
    
    if not learner_audio:
        # 기본 파일 사용
        learner_audio = CURRENT_CONFIG["learner_audio"]
        logger.info("📁 기본 학습자 파일 사용")
    
    # 설정 업데이트
    if args.model_size:
        config["whisper_model"] = args.model_size
        
    if args.no_rag:
        config["use_rag"] = False
        
    if args.output_dir:
        config["output_dir"] = args.output_dir
        
    # 로깅 레벨 조정
    if args.quiet:
        config["logging"] = {"level": "WARNING"}
        
    return {
        "learner_audio": learner_audio,
        "native_audio": native_audio,
        "script_text": script_text,
        "visualize": not args.no_visualization,
        "config": config
    }

# =============================================================================
# 결과 출력 함수 (Output Functions)
# =============================================================================

def print_startup_banner():
    """시작 배너 출력"""
    print("🚀 Koach - 한국어 발음 교정 도우미")
    print("=" * 60)

def print_gpt_feedback(result: Dict[str, Any]) -> None:
    """GPT 피드백을 터미널에 예쁘게 출력 (베타 스타일)"""
    if not result.get("feedback"):
        return
    
    feedback = result["feedback"]
    
    print("\n" + "="*80)
    print("🤖 GPT 발음 교정 피드백")
    print("="*80)
    
    if isinstance(feedback, str):
        # 베타 스타일의 GPT 응답
        print(feedback)
    elif isinstance(feedback, dict):
        # 구조화된 피드백
        if "summary" in feedback:
            print(f"\n📋 요약: {feedback['summary']}")
        
        if "detailed_analysis" in feedback:
            print("\n📝 상세 분석:")
            print("-" * 40)
            print(feedback["detailed_analysis"])
        
        if "suggestions" in feedback:
            print("\n💡 제안사항:")
            print("-" * 40)
            for i, suggestion in enumerate(feedback["suggestions"], 1):
                print(f"   {i}. {suggestion}")
    
    print("\n" + "="*80)

def print_analysis_summary(result: Dict[str, Any]) -> None:
    """분석 결과 요약을 터미널에 출력"""
    print("\n" + "="*60)
    print("📊 발음 분석 결과 요약")
    print("="*60)
    
    # 전사 결과
    if "learner_text" in result:
        print(f"\n🎤 학습자 발화: \"{result['learner_text']}\"")
    
    if "native_text" in result:
        print(f"🎯 원어민 발화: \"{result['native_text']}\"")
    
    if "script_text" in result:
        print(f"📝 목표 텍스트: \"{result['script_text']}\"")
    
    # 처리 단계 상태
    if "steps" in result:
        print(f"\n🔧 처리 단계:")
        for step, status in result["steps"].items():
            status_icon = "✅" if status == "성공" else "❌" if status == "실패" else "⚠️"
            print(f"  {status_icon} {step}: {status}")
    
    # 오류 정보
    if "errors" in result and result["errors"]:
        print(f"\n⚠️ 발생한 오류:")
        for error in result["errors"]:
            print(f"  - {error}")

def save_results(result: Dict[str, Any], output_dir: str) -> str:
    """결과를 JSON 파일로 저장"""
    output_file = os.path.join(output_dir, "analysis_result.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return output_file

# =============================================================================
# 메인 실행 함수 (Main Execution)
# =============================================================================

def main():
    """메인 실행 함수"""
    try:
        # 시작 배너
        print_startup_banner()
        
        # 명령행 인수 파싱
        parser = setup_argparse()
        args = parser.parse_args()
        
        # 인수 파싱 및 설정 생성
        parsed = parse_arguments(args)
        
        # 사용자 설정 적용
        if parsed["config"]:
            update_config(parsed["config"])
        
        # 환경 변수 확인
        env_errors = validate_environment()
        if env_errors:
            for error in env_errors:
                print(error)
            print("\n💡 해결 방법:")
            print("  export OPENAI_API_KEY='your_openai_api_key'")
            return 1
        
        # 입력 파일 검증
        learner_audio = parsed["learner_audio"]
        if learner_audio and not os.path.exists(learner_audio):
            print(f"❌ 학습자 음성 파일을 찾을 수 없습니다: {learner_audio}")
            return 1
        
        native_audio = parsed["native_audio"]
        if native_audio and not os.path.exists(native_audio):
            print(f"❌ 원어민 음성 파일을 찾을 수 없습니다: {native_audio}")
            return 1
        
        # 분석 시작
        logger.info("🔧 Koach 시스템 초기화 중...")
        koach = Koach()
        
        logger.info("🎯 발음 분석 시작...")
        result = koach.analyze_pronunciation(
            learner_audio=learner_audio,
            native_audio=native_audio,
            script=parsed["script_text"],
            visualize=parsed["visualize"]
        )
        
        if result and result.get("status") in ["완료", "success"]:
            # 결과 출력
            print_analysis_summary(result)
            print_gpt_feedback(result)
            
            # 결과 저장
            output_file = save_results(result, PATHS["output_dir"])
            print(f"\n💾 상세 결과가 {output_file}에 저장되었습니다.")
            
            print("\n✅ 분석 완료!")
            return 0
        else:
            print("\n❌ 분석 실패")
            if result and result.get("errors"):
                for error in result["errors"]:
                    print(f"  - {error}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"❌ 예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
