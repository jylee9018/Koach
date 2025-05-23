import os
import logging
import argparse
from pathlib import Path
from typing import Optional
import json
import subprocess
import tempfile

from core.koach import Koach
from config.settings import CURRENT_CONFIG

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, CURRENT_CONFIG["logging"]["level"]),
    format=CURRENT_CONFIG["logging"]["format"],
    datefmt=CURRENT_CONFIG["logging"]["date_format"],
)
logger = logging.getLogger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """명령행 인자 파서 설정"""
    parser = argparse.ArgumentParser(description="한국어 발음 교정 도우미")

    parser.add_argument(
        "input_file",
        type=str,
        help="분석할 오디오 파일 경로",
    )

    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        help="참조할 오디오 파일 경로 (선택사항)",
    )

    parser.add_argument(
        "--text",
        "-t",
        type=str,
        help="발음할 텍스트 (선택사항)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="결과를 저장할 디렉토리 경로 (선택사항)",
    )

    parser.add_argument(
        "--model-size",
        "-m",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper 모델 크기 (기본값: base)",
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="ko",
        help="언어 코드 (기본값: ko)",
    )

    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="시각화 결과 생성",
    )

    return parser


def convert_audio_to_wav(input_path: str) -> str:
    """오디오 파일을 WAV 형식으로 변환"""
    if input_path.lower().endswith(".wav"):
        return input_path

    # 임시 파일 생성
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"{Path(input_path).stem}.wav")

    try:
        # ffmpeg 명령어 구성
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-ar",
            "16000",  # 샘플 레이트
            "-ac",
            "1",  # 모노 채널
            "-y",  # 기존 파일 덮어쓰기
            output_path,
        ]

        # 변환 실행
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"오디오 변환 실패: {result.stderr}")

        logger.info(f"오디오 파일이 변환되었습니다: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"오디오 변환 중 오류 발생: {str(e)}")
        raise


def validate_inputs(args: argparse.Namespace) -> None:
    """입력값 검증"""
    # 입력 파일 검증
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {args.input_file}")

    # 참조 파일 검증
    if args.reference and not os.path.exists(args.reference):
        raise FileNotFoundError(f"참조 파일을 찾을 수 없습니다: {args.reference}")

    # 출력 디렉토리 생성
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)


def print_gpt_feedback(result: dict) -> None:
    """GPT 피드백을 터미널에 예쁘게 출력"""
    if not result.get("feedback"):
        return
    
    feedback = result["feedback"]
    
    print("\n" + "="*80)
    print("🤖 GPT 발음 교정 피드백 결과")
    print("="*80)
    
    # 기본 요약
    if "summary" in feedback:
        print(f"\n📋 요약: {feedback['summary']}")
    
    # 상세 분석 결과
    if "detailed_analysis" in feedback:
        print("\n📝 상세 분석:")
        print("-" * 40)
        analysis = feedback["detailed_analysis"]
        
        # 긴 텍스트를 적절히 포맷팅
        lines = analysis.split('\n')
        for line in lines:
            if line.strip():
                print(f"   {line}")
    
    # 간단한 제안사항 (이전 버전 호환)
    elif "suggestions" in feedback:
        print("\n💡 제안사항:")
        print("-" * 40)
        for i, suggestion in enumerate(feedback["suggestions"], 1):
            print(f"   {i}. {suggestion}")
    
    # 기술적 정보
    if "token_optimized" in feedback:
        print(f"\n⚡ 토큰 최적화: {'적용됨' if feedback['token_optimized'] else '미적용'}")
    
    if "model_used" in feedback:
        print(f"🔧 사용된 모델: {feedback['model_used']}")
    
    print("\n" + "="*80)


def print_analysis_summary(result: dict) -> None:
    """분석 결과 요약을 터미널에 출력"""
    print("\n" + "="*80)
    print("📊 발음 분석 결과 요약")
    print("="*80)
    
    # 전사 결과
    if "learner" in result and "transcription" in result["learner"]:
        learner_text = result["learner"]["transcription"].get("text", "")
        print(f"\n🎤 학습자 발화: \"{learner_text}\"")
    
    if "native" in result and "transcription" in result["native"]:
        native_text = result["native"]["transcription"].get("text", "")
        print(f"🎯 원어민 발화: \"{native_text}\"")
    
    # 발음 문제점
    if "comparison" in result and "issues" in result["comparison"]:
        issues = result["comparison"]["issues"]
        if issues:
            print(f"\n⚠️  발견된 문제점 ({len(issues)}개):")
            for i, issue in enumerate(issues[:3], 1):  # 최대 3개만 표시
                if issue["type"] == "text_mismatch":
                    print(f"   {i}. 텍스트 불일치")
                elif issue["type"] == "word_mismatch":
                    print(f"   {i}. 단어 발음 문제: '{issue['learner']}' → '{issue['reference']}'")
                else:
                    print(f"   {i}. {issue['type']}")
            
            if len(issues) > 3:
                print(f"   ... 및 {len(issues) - 3}개 추가 문제점")
    
    # 운율 분석 요약
    if "comparison" in result and "prosody" in result["comparison"]:
        prosody = result["comparison"]["prosody"]
        if "differences" in prosody:
            diff = prosody["differences"]
            print(f"\n🎵 운율 분석:")
            pitch_diff = diff.get("pitch", {}).get("mean", 0)
            energy_diff = diff.get("energy", {}).get("mean", 0)
            time_diff = diff.get("time", {}).get("total_duration", 0)
            
            print(f"   • 피치 차이: {pitch_diff:+.1f}Hz")
            print(f"   • 에너지 차이: {energy_diff:+.3f}")
            print(f"   • 길이 차이: {time_diff:+.1f}초")
    
    print("\n" + "="*80)


def main():
    """메인 함수"""
    try:
        # 명령행 인자 파싱
        parser = setup_argparse()
        args = parser.parse_args()

        # 입력값 검증
        validate_inputs(args)

        # 오디오 파일 변환
        input_wav = convert_audio_to_wav(args.input_file)
        reference_wav = convert_audio_to_wav(args.reference) if args.reference else None

        # Koach 인스턴스 생성
        koach = Koach(
            config={
                "whisper_model": "base",
                "language": "ko",
                "use_rag": True,
            }
        )

        # 발음 분석 실행
        result = koach.analyze_pronunciation(
            learner_audio=input_wav,
            native_audio=reference_wav,
            script=args.text,
            visualize=args.visualize,
        )

        # 결과 저장
        output_dir = Path(args.output_dir) if args.output_dir else Path("output")
        output_file = output_dir / f"{Path(args.input_file).stem}_result.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"분석 결과가 저장되었습니다: {output_file}")

        # 시각화 결과 경로 출력
        if args.visualize and "visualization_paths" in result:
            logger.info("시각화 결과:")
            for path in result["visualization_paths"]:
                logger.info(f"- {path}")

        # 피드백 출력
        print_gpt_feedback(result)
        print_analysis_summary(result)

    except Exception as e:
        logger.error(f"오류가 발생했습니다: {str(e)}")
        raise


if __name__ == "__main__":
    main()
