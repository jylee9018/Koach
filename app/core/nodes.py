from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ..services.audio_processor import AudioProcessor
from ..services.pronunciation_analyzer import PronunciationAnalyzer
from .prompts import FEEDBACK_PROMPT, COMPARISON_FEEDBACK_PROMPT


class KoachNodes:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.analyzer = PronunciationAnalyzer()
        self.llm = ChatOpenAI(model="gpt-4")

    async def extract_features(self, state: Dict) -> Dict:
        """음성 특징 추출"""
        try:
            # 스크립트가 있는 경우 정렬 기반 처리
            if state.get("script"):
                features = await self.audio_processor.process_audio_with_alignment(
                    state["audio_path"], state["script"]
                )
            else:
                # 스크립트가 없는 경우 기본 처리
                features = await self.audio_processor.process_audio(state["audio_path"])

            result = {"features": features}

            # 원어민 오디오가 있는 경우 함께 처리
            if state.get("native_audio_path"):
                # 스크립트가 있는 경우 정렬 기반 처리
                if state.get("script"):
                    native_features = (
                        await self.audio_processor.process_audio_with_alignment(
                            state["native_audio_path"], state["script"]
                        )
                    )
                else:
                    # 스크립트가 없는 경우 기본 처리
                    native_features = await self.audio_processor.process_audio(
                        state["native_audio_path"]
                    )

                result["native_features"] = native_features

                # 음성 비교 분석 수행
                comparison = await self.audio_processor.compare_audios(
                    features, native_features
                )
                result["comparison_result"] = comparison

            return result
        except Exception as e:
            return {"error": str(e)}

    async def analyze_features(self, state: Dict) -> Dict:
        """발음 분석"""
        try:
            # 원어민 오디오 비교 결과가 있는 경우, 비교 분석 수행
            if state.get("comparison_result") and state.get("script"):
                analysis = await self.analyzer.analyze_comparison(
                    state["comparison_result"], state["script"]
                )

                # 정렬 기반 분석 추가
                alignment_analysis = await self.analyzer.analyze_with_alignment(
                    state["audio_path"], state["script"]
                )

                # 결과 통합
                analysis["alignment_result"] = alignment_analysis.get("alignment")
                analysis["phoneme_errors"] = alignment_analysis.get(
                    "phoneme_errors", []
                )

                return {
                    "analysis_result": analysis,
                    "similarity_score": analysis.get("similarity", 0.0),
                    "pronunciation_errors": analysis.get("pronunciation_errors", []),
                    "phoneme_errors": alignment_analysis.get("phoneme_errors", []),
                }
            elif state.get("script"):
                # 스크립트가 있지만 원어민 오디오가 없는 경우, 정렬 기반 분석만 수행
                alignment_analysis = await self.analyzer.analyze_with_alignment(
                    state["audio_path"], state["script"]
                )

                return {
                    "analysis_result": alignment_analysis,
                    "similarity_score": alignment_analysis.get("similarity", 0.0),
                    "pronunciation_errors": [],
                    "phoneme_errors": alignment_analysis.get("phoneme_errors", []),
                }
            else:
                # 기존 분석 로직 (스크립트가 없는 경우)
                analysis = await self.analyzer.analyze(state["features"])

                return {
                    "analysis_result": analysis,
                    "similarity_score": analysis.get("similarity", 0.0),
                    "pronunciation_errors": analysis.get("errors", []),
                }
        except Exception as e:
            return {"error": str(e)}

    async def classify_feedback_type(self, state: Dict) -> Dict:
        """피드백 유형 분류"""
        try:
            # similarity_score가 None인 경우 기본값 0.0으로 설정
            similarity_score = state.get("similarity_score", 0.0)

            # None 체크 추가
            if similarity_score is None:
                similarity_score = 0.0

            if similarity_score > 0.8:
                return {"feedback_type": "praise"}
            elif similarity_score > 0.6:
                return {"feedback_type": "suggest"}
            return {"feedback_type": "fix"}
        except Exception as e:
            return {"error": f"피드백 분류 중 오류 발생: {str(e)}"}

    async def generate_feedback(self, state: Dict) -> Dict:
        """피드백 생성"""
        try:
            feedback_type = state.get("feedback_type", "suggest")
            analysis_result = state.get("analysis_result", {})
            script = state.get("script", "")

            # 비교 결과가 있는 경우 (원어민 오디오 있음)
            if "comparison_result" in state:
                # 비교 피드백 프롬프트 사용
                prompt = COMPARISON_FEEDBACK_PROMPT.format(
                    script=script,
                    similarity=state.get("similarity_score", 0.0),
                    pronunciation_errors=state.get("pronunciation_errors", []),
                    analysis_details=analysis_result.get("analysis_details", {}),
                )
            else:
                # 정렬 기반 피드백 프롬프트 사용
                phoneme_errors = state.get("phoneme_errors", [])
                phoneme_error_text = "\n".join(
                    [
                        f"- {error.get('phoneme', '')}: {error.get('error_type', '')} "
                        f"({error.get('start_time', 0):.2f}s ~ {error.get('end_time', 0):.2f}s)"
                        for error in phoneme_errors
                    ]
                )

                prompt = f"""
                다음은 학습자의 한국어 발음을 분석한 결과입니다:

                스크립트: {script}
                전체 유사도: {state.get("similarity_score", 0.0):.2f}
                
                음소 단위 오류:
                {phoneme_error_text}
                
                이 분석 결과를 바탕으로 학습자에게 도움이 되는 피드백을 제공해주세요.
                다음 형식으로 피드백을 생성해주세요:

                1. 잘한 점
                2. 개선이 필요한 발음 요소
                3. 구체적인 연습 방법 및 팁
                4. 추천 연습 문장

                답변은 한국어 학습자가 이해하기 쉽게 작성해주세요.
                """

            # 시스템 메시지
            system_message = SystemMessage(
                content="당신은 한국어 발음 전문가입니다. 발음 분석 결과를 바탕으로 학습자에게 도움이 되는 피드백을 제공해주세요."
            )

            # 사용자 메시지
            human_message = HumanMessage(content=prompt)

            # GPT 호출
            response = self.llm.invoke([system_message, human_message])

            return {"gpt_result": response.content}
        except Exception as e:
            return {"error": f"피드백 생성 중 오류 발생: {str(e)}"}

    async def retry_gpt(self, state: Dict) -> Dict:
        """GPT 피드백 생성 재시도"""
        try:
            prompt = self._create_feedback_prompt(state)

            messages = [
                SystemMessage(
                    content="당신은 한국어 발음 전문가입니다. 한국어 학습자의 발음을 분석하고 개선을 위한 피드백을 제공합니다."
                ),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            return {"gpt_result": response.content, "retry_attempted": True}
        except Exception as e:
            return {"error": str(e), "retry_attempted": True}

    async def generate_fallback(self, state: Dict) -> Dict:
        """GPT 피드백 실패 시 대체 피드백 생성"""
        try:
            errors = state.get("pronunciation_errors", [])
            score = state.get("similarity_score", 0)

            # 간단한 대체 피드백
            if score > 0.8:
                feedback = "전반적으로 발음이 좋습니다. 계속 연습하세요."
            elif score > 0.6:
                feedback = (
                    f"발음이 괜찮지만 개선이 필요합니다. 주의할 점: {', '.join(errors)}"
                )
            else:
                feedback = f"발음 개선이 필요합니다. 중점적으로 연습할 부분: {', '.join(errors)}"

            return {"fallback_result": feedback}
        except Exception as e:
            return {
                "fallback_result": "발음 피드백을 생성할 수 없습니다. 나중에 다시 시도해주세요."
            }

    async def save_feedback(self, state: Dict) -> Dict:
        """피드백 저장 (간단한 패스스루)"""
        return state

    async def log_state(self, state: Dict) -> Dict:
        """상태 로깅 (간단한 패스스루)"""
        return state

    async def user_notification(self, state: Dict) -> Dict:
        """사용자 알림 생성 (간단한 패스스루)"""
        return {"user_notification": "분석이 완료되었습니다."}

    def _create_feedback_prompt(self, state: Dict) -> str:
        script = state.get("script", "")

        # 비교 분석이 있는 경우
        if state.get("comparison_result"):
            return COMPARISON_FEEDBACK_PROMPT.format(
                script=script,
                similarity=state.get("similarity_score", 0),
                pronunciation_errors=state.get("pronunciation_errors", []),
                analysis_details=state.get("analysis_details", {}),
            )
        else:
            # 기존 로직
            return FEEDBACK_PROMPT.format(
                analysis=f"""
                유사도: {state.get('similarity_score', 0)}
                발음 오류: {state.get('pronunciation_errors', [])}
                피치 분석: {state.get('pitch_analysis', {})}
                """
            )

    def _create_user_notification(self, state: Dict) -> str:
        if state.get("error"):
            return "죄송합니다. 분석 중 오류가 발생했습니다."

        feedback = state.get("gpt_result") or state.get("fallback_result")
        return f"분석이 완료되었습니다.\n{feedback}"
