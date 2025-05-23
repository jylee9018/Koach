from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class FeedbackGenerator:
    def __init__(self, model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model)

    def generate_feedback(self, analysis_result: Dict) -> Dict:
        """분석 결과를 바탕으로 피드백 생성"""
        system_prompt = SystemMessage(
            content="""
        당신은 한국어 발음 전문가입니다. 
        발음 분석 결과를 바탕으로 학습자에게 도움이 되는 피드백을 제공해주세요.
        """
        )

        human_prompt = HumanMessage(
            content=f"""
        다음 분석 결과를 바탕으로 피드백을 생성해주세요:
        
        단어: {analysis_result['word']}
        유사도: {analysis_result['similarity']}
        발음 오류: {analysis_result['phoneme_errors']}
        억양 특징: {analysis_result['pitch_issue']}
        
        다음 형식으로 피드백을 제공해주세요:
        1. 잘한 점
        2. 개선이 필요한 점
        3. 구체적인 연습 방법
        4. 예시 단어/문장
        """
        )

        response = self.llm.invoke([system_prompt, human_prompt])
        return {"feedback": response.content}
