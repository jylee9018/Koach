from typing import Dict, Any
from langgraph.graph import StateGraph
from .nodes import KoachNodes
from .states import KoachState
import logging

# 로깅 추가
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("koach_graph")


def create_koach_graph():
    """Koach 워크플로우 그래프 생성"""
    nodes = KoachNodes()
    workflow = StateGraph(KoachState)

    # 노드 추가 (이름을 더 구체적으로 변경)
    workflow.add_node("audio_feature_extraction", nodes.extract_features)
    workflow.add_node("pronunciation_analysis", nodes.analyze_features)
    workflow.add_node("feedback_classification", nodes.classify_feedback_type)
    workflow.add_node("gpt_feedback_generation", nodes.generate_feedback)
    workflow.add_node("gpt_retry_handler", nodes.retry_gpt)
    workflow.add_node("fallback_feedback_generation", nodes.generate_fallback)
    workflow.add_node("feedback_persistence", nodes.save_feedback)
    workflow.add_node("state_logging", nodes.log_state)
    workflow.add_node("notification_creation", nodes.user_notification)

    # 기본 플로우
    workflow.add_edge("audio_feature_extraction", "pronunciation_analysis")
    workflow.add_edge("pronunciation_analysis", "feedback_classification")
    workflow.add_edge("feedback_classification", "gpt_feedback_generation")
    workflow.add_edge("feedback_persistence", "state_logging")
    workflow.add_edge("state_logging", "notification_creation")

    # 에러 처리 플로우
    workflow.add_conditional_edges(
        "gpt_feedback_generation",
        lambda x: (
            "gpt_retry_handler"
            if x.get("error") and not x.get("retry_attempted")
            else (
                "fallback_feedback_generation"
                if x.get("error")
                else "feedback_persistence"
            )
        ),
        {
            "gpt_retry_handler": "gpt_retry_handler",
            "fallback_feedback_generation": "fallback_feedback_generation",
            "feedback_persistence": "feedback_persistence",
        },
    )

    workflow.add_edge("gpt_retry_handler", "feedback_persistence")
    workflow.add_edge("fallback_feedback_generation", "feedback_persistence")

    workflow.set_entry_point("audio_feature_extraction")
    workflow.set_finish_point("notification_creation")

    # 노드 실행 전후 로깅 추가
    async def ainvoke(self, state):
        current_state = state.copy()
        for node_key, node_func in self.nodes.items():
            try:
                logger.debug(f"실행 노드: {node_key}")
                node_result = await node_func(current_state)
                current_state.update(node_result)
                logger.debug(f"노드 {node_key} 결과: {node_result.keys()}")
            except Exception as e:
                logger.error(f"노드 {node_key} 실행 오류: {str(e)}")
                current_state["error"] = str(e)
                break
        return current_state

    return workflow.compile()


# API용 그래프 인스턴스
koach_graph = create_koach_graph()
