from langgraph.graph import StateGraph, END
from .states import FeedbackState
from .nodes import FeedbackNodes

class FeedbackGraph:
    def __init__(self):
        self.nodes = FeedbackNodes()
        self.builder = StateGraph(FeedbackState)
        self._build_graph()

    def _build_graph(self):
        # Add nodes
        self.builder.add_node("ExtractFeatures", self.nodes.extract_features)
        self.builder.add_node("AnalyzeFeatures", self.nodes.analyze_features)
        self.builder.add_node("ClassifyFeedback", self.nodes.classify_feedback)
        self.builder.add_node("GPT_Praise", self.nodes.gpt_feedback)
        self.builder.add_node("GPT_Suggest", self.nodes.gpt_feedback)
        self.builder.add_node("GPT_Fix", self.nodes.gpt_feedback)
        self.builder.add_node("RetryGPT", self.nodes.retry_gpt)
        self.builder.add_node("FallbackFeedback", self.nodes.fallback_feedback)
        self.builder.add_node("SaveFeedback", self.nodes.save_feedback)
        self.builder.add_node("LogState", self.nodes.log_state)
        self.builder.add_node("UserNotification", self.nodes.user_notification)
        self.builder.add_node("LogError", self.nodes.log_error)

        # Set entry point
        self.builder.set_entry_point("ExtractFeatures")

        # Add edges
        self._add_edges()
        self._add_conditional_edges()

    def _add_edges(self):
        # Linear path
        self.builder.add_edge("ExtractFeatures", "AnalyzeFeatures")
        self.builder.add_edge("AnalyzeFeatures", "ClassifyFeedback")

    def _add_conditional_edges(self):
        # Conditional branching logic
        self.builder.add_conditional_edges(
            "ClassifyFeedback",
            lambda state: {"praise": "GPT_Praise", "suggest": "GPT_Suggest", "fix": "GPT_Fix"}[
                state["feedback_type"]
            ],
            {
                "GPT_Praise": "GPT_Praise",
                "GPT_Suggest": "GPT_Suggest",
                "GPT_Fix": "GPT_Fix",
            },
        )

    def get_graph(self):
        return self.builder.compile()

# Graph instance for API use
feedback_graph = FeedbackGraph().get_graph()