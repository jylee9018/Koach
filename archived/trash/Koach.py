from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Optional
import random


# -------- 1. Define the state --------
class FeedbackState(TypedDict, total=False):
    features: dict
    analysis: dict
    feedback_type: Literal["praise", "suggest", "fix"]
    gpt_result: str
    fallback_result: str
    error: str
    user_notification: str
    retry_attempted: bool


# -------- 2. Define each node --------
def extract_features(state: FeedbackState) -> FeedbackState:
    print("🎧 Extracting features...")
    return {"features": {"pitch": 0.75, "accuracy": 0.82}}


def analyze_features(state: FeedbackState) -> FeedbackState:
    print("🧠 Analyzing features...")
    return {"analysis": {"accuracy": 0.82, "has_error": False}}


def classify_feedback(state: FeedbackState) -> FeedbackState:
    print("📂 Classifying feedback type...")
    feedback_type: Literal["praise", "suggest", "fix"] = random.choice(
        ["praise", "suggest", "fix"]
    )
    print(f"➡️ Classified as: {feedback_type}")
    return {"feedback_type": feedback_type}


def gpt_feedback(state: FeedbackState) -> FeedbackState:
    print("🤖 Calling GPT...")
    if random.random() < 0.3:  # 30% chance of failure
        raise Exception("GPT failed")
    return {"gpt_result": f"Generated {state['feedback_type']} feedback."}


def retry_gpt(state: FeedbackState) -> FeedbackState:
    print("🔁 Retrying GPT...")
    if state.get("retry_attempted"):
        raise Exception("Retry already attempted.")
    if random.random() < 0.5:
        return {
            "gpt_result": f"Recovered {state['feedback_type']} feedback on retry.",
            "retry_attempted": True,
        }
    raise Exception("Retry GPT failed")


def fallback_feedback(state: FeedbackState) -> FeedbackState:
    print("🛟 Generating fallback feedback...")
    return {"fallback_result": "Basic fallback feedback due to GPT failure."}


def save_feedback(state: FeedbackState) -> FeedbackState:
    print("💾 Saving feedback...")
    return state


def log_state(state: FeedbackState) -> FeedbackState:
    print("📊 Logging state...")
    return state


def user_notification(state: FeedbackState) -> FeedbackState:
    print("📣 Composing user notification...")
    return {"user_notification": "✅ Feedback is ready!"}


def log_error(state: FeedbackState) -> FeedbackState:
    print("📝 Logging error...")
    return {"error": "An error occurred during feedback generation."}


# -------- 3. Build the graph --------
builder = StateGraph(FeedbackState)

# Add nodes
builder.add_node("ExtractFeatures", extract_features)
builder.add_node("AnalyzeFeatures", analyze_features)
builder.add_node("ClassifyFeedback", classify_feedback)
builder.add_node("GPT_Praise", gpt_feedback)
builder.add_node("GPT_Suggest", gpt_feedback)
builder.add_node("GPT_Fix", gpt_feedback)
builder.add_node("RetryGPT", retry_gpt)
builder.add_node("FallbackFeedback", fallback_feedback)
builder.add_node("SaveFeedback", save_feedback)
builder.add_node("LogState", log_state)
builder.add_node("UserNotification", user_notification)
builder.add_node("LogError", log_error)

# Entry point
builder.set_entry_point("ExtractFeatures")

# Linear path
builder.add_edge("ExtractFeatures", "AnalyzeFeatures")
builder.add_edge("AnalyzeFeatures", "ClassifyFeedback")

# Conditional branching to different GPT nodes
builder.add_conditional_edges(
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


# GPT feedback → conditional: success or fail
def gpt_result_condition(state: FeedbackState):
    return "success" if "gpt_result" in state else "fail"


for gpt_node in ["GPT_Praise", "GPT_Suggest", "GPT_Fix"]:
    builder.add_conditional_edges(
        gpt_node, gpt_result_condition, {"success": "SaveFeedback", "fail": "RetryGPT"}
    )


# RetryGPT → conditional: success or fail
def retry_condition(state: FeedbackState):
    return "success" if "gpt_result" in state else "fail"


builder.add_conditional_edges(
    "RetryGPT", retry_condition, {"success": "SaveFeedback", "fail": "LogError"}
)

# LogError → fallback → save
builder.add_edge("LogError", "FallbackFeedback")
builder.add_edge("FallbackFeedback", "SaveFeedback")

# Finalization
builder.add_edge("SaveFeedback", "LogState")
builder.add_edge("LogState", "UserNotification")
builder.add_edge("UserNotification", END)


# -------- 4. Compile the graph --------
graph = builder.compile()


# -------- 5. Run a test --------
if __name__ == "__main__":
    print("🚀 Running LangGraph test...\n")
    result = graph.invoke({})
    print("\n✅ Final result:")
    print(result)
