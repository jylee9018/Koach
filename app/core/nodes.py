from .states import FeedbackState
from typing import Literal
import random

class FeedbackNodes:
    @staticmethod
    def extract_features(state: FeedbackState) -> FeedbackState:
        print("🎧 Extracting features...")
        return {"features": {"pitch": 0.75, "accuracy": 0.82}}

    @staticmethod
    def analyze_features(state: FeedbackState) -> FeedbackState:
        print("🧠 Analyzing features...")
        return {"analysis": {"accuracy": 0.82, "has_error": False}}

    @staticmethod
    def classify_feedback(state: FeedbackState) -> FeedbackState:
        print("📂 Classifying feedback type...")
        feedback_type: Literal["praise", "suggest", "fix"] = random.choice(
            ["praise", "suggest", "fix"]
        )
        print(f"➡️ Classified as: {feedback_type}")
        return {"feedback_type": feedback_type}

    @staticmethod
    def gpt_feedback(state: FeedbackState) -> FeedbackState:
        print("🤖 Calling GPT...")
        if random.random() < 0.3:  # 30% chance of failure
            raise Exception("GPT failed")
        return {"gpt_result": f"Generated {state['feedback_type']} feedback."}

    @staticmethod
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

    @staticmethod
    def fallback_feedback(state: FeedbackState) -> FeedbackState:
        print("🛟 Generating fallback feedback...")
        return {"fallback_result": "Basic fallback feedback due to GPT failure."}

    @staticmethod
    def save_feedback(state: FeedbackState) -> FeedbackState:
        print("💾 Saving feedback...")
        return state

    @staticmethod
    def log_state(state: FeedbackState) -> FeedbackState:
        print("📊 Logging state...")
        return state

    @staticmethod
    def user_notification(state: FeedbackState) -> FeedbackState:
        print("📣 Composing user notification...")
        return {"user_notification": "✅ Feedback is ready!"}

    @staticmethod
    def log_error(state: FeedbackState) -> FeedbackState:
        print("📝 Logging error...")
        return {"error": "An error occurred during feedback generation."}