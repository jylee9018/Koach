from typing import TypedDict, Literal

class FeedbackState(TypedDict, total=False):
    features: dict
    analysis: dict
    feedback_type: Literal["praise", "suggest", "fix"]
    gpt_result: str
    fallback_result: str
    error: str
    user_notification: str
    retry_attempted: bool