from typing import TypedDict, Literal, List, Optional, Dict


class AudioFeatures(TypedDict):
    pitch: Dict
    duration: Dict
    energy: Dict


class PronunciationError(TypedDict):
    phoneme: str
    error_type: Literal["누락", "왜곡", "치환", "추가", "연음"]
    position: int
    confidence: float


class FeedbackState(TypedDict, total=False):
    # 입력 상태
    audio_path: str
    features: Dict

    # 분석 상태
    analysis: Dict
    feedback_type: Literal["praise", "suggest", "fix"]

    # 결과 상태
    gpt_result: str
    feedback: str
    error: str


class KoachState(TypedDict, total=False):
    # 입력 상태
    audio_path: str
    native_audio_path: Optional[str]
    script: str
    features: AudioFeatures
    native_features: Optional[AudioFeatures]

    # 분석 상태
    pronunciation_errors: List[str]
    similarity_score: float
    pitch_analysis: dict
    comparison_result: Optional[Dict]

    # 피드백 상태
    feedback_type: Literal["praise", "suggest", "fix"]
    gpt_result: str
    fallback_result: str

    # 에러 처리
    error: Optional[str]
    retry_attempted: bool
    user_notification: str
