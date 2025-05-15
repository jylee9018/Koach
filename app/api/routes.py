from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.graph import feedback_graph

router = APIRouter()

class FeedbackRequest(BaseModel):
    audio_features: dict

@router.post("/feedback")
async def generate_feedback(request: FeedbackRequest):
    try:
        result = feedback_graph.invoke({"features": request.audio_features})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))