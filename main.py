import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
from typing import Dict

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered Korean pronunciation feedback system",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 경로
    redoc_url="/redoc",  # ReDoc 경로
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_V1_STR)

def save_results(result: Dict, output_dir: str) -> str:
    """결과를 JSON 파일로 저장"""
    try:
        import json
        import numpy as np
        from pathlib import Path
        
        def convert_numpy_types(obj):
            """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # numpy 타입 변환
        result = convert_numpy_types(result)
        
        # 파일 저장
        output_path = Path(output_dir) / "analysis_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")
        return ""

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
