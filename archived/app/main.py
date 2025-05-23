from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import routes

app = FastAPI(title="Koach API")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 - 여기서 접두사를 확인
app.include_router(routes.router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "Welcome to Koach API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
