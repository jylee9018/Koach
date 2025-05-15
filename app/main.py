from fastapi import FastAPI
from .api.routes import router

app = FastAPI(title="Koach API")
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}