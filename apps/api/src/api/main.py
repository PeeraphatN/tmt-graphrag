from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

from src.pipeline import GraphRAGPipeline

# Global Pipeline Instance
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Pipeline on Startup"""
    global pipeline
    logger.info("🚀 Starting TMT GraphRAG API...")
    try:
        pipeline = GraphRAGPipeline()
        pipeline.warmup()
        logger.info("✅ Pipeline Ready!")
    except Exception as e:
        logger.error("❌ Startup Error: %s", e)
    yield
    logger.info("🛑 Shutting down API...")

# Create FastAPI App
app = FastAPI(
    title="TMT Drug RAG API",
    description="API for Thai Medicinal Terminology RAG System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS (Allow Frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# API Endpoints
@app.get("/")
async def root():
    return {"status": "ok", "message": "TMT Drug RAG API is running"}

@app.get("/health")
async def health_check():
    if pipeline:
        return {"status": "healthy", "model_ready": True}
    return {"status": "unhealthy", "model_ready": False}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")

    try:
        import asyncio
        from starlette.concurrency import run_in_threadpool
        from src.config import CHAT_TIMEOUT_SECONDS

        # Offload blocking task to threadpool to prevent blocking the Event Loop
        answer = await asyncio.wait_for(
            run_in_threadpool(pipeline.run, request.message),
            timeout=CHAT_TIMEOUT_SECONDS,
        )
        return ChatResponse(response=answer)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {CHAT_TIMEOUT_SECONDS}s — LLM or retrieval took too long",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Hot reload enabled for dev
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
