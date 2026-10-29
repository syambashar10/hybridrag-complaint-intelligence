from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .analyzer import ComplaintAnalyzer
from .config import FAISS_INDEX_PATH, PROCESSED_DATA_PATH
from .hybrid_search import HybridSearchEngine

app = FastAPI(
    title="HybridRAG Complaint Intelligence API",
    version="1.0.0",
    description="Hybrid lexical + semantic retrieval over CFPB consumer complaints.",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)


class AnalyzeRequest(BaseModel):
    complaint: str = Field(min_length=3, max_length=10_000)
    evidence_k: int = Field(default=8, ge=3, le=20)


@lru_cache(maxsize=1)
def get_engine() -> HybridSearchEngine:
    return HybridSearchEngine().load()


@lru_cache(maxsize=1)
def get_analyzer() -> ComplaintAnalyzer:
    return ComplaintAnalyzer(get_engine())


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "processed_data_ready": PROCESSED_DATA_PATH.exists(),
        "semantic_index_ready": FAISS_INDEX_PATH.exists(),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict:
    try:
        results = get_engine().search(request.query, top_k=request.top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"query": request.query, "count": len(results), "results": results}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        result = get_analyzer().analyze(request.complaint, request.evidence_k)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"complaint": request.complaint, **result}
