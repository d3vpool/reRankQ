"""
FastAPI service for reRankQ.
Provides /query endpoint for RAG queries.
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils.config import get_config
from src.utils.logging import setup_logging, LatencyTracker


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True
    return_chunks: bool = False


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    url: str
    title: str
    section: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    latency_ms: float
    chunks_used: int
    method: str


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict


# Global state
app_state = {
    "retriever": None,
    "generator": None,
    "reranker": None,
    "latency_tracker": LatencyTracker(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup/shutdown."""
    logger = setup_logging()
    
    logger.info("Loading retrieval components...")
    
    try:
        # Try to load hybrid retriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        retriever.load_indexes()
        app_state["retriever"] = retriever
        logger.info("Loaded hybrid retriever")
    except Exception as e:
        logger.warning(f"Could not load hybrid retriever: {e}")
        
        try:
            # Fallback to BM25
            from src.retrieval.bm25_retriever import BM25Retriever
            retriever = BM25Retriever()
            retriever.load_index()
            app_state["retriever"] = retriever
            logger.info("Loaded BM25 retriever (fallback)")
        except Exception as e2:
            logger.error(f"Could not load any retriever: {e2}")
    
    try:
        # Load reranker
        from src.reranking.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        app_state["reranker"] = reranker
        logger.info("Loaded reranker")
    except Exception as e:
        logger.warning(f"Could not load reranker: {e}")
    
    try:
        # Load generator
        from src.generation.rag import RAGGenerator
        generator = RAGGenerator()
        app_state["generator"] = generator
        logger.info("Loaded RAG generator")
    except Exception as e:
        logger.warning(f"Could not load RAG generator: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="reRankQ",
    description="Hybrid search and RAG system with evaluation and ablations",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        components={
            "retriever": type(app_state["retriever"]).__name__ if app_state["retriever"] else None,
            "reranker": type(app_state["reranker"]).__name__ if app_state["reranker"] else None,
            "generator": type(app_state["generator"]).__name__ if app_state["generator"] else None,
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: QueryRequest with query and parameters
    
    Returns:
        QueryResponse with answer, citations, and metadata
    """
    if not app_state["retriever"]:
        raise HTTPException(status_code=503, detail="Retriever not loaded")
    
    if not app_state["generator"]:
        raise HTTPException(status_code=503, detail="Generator not loaded")
    
    latency_tracker = app_state["latency_tracker"]
    
    try:
        # Retrieve
        start_time = latency_tracker.start("retrieve")
        retrieved = app_state["retriever"].retrieve(
            request.query,
            top_k=request.top_k * 2 if request.use_reranking else request.top_k,
        )
        retrieve_time = latency_tracker.end("retrieve", start_time)
        
        # Rerank (optional)
        rerank_time = 0.0
        if request.use_reranking and app_state["reranker"]:
            start_time = latency_tracker.start("rerank")
            retrieved = app_state["reranker"].rerank(
                request.query,
                retrieved,
                top_k=request.top_k,
            )
            rerank_time = latency_tracker.end("rerank", start_time)
        
        # Add citations
        for chunk in retrieved:
            chunk["citation"] = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "url": chunk.get("metadata", {}).get("url", ""),
                "title": chunk.get("metadata", {}).get("title", ""),
                "section": chunk.get("metadata", {}).get("section", ""),
            }
        
        # Generate
        start_time = latency_tracker.start("generate")
        response = app_state["generator"].generate(
            request.query,
            retrieved[:request.top_k],
        )
        generate_time = latency_tracker.end("generate", start_time)
        
        # Calculate total latency
        total_latency = (retrieve_time + rerank_time + generate_time) * 1000  # Convert to ms
        
        # Build citations
        citations = [
            Citation(**chunk["citation"])
            for chunk in retrieved[:request.top_k]
            if chunk.get("citation")
        ]
        
        return QueryResponse(
            query=request.query,
            answer=response.answer,
            citations=citations,
            latency_ms=total_latency,
            chunks_used=len(retrieved[:request.top_k]),
            method="hybrid_rerank" if request.use_reranking else "hybrid",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latency")
async def get_latency_stats():
    """Get latency statistics for all pipeline stages."""
    latency_tracker = app_state["latency_tracker"]
    return latency_tracker.get_all_stats()


@app.get("/metrics")
async def get_metrics():
    """Get system metrics and stats."""
    config = get_config()
    
    # Check if we have results
    results_dir = Path("results")
    metrics = {
        "config": {
            "corpus": config.corpus.name,
            "embedding_model": config.embedding.model_name,
            "reranking_model": config.retrieval.reranking.model_name,
        },
        "ablation_results": None,
    }
    
    # Load ablation results if available
    ablation_file = results_dir / "ablation_results.json"
    if ablation_file.exists():
        with open(ablation_file) as f:
            metrics["ablation_results"] = json.load(f)
    
    return metrics


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
    )
