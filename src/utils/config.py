"""
Configuration management for reRankQ.
Loads YAML configs and provides typed access to settings.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    respect_headers: bool = True


class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class DenseConfig(BaseModel):
    top_k: int = 20


class HybridConfig(BaseModel):
    method: str = "rrf"
    k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5


class RerankingConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


class RetrievalConfig(BaseModel):
    bm25: BM25Config = BM25Config()
    dense: DenseConfig = DenseConfig()
    hybrid: HybridConfig = HybridConfig()
    reranking: RerankingConfig = RerankingConfig()


class GenerationConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 1024
    temperature: float = 0.1


class EvaluationConfig(BaseModel):
    eval_set_size: int = 50
    metrics: list[str] = ["recall@k", "mrr", "ndcg@k"]
    k_values: list[int] = [1, 3, 5, 10]


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_latency: bool = True


class CorpusConfig(BaseModel):
    name: str = "stripe_api_docs"
    source: str = "https://docs.stripe.com"
    base_url: str = "https://docs.stripe.com"
    max_pages: int = 500


class AppConfig(BaseModel):
    corpus: CorpusConfig = CorpusConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    api: APIConfig = APIConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        # Look for config in standard locations
        possible_paths = [
            Path("configs/default.yaml"),
            Path("config.yaml"),
            Path(os.environ.get("RERANKQ_CONFIG", "configs/default.yaml")),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        return AppConfig(**raw_config)
    
    return AppConfig()


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config(config_path: Optional[str] = None) -> AppConfig:
    """Reset and reload configuration."""
    global _config
    _config = load_config(config_path)
    return _config
