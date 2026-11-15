"""
Configuration management for reRankQ.
Loads YAML configs and provides typed access to settings.

The models below mirror the structure of ``configs/default.yaml``. Every model
forbids unknown keys, so a misspelled or unsupported setting raises an error
instead of being silently discarded. All size limits are expressed in tokens,
matching the tokenizer used by the chunkers.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class FixedSizeChunkingConfig(StrictConfigModel):
    """Parameters for fixed-size chunking. Sizes are measured in tokens."""

    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)


class StructureAwareChunkingConfig(StrictConfigModel):
    """Parameters for structure-aware chunking. Sizes are measured in tokens."""

    max_chunk_size: int = Field(default=512, ge=1)
    min_chunk_size: int = Field(default=100, ge=0)
    respect_headers: bool = True

    @model_validator(mode="after")
    def _validate_size_range(self) -> "StructureAwareChunkingConfig":
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError(
                "chunking.structure_aware.min_chunk_size "
                f"({self.min_chunk_size}) must not exceed max_chunk_size "
                f"({self.max_chunk_size})"
            )
        return self


class ChunkingConfig(StrictConfigModel):
    fixed_size: FixedSizeChunkingConfig = Field(
        default_factory=FixedSizeChunkingConfig
    )
    structure_aware: StructureAwareChunkingConfig = Field(
        default_factory=StructureAwareChunkingConfig
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_flat_keys(cls, data: Any) -> Any:
        """
        Keep pre-nesting configurations working.

        Older configs placed ``chunk_size``/``chunk_overlap``/``respect_headers``
        directly under ``chunking``. They are migrated into the nested sections;
        any other unrecognised key still fails validation.
        """
        if not isinstance(data, dict):
            return data

        legacy_fields = {
            "chunk_size": ("fixed_size", "chunk_size"),
            "chunk_overlap": ("fixed_size", "chunk_overlap"),
            "respect_headers": ("structure_aware", "respect_headers"),
        }

        migrated = dict(data)
        for legacy_key, (section, field_name) in legacy_fields.items():
            if legacy_key not in migrated:
                continue
            value = migrated.pop(legacy_key)
            section_data = dict(migrated.get(section) or {})
            section_data.setdefault(field_name, value)
            migrated[section] = section_data

        return migrated


class EmbeddingConfig(StrictConfigModel):
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = Field(default=64, ge=1)
    normalize: bool = True


class BM25Config(StrictConfigModel):
    k1: float = Field(default=1.5, ge=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)


class DenseConfig(StrictConfigModel):
    top_k: int = Field(default=20, ge=1)


class HybridConfig(StrictConfigModel):
    method: str = "rrf"
    k: int = Field(default=60, ge=1)
    bm25_weight: float = Field(default=0.5, ge=0.0)
    dense_weight: float = Field(default=0.5, ge=0.0)


class RerankingConfig(StrictConfigModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(default=5, ge=1)


class RetrievalConfig(StrictConfigModel):
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)


class GenerationConfig(StrictConfigModel):
    #: "anthropic", "openai" or "groq" (Groq uses the OpenAI-compatible API).
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    #: Optional override for OpenAI-compatible providers (e.g. a Groq gateway).
    base_url: Optional[str] = None
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class EvaluationConfig(StrictConfigModel):
    eval_set_size: int = Field(default=50, ge=1)
    metrics: list[str] = Field(
        default_factory=lambda: ["recall@k", "mrr", "ndcg@k"]
    )
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])


class APIConfig(StrictConfigModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_latency: bool = True


class CorpusConfig(StrictConfigModel):
    name: str = "stripe_api_docs"
    source: str = "https://docs.stripe.com"
    base_url: str = "https://docs.stripe.com"
    max_pages: int = Field(default=500, ge=1)


class AppConfig(StrictConfigModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def _read_yaml(config_path: Path) -> dict:
    """Read a YAML file and return its top-level mapping."""
    with open(config_path, "r") as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Could not parse configuration file {config_path}: {exc}"
            ) from exc

    if raw_config is None:  # empty file
        return {}

    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Configuration file {config_path} must contain a mapping at the "
            f"top level, got {type(raw_config).__name__}"
        )

    return raw_config


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from a YAML file.

    Falls back to built-in defaults when no configuration file is found. Invalid
    files raise ``ValueError`` with the file path and the validation details.
    """
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

    if config_path is None or not Path(config_path).exists():
        return AppConfig()

    path = Path(config_path)
    raw_config = _read_yaml(path)

    try:
        return AppConfig(**raw_config)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {path}:\n{exc}") from exc


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
