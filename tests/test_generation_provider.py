"""Provider wiring tests (no network calls)."""

import pytest

from src.generation.rag import RAGGenerator


def test_groq_uses_openai_compatible_base_url(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    generator = RAGGenerator(provider="groq", model="openai/gpt-oss-20b")
    generator._init_client()

    assert "api.groq.com/openai/v1" in str(generator.client.base_url)


def test_groq_base_url_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    generator = RAGGenerator(provider="groq", base_url="http://localhost:9999/v1")
    generator._init_client()

    assert "localhost:9999" in str(generator.client.base_url)


def test_groq_requires_its_own_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        RAGGenerator(provider="groq")._init_client()


def test_unsupported_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported provider"):
        RAGGenerator(provider="not-a-provider")._init_client()


def test_configured_groq_defaults_are_consistent():
    from src.utils.config import get_config

    config = get_config().generation
    assert config.provider == "groq"
    assert config.model
