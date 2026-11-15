"""Tests for configuration loading, validation and propagation."""

import textwrap
from pathlib import Path

import pytest

from src.ingestion.chunker import FixedSizeChunker, StructureAwareChunker
from src.utils.config import AppConfig, load_config, reset_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(content))
    return path


# ---------------------------------------------------------------------------
# Defaults and repository configuration
# ---------------------------------------------------------------------------

def test_repository_yaml_loads_nested_chunking_values():
    config = load_config(str(DEFAULT_CONFIG_PATH))

    assert config.chunking.fixed_size.chunk_size == 512
    assert config.chunking.fixed_size.chunk_overlap == 50
    assert config.chunking.structure_aware.max_chunk_size == 512
    assert config.chunking.structure_aware.min_chunk_size == 100
    assert config.chunking.structure_aware.respect_headers is True


def test_repository_yaml_other_sections_still_load():
    config = load_config(str(DEFAULT_CONFIG_PATH))

    assert config.corpus.name == "stripe_api_docs"
    assert config.corpus.max_pages == 500
    assert config.embedding.model_name == "all-MiniLM-L6-v2"
    assert config.retrieval.hybrid.k == 60
    assert config.retrieval.bm25.k1 == 1.5
    assert config.retrieval.reranking.top_k == 5
    assert config.generation.provider == "groq"
    assert config.evaluation.k_values == [1, 3, 5, 10]
    assert config.api.port == 8000


def test_default_config_discovery_without_arguments(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    config = load_config()
    assert config.chunking.fixed_size.chunk_size == 512


def test_builtin_defaults_when_no_file_is_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RERANKQ_CONFIG", raising=False)

    config = load_config()

    assert config.chunking.fixed_size.chunk_size == 512
    assert config.chunking.fixed_size.chunk_overlap == 50
    assert config.chunking.structure_aware.max_chunk_size == 512
    assert config.chunking.structure_aware.min_chunk_size == 100


def test_empty_yaml_file_falls_back_to_defaults(tmp_path):
    path = write_config(tmp_path, "\n")
    assert load_config(str(path)) == AppConfig()


# ---------------------------------------------------------------------------
# Custom configuration and propagation to the chunkers
# ---------------------------------------------------------------------------

def test_custom_yaml_chunking_values_load(tmp_path):
    path = write_config(
        tmp_path,
        """
        chunking:
          fixed_size:
            chunk_size: 128
            chunk_overlap: 16
          structure_aware:
            max_chunk_size: 256
            min_chunk_size: 32
            respect_headers: false
        """,
    )

    config = load_config(str(path))

    assert config.chunking.fixed_size.chunk_size == 128
    assert config.chunking.fixed_size.chunk_overlap == 16
    assert config.chunking.structure_aware.max_chunk_size == 256
    assert config.chunking.structure_aware.min_chunk_size == 32
    assert config.chunking.structure_aware.respect_headers is False


def test_fixed_size_parameters_reach_the_chunker(tmp_path):
    path = write_config(
        tmp_path,
        """
        chunking:
          fixed_size:
            chunk_size: 128
            chunk_overlap: 16
        """,
    )
    reset_config(str(path))

    chunker = FixedSizeChunker()

    assert chunker.chunk_size == 128
    assert chunker.chunk_overlap == 16
    assert chunker.effective_overlap == 16
    assert chunker.window_chars == 128 * FixedSizeChunker.CHARS_PER_TOKEN
    assert chunker.step_chars == (128 - 16) * FixedSizeChunker.CHARS_PER_TOKEN


def test_structure_aware_parameters_reach_the_chunker(tmp_path):
    path = write_config(
        tmp_path,
        """
        chunking:
          structure_aware:
            max_chunk_size: 256
            min_chunk_size: 32
            respect_headers: false
        """,
    )
    reset_config(str(path))

    chunker = StructureAwareChunker()

    assert chunker.max_chunk_size == 256
    assert chunker.min_chunk_size == 32
    assert chunker.respect_headers is False


def test_missing_optional_fields_keep_defaults(tmp_path):
    path = write_config(
        tmp_path,
        """
        chunking:
          fixed_size:
            chunk_size: 128
        """,
    )

    config = load_config(str(path))

    assert config.chunking.fixed_size.chunk_size == 128
    assert config.chunking.fixed_size.chunk_overlap == 50
    assert config.chunking.structure_aware.max_chunk_size == 512
    assert config.chunking.structure_aware.min_chunk_size == 100
    assert config.chunking.structure_aware.respect_headers is True


def test_legacy_flat_chunking_keys_are_migrated(tmp_path):
    path = write_config(
        tmp_path,
        """
        chunking:
          chunk_size: 200
          chunk_overlap: 20
        """,
    )

    config = load_config(str(path))

    assert config.chunking.fixed_size.chunk_size == 200
    assert config.chunking.fixed_size.chunk_overlap == 20


# ---------------------------------------------------------------------------
# Invalid and unknown configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "chunking_yaml",
    [
        "  fixed_size:\n    chunk_size: 0\n",
        "  fixed_size:\n    chunk_size: -10\n",
        "  fixed_size:\n    chunk_overlap: -1\n",
        "  fixed_size:\n    chunk_size: many\n",
        "  structure_aware:\n    min_chunk_size: 1024\n    max_chunk_size: 128\n",
        "  structure_aware:\n    max_chunk_size: 0\n",
    ],
    ids=[
        "zero_chunk_size",
        "negative_chunk_size",
        "negative_overlap",
        "non_integer_chunk_size",
        "min_greater_than_max",
        "zero_max_chunk_size",
    ],
)
def test_invalid_chunking_values_are_rejected(tmp_path, chunking_yaml):
    path = write_config(tmp_path, "chunking:\n" + chunking_yaml)

    with pytest.raises(ValueError) as excinfo:
        load_config(str(path))

    assert str(path) in str(excinfo.value)


def test_invalid_non_chunking_values_are_rejected(tmp_path):
    path = write_config(tmp_path, "api:\n  port: 70000\n")

    with pytest.raises(ValueError):
        load_config(str(path))


@pytest.mark.parametrize(
    "content",
    [
        "chunking:\n  fixed_sized:\n    chunk_size: 128\n",
        "chunking:\n  fixed_size:\n    chunk_sizes: 128\n",
        "chunknigs:\n  fixed_size:\n    chunk_size: 128\n",
        "api:\n  prot: 8000\n",
    ],
    ids=["misspelled_section", "misspelled_field", "misspelled_top_level", "misspelled_api_field"],
)
def test_unknown_configuration_keys_are_rejected(tmp_path, content):
    path = write_config(tmp_path, content)

    with pytest.raises(ValueError) as excinfo:
        load_config(str(path))

    message = str(excinfo.value)
    assert "extra_forbidden" in message or "Extra inputs are not permitted" in message


def test_invalid_yaml_syntax_is_rejected(tmp_path):
    path = write_config(tmp_path, "chunking: [unclosed\n")

    with pytest.raises(ValueError) as excinfo:
        load_config(str(path))

    assert "Could not parse configuration file" in str(excinfo.value)


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = write_config(tmp_path, "- just\n- a\n- list\n")

    with pytest.raises(ValueError) as excinfo:
        load_config(str(path))

    assert "must contain a mapping" in str(excinfo.value)


def test_chunkers_use_values_from_repository_yaml():
    fixed = FixedSizeChunker()
    structure = StructureAwareChunker()

    assert (fixed.chunk_size, fixed.chunk_overlap) == (512, 50)
    assert (structure.max_chunk_size, structure.min_chunk_size) == (512, 100)
