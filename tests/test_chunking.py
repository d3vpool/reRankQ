"""Tests for fixed-size and structure-aware chunking."""

from pathlib import Path

import pytest

from src.ingestion.chunker import FixedSizeChunker, StructureAwareChunker
from src.ingestion.crawler import Document
from src.utils.config import reset_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_chunking_config(
    tmp_path: Path,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    max_chunk_size: int = 512,
    min_chunk_size: int = 100,
    respect_headers: bool = True,
) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "chunking:\n"
        "  fixed_size:\n"
        f"    chunk_size: {chunk_size}\n"
        f"    chunk_overlap: {chunk_overlap}\n"
        "  structure_aware:\n"
        f"    max_chunk_size: {max_chunk_size}\n"
        f"    min_chunk_size: {min_chunk_size}\n"
        f"    respect_headers: {str(respect_headers).lower()}\n"
    )
    return path


def make_doc(content: str, doc_id: str = "doc-1") -> Document:
    return Document(
        doc_id=doc_id,
        url=f"https://example.test/{doc_id}",
        title="Synthetic Test Document",
        content=content,
        section="Testing",
        subsection="Chunking",
    )


def token_text(count: int, prefix: str = "token") -> str:
    """Text with no paragraph or sentence boundaries."""
    return " ".join(f"{prefix}{i}" for i in range(count))


def sentence_text(sentences: int, letter: str = "a") -> str:
    """Text of fixed-width sentences, each ending in '. '."""
    return "".join(f"{letter * 38}. " for _ in range(sentences))


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------

def test_short_document_is_a_single_chunk(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=128, chunk_overlap=16)))
    chunker = FixedSizeChunker()
    content = "A short document about payments."

    chunks = chunker.chunk_document(make_doc(content))

    assert len(chunks) == 1
    assert chunks[0].content == content
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(content)
    assert chunks[0].token_count == chunker.count_tokens(content)
    assert chunks[0].chunk_index == 0


@pytest.mark.parametrize("content", ["", "   ", "\n\n   \n"])
def test_empty_documents_produce_no_chunks(tmp_path, content):
    reset_config(str(write_chunking_config(tmp_path)))
    chunker = FixedSizeChunker()

    assert chunker.chunk_document(make_doc(content)) == []


def test_long_document_produces_multiple_chunks(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=10, chunk_overlap=4)))
    chunker = FixedSizeChunker()
    content = token_text(120)

    chunks = chunker.chunk_document(make_doc(content))

    assert len(chunks) > 5
    assert chunks[0].start_char == 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    starts = [c.start_char for c in chunks]
    assert starts == sorted(starts)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_consecutive_chunks_actually_overlap(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=10, chunk_overlap=4)))
    chunker = FixedSizeChunker()
    content = token_text(120)

    chunks = chunker.chunk_document(make_doc(content))

    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        overlap_chars = previous.end_char - current.start_char
        # 4 overlap tokens at 4 chars/token == 16 characters of shared text.
        assert overlap_chars >= FixedSizeChunker.CHARS_PER_TOKEN
        # The shared span is visible in both chunks' content.
        assert previous.content[-4:] in current.content


def test_zero_overlap_produces_disjoint_chunks(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=10, chunk_overlap=0)))
    chunker = FixedSizeChunker()
    content = token_text(120)

    chunks = chunker.chunk_document(make_doc(content))

    assert chunker.effective_overlap == 0
    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start_char >= previous.end_char


def test_custom_overlap_changes_the_shared_span(tmp_path):
    content = token_text(120)

    reset_config(str(write_chunking_config(tmp_path, chunk_size=10, chunk_overlap=2)))
    low_overlap = FixedSizeChunker().chunk_document(make_doc(content))

    reset_config(str(write_chunking_config(tmp_path, chunk_size=10, chunk_overlap=7)))
    high_overlap = FixedSizeChunker().chunk_document(make_doc(content))

    low_shared = low_overlap[0].end_char - low_overlap[1].start_char
    high_shared = high_overlap[0].end_char - high_overlap[1].start_char

    assert high_shared > low_shared
    assert len(high_overlap) > len(low_overlap)


def test_overlap_equal_to_chunk_size_is_clamped(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=5, chunk_overlap=5)))
    chunker = FixedSizeChunker()

    chunks = chunker.chunk_document(make_doc(token_text(80)))

    assert chunker.effective_overlap == 4
    assert chunker.step_chars == 1 * FixedSizeChunker.CHARS_PER_TOKEN
    assert len(chunks) > 1
    starts = [c.start_char for c in chunks]
    assert starts == sorted(starts)
    assert starts[0] < starts[-1]


def test_overlap_larger_than_chunk_size_is_clamped(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=8, chunk_overlap=100)))
    chunker = FixedSizeChunker()

    chunks = chunker.chunk_document(make_doc(token_text(40)))

    assert chunker.effective_overlap == 7
    assert chunker.step_chars >= 1
    assert len(chunks) > 1


def test_custom_chunk_size_changes_chunk_count(tmp_path):
    content = token_text(150)

    reset_config(str(write_chunking_config(tmp_path, chunk_size=8, chunk_overlap=0)))
    small = FixedSizeChunker().chunk_document(make_doc(content))

    reset_config(str(write_chunking_config(tmp_path, chunk_size=40, chunk_overlap=0)))
    large = FixedSizeChunker().chunk_document(make_doc(content))

    assert len(small) > len(large)


def test_exact_window_is_one_chunk_and_one_extra_char_adds_a_chunk(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=10, chunk_overlap=0)))
    chunker = FixedSizeChunker()
    window = 10 * FixedSizeChunker.CHARS_PER_TOKEN
    content = "abcdefghij" * 4
    assert len(content) == window

    assert len(chunker.chunk_document(make_doc(content))) == 1
    assert len(chunker.chunk_document(make_doc(content + "x"))) == 2


def test_chunks_stop_at_sentence_boundaries_when_available(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=10, chunk_overlap=0)))
    chunker = FixedSizeChunker()
    content = sentence_text(4)

    chunks = chunker.chunk_document(make_doc(content))

    assert len(chunks) == 4
    for chunk in chunks:
        assert chunk.content.endswith(".")


def test_chunk_metadata_matches_stored_content(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=10, chunk_overlap=4)))
    chunker = FixedSizeChunker()
    doc = make_doc(token_text(60))

    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 2
    for chunk in chunks:
        assert doc.content[chunk.start_char:chunk.end_char] == chunk.content
        assert chunk.token_count == chunker.count_tokens(chunk.content)
        assert chunk.doc_id == doc.doc_id
        assert chunk.strategy == "fixed_size"
        assert chunk.metadata["title"] == doc.title
        assert chunk.metadata["url"] == doc.url


def test_overlapping_windows_do_not_emit_duplicate_content(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, chunk_size=12, chunk_overlap=4)))
    chunker = FixedSizeChunker()

    chunks = chunker.chunk_document(make_doc(token_text(200)))

    assert len(chunks) == len({c.content for c in chunks})
    assert sum(len(c.content) for c in chunks) > len(token_text(200))


# ---------------------------------------------------------------------------
# Structure-aware chunking
# ---------------------------------------------------------------------------

MULTI_SECTION_DOC = """Overview of the payments platform.

# Payments

Payments overview content goes here.

## Payment Intents

How to create a payment intent in your integration.

## Refunds

How to refund a charge once it has been captured.
"""


def test_markdown_headings_are_preserved_in_metadata(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, min_chunk_size=1)))
    chunker = StructureAwareChunker()

    chunks = chunker.chunk_document(make_doc(MULTI_SECTION_DOC))

    headers = [c.metadata["header"] for c in chunks]
    assert "# Payments" in headers
    assert "## Payment Intents" in headers
    assert "## Refunds" in headers
    assert headers[0] == ""
    # Heading text is part of the embedded content, not only metadata.
    assert "## Payment Intents" in chunks[2].content
    assert chunks[2].heading_path == ["# Payments", "## Payment Intents"]
    assert chunks[2].heading_context == "Payments"


def test_multiple_sections_become_separate_chunks(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, min_chunk_size=1)))
    chunker = StructureAwareChunker()

    chunks = chunker.chunk_document(make_doc(MULTI_SECTION_DOC))

    assert len(chunks) == 4
    assert "Payments overview content" in chunks[1].content
    assert "create a payment intent" in chunks[2].content
    assert "refund a charge" in chunks[3].content


def test_small_sections_are_merged(tmp_path):
    reset_config(
        str(write_chunking_config(tmp_path, min_chunk_size=1000, max_chunk_size=4096))
    )
    chunker = StructureAwareChunker()

    chunks = chunker.chunk_document(make_doc(MULTI_SECTION_DOC))

    assert len(chunks) == 1
    for expected in (
        "Overview of the payments platform",
        "Payments overview content",
        "create a payment intent",
        "refund a charge",
    ):
        assert expected in chunks[0].content


def test_merge_threshold_is_configurable(tmp_path):
    doc = make_doc(MULTI_SECTION_DOC)

    reset_config(
        str(write_chunking_config(tmp_path, min_chunk_size=1, max_chunk_size=4096))
    )
    unmerged = StructureAwareChunker().chunk_document(doc)

    reset_config(
        str(write_chunking_config(tmp_path, min_chunk_size=1000, max_chunk_size=4096))
    )
    merged = StructureAwareChunker().chunk_document(doc)

    assert len(merged) < len(unmerged)


def test_oversized_section_is_split(tmp_path):
    reset_config(
        str(
            write_chunking_config(
                tmp_path, max_chunk_size=20, min_chunk_size=1, respect_headers=True
            )
        )
    )
    chunker = StructureAwareChunker()
    paragraphs = "\n\n".join(
        f"Paragraph {i} explains one small detail of the feature." for i in range(20)
    )
    doc = make_doc("# Long Section\n\n" + paragraphs)

    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    assert {c.metadata["header"] for c in chunks} == {"# Long Section"}


def test_configured_max_chunk_size_is_respected(tmp_path):
    paragraphs = "\n\n".join(
        f"Paragraph {i} explains one small detail." for i in range(20)
    )
    doc = make_doc("# Section\n\n" + paragraphs)

    reset_config(
        str(write_chunking_config(tmp_path, max_chunk_size=20, min_chunk_size=1))
    )
    small_max = StructureAwareChunker().chunk_document(doc)

    reset_config(
        str(write_chunking_config(tmp_path, max_chunk_size=200, min_chunk_size=1))
    )
    large_max = StructureAwareChunker().chunk_document(doc)

    assert len(small_max) > len(large_max)
    assert all(c.token_count <= 20 for c in small_max)


def test_configured_min_chunk_size_is_respected(tmp_path):
    doc = make_doc(MULTI_SECTION_DOC)

    reset_config(
        str(write_chunking_config(tmp_path, min_chunk_size=1, max_chunk_size=4096))
    )
    chunks = StructureAwareChunker().chunk_document(doc)

    merged = [c for c in chunks if c.token_count >= 1]
    assert merged
    assert len(chunks) > 1


def test_respect_headers_false_treats_document_as_one_section(tmp_path):
    reset_config(
        str(
            write_chunking_config(
                tmp_path,
                min_chunk_size=0,
                max_chunk_size=4096,
                respect_headers=False,
            )
        )
    )
    chunker = StructureAwareChunker()

    chunks = chunker.chunk_document(make_doc(MULTI_SECTION_DOC))

    assert len(chunks) == 1
    assert chunks[0].metadata["header"] == ""
    assert chunks[0].heading_context == ""
    # Markdown is preserved verbatim and no content is discarded.
    assert "# Payments" in chunks[0].content
    assert "refund a charge" in chunks[0].content


def test_document_without_headings_is_one_chunk(tmp_path):
    reset_config(str(write_chunking_config(tmp_path, min_chunk_size=1)))
    chunker = StructureAwareChunker()

    chunks = chunker.chunk_document(make_doc("Just a short paragraph."))

    assert len(chunks) == 1
    assert chunks[0].metadata["header"] == ""
    assert chunks[0].strategy == "structure_aware"
    assert chunks[0].content == "Just a short paragraph."
    assert chunks[0].source_spans == [[0, 23]]
