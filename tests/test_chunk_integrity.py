"""Regression tests for chunk integrity: heading context, spans, identity."""

from pathlib import Path

from src.ingestion.chunker import (
    FixedSizeChunker,
    StructureAwareChunker,
    chunk_documents,
)
from src.ingestion.crawler import Document
from src.utils.config import reset_config

NESTED_DOC = """Intro paragraph before any heading.

# Payments

Payments overview.

## Payment Intents

Create a payment intent with the API.

### Errors

Handle card errors carefully.

## Refunds

Refund a captured charge.
"""


def configure(tmp_path: Path, **overrides) -> None:
    values = {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "max_chunk_size": 4096,
        "min_chunk_size": 1,
        "respect_headers": True,
    }
    values.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(
        "chunking:\n"
        "  fixed_size:\n"
        f"    chunk_size: {values['chunk_size']}\n"
        f"    chunk_overlap: {values['chunk_overlap']}\n"
        "  structure_aware:\n"
        f"    max_chunk_size: {values['max_chunk_size']}\n"
        f"    min_chunk_size: {values['min_chunk_size']}\n"
        f"    respect_headers: {str(values['respect_headers']).lower()}\n"
    )
    reset_config(str(path))


def make_doc(content: str = NESTED_DOC, doc_id: str = "doc-1") -> Document:
    return Document(
        doc_id=doc_id,
        url=f"https://example.test/{doc_id}",
        title="Nested Document",
        content=content,
        section="Payments",
        subsection="Intents",
    )


# ---------------------------------------------------------------------------
# Heading context
# ---------------------------------------------------------------------------

def test_ancestor_headings_form_a_breadcrumb(tmp_path):
    configure(tmp_path)

    chunks = StructureAwareChunker().chunk_document(make_doc())

    nested = next(c for c in chunks if c.metadata["header"] == "## Payment Intents")
    assert nested.heading_context == "Payments"
    assert nested.content.startswith("Payments\n\n## Payment Intents\n\n")
    assert nested.heading_path == ["# Payments", "## Payment Intents"]


def test_deep_heading_keeps_full_ancestry(tmp_path):
    configure(tmp_path)

    chunks = StructureAwareChunker().chunk_document(make_doc())

    deep = next(c for c in chunks if c.metadata["header"] == "### Errors")
    assert deep.heading_path == ["# Payments", "## Payment Intents", "### Errors"]
    assert deep.heading_context == "Payments > Payment Intents"
    assert "### Errors" in deep.content


def test_heading_text_is_not_duplicated(tmp_path):
    configure(tmp_path)

    chunks = StructureAwareChunker().chunk_document(make_doc())

    nested = next(c for c in chunks if c.metadata["header"] == "## Payment Intents")
    assert nested.content.count("## Payment Intents") == 1
    assert nested.content.count("Payments") >= 1


def test_merged_chunk_preserves_every_heading(tmp_path):
    configure(tmp_path, min_chunk_size=5000, max_chunk_size=8192)

    chunks = StructureAwareChunker().chunk_document(make_doc())

    assert len(chunks) == 1
    content = chunks[0].content
    for heading in ("# Payments", "## Payment Intents", "### Errors", "## Refunds"):
        assert heading in content
    assert chunks[0].metadata["merged_headings"] == [
        "# Payments",
        "## Payment Intents",
        "### Errors",
        "## Refunds",
    ]


def test_preamble_has_no_heading_context(tmp_path):
    configure(tmp_path)

    chunks = StructureAwareChunker().chunk_document(make_doc())

    assert chunks[0].metadata["header"] == ""
    assert chunks[0].heading_context == ""
    assert chunks[0].content == "Intro paragraph before any heading."


# ---------------------------------------------------------------------------
# Source spans and offsets
# ---------------------------------------------------------------------------

def test_spans_point_at_original_document_text(tmp_path):
    configure(tmp_path)
    doc = make_doc()

    for chunk in StructureAwareChunker().chunk_document(doc):
        assert chunk.source_spans
        for start, end in chunk.source_spans:
            span_text = doc.content[start:end]
            assert span_text
            assert span_text in chunk.content
        assert chunk.start_char == chunk.source_spans[0][0]
        assert chunk.end_char == chunk.source_spans[-1][1]
        flat = [index for span in chunk.source_spans for index in span]
        assert flat == sorted(flat)


def test_merged_chunk_reports_multiple_spans(tmp_path):
    configure(tmp_path, min_chunk_size=5000, max_chunk_size=8192)
    doc = make_doc()

    chunks = StructureAwareChunker().chunk_document(doc)

    assert len(chunks) == 1
    assert len(chunks[0].source_spans) > 1
    assert chunks[0].start_char < chunks[0].end_char


def test_split_section_parts_keep_their_own_heading(tmp_path):
    configure(tmp_path, max_chunk_size=12, min_chunk_size=1)
    paragraphs = "\n\n".join(
        f"Paragraph {i} explains one small part of the payment flow." for i in range(8)
    )
    doc = make_doc("# Payments\n\n" + paragraphs)

    chunks = StructureAwareChunker().chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["header"] == "# Payments"
        assert chunk.content.startswith("# Payments\n\n")


def test_merging_never_exceeds_max_chunk_size(tmp_path):
    configure(tmp_path, max_chunk_size=40, min_chunk_size=30)
    sections = []
    for i in range(12):
        body = " ".join(f"detail{i}word{j}" for j in range(25))
        sections.append(f"## Section {i}\n\n{body}")
    doc = make_doc("\n\n".join(sections))
    chunker = StructureAwareChunker()

    chunks = chunker.chunk_document(doc)

    heading_allowance = 20  # heading context is added after size selection
    assert len(chunks) > 1
    assert all(
        chunk.token_count <= chunker.max_chunk_size + heading_allowance
        for chunk in chunks
    )


def test_single_oversized_paragraph_is_still_split(tmp_path):
    configure(tmp_path, max_chunk_size=20, min_chunk_size=1)
    sentence = "This sentence describes one detail of the payment flow. "
    doc = make_doc("# Payments\n\n" + sentence * 40)

    chunks = StructureAwareChunker().chunk_document(doc)

    assert len(chunks) > 1
    assert all(chunk.metadata["header"] == "# Payments" for chunk in chunks)
    assert all(chunk.source_spans for chunk in chunks)


def test_unbreakable_text_is_hard_split(tmp_path):
    configure(tmp_path, max_chunk_size=10, min_chunk_size=1)
    doc = make_doc("# Payments\n\n" + "x" * 2000)

    chunks = StructureAwareChunker().chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content
        for start, end in chunk.source_spans:
            assert doc.content[start:end] in chunk.content


def test_fixed_size_spans_match_content_exactly(tmp_path):
    configure(tmp_path, chunk_size=10, chunk_overlap=4)
    content = " ".join(f"word{i}" for i in range(80))
    doc = make_doc(content)

    for chunk in FixedSizeChunker().chunk_document(doc):
        assert chunk.source_spans == [[chunk.start_char, chunk.end_char]]
        assert doc.content[chunk.start_char:chunk.end_char] == chunk.content
        assert chunk.heading_path == []
        assert chunk.heading_context == ""


# ---------------------------------------------------------------------------
# Chunk identity and stability
# ---------------------------------------------------------------------------

def test_chunk_ids_are_deterministic(tmp_path):
    configure(tmp_path)
    doc = make_doc()

    first = [c.chunk_id for c in StructureAwareChunker().chunk_document(doc)]
    second = [c.chunk_id for c in StructureAwareChunker().chunk_document(doc)]

    assert first == second
    assert len(first) == len(set(first))


def test_chunk_ids_do_not_collide_across_documents(tmp_path):
    configure(tmp_path)
    doc_a = make_doc(doc_id="doc-a")
    doc_b = make_doc(doc_id="doc-b")

    ids_a = [c.chunk_id for c in StructureAwareChunker().chunk_document(doc_a)]
    ids_b = [c.chunk_id for c in StructureAwareChunker().chunk_document(doc_b)]

    assert not set(ids_a) & set(ids_b)


def test_reingestion_is_idempotent(tmp_path):
    configure(tmp_path)
    documents = [make_doc(doc_id="doc-a"), make_doc(doc_id="doc-b")]
    output = tmp_path / "processed"

    first = chunk_documents(documents, strategy="both", output_dir=str(output))
    second = chunk_documents(documents, strategy="both", output_dir=str(output))

    for strategy in ("fixed_size", "structure_aware"):
        assert len(first[strategy]) == len(second[strategy])
        assert [c.chunk_id for c in first[strategy]] == [c.chunk_id for c in second[strategy]]
        lines = (output / f"chunks_{strategy}.jsonl").read_text().strip().splitlines()
        assert len(lines) == len(second[strategy])


def test_chunk_order_follows_document_order(tmp_path):
    configure(tmp_path, chunk_size=10, chunk_overlap=2)
    content = " ".join(f"word{i}" for i in range(120))
    doc = make_doc(content)

    chunks = FixedSizeChunker().chunk_document(doc)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert [c.start_char for c in chunks] == sorted(c.start_char for c in chunks)
    assert [c.chunk_id for c in chunks] == list(dict.fromkeys(c.chunk_id for c in chunks))


def test_chunk_metadata_contains_source_information(tmp_path):
    configure(tmp_path)
    doc = make_doc()

    chunks = StructureAwareChunker().chunk_document(doc)

    for chunk in chunks:
        assert chunk.doc_id == doc.doc_id
        assert chunk.metadata["url"] == doc.url
        assert chunk.metadata["title"] == doc.title
        assert chunk.metadata["section"] == doc.section
        assert chunk.strategy == "structure_aware"
