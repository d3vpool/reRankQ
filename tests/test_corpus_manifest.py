"""Tests for corpus snapshot manifests."""

import json

import pytest

from src.evaluation.corpus_manifest import (
    build_manifest,
    corpus_version,
    load_manifest,
    sha256_text,
    verify_corpus,
    write_manifest,
)
from src.ingestion.chunker import Chunk
from src.ingestion.crawler import Document


def make_doc(doc_id: str, content: str, url: str = None) -> Document:
    return Document(
        doc_id=doc_id,
        url=url or f"https://example.test/{doc_id}",
        title=f"Title {doc_id}",
        content=content,
        section="Testing",
    )


def make_chunk(chunk_id: str, doc_id: str, strategy: str = "fixed_size") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content="chunk body",
        chunk_index=0,
        strategy=strategy,
        start_char=0,
        end_char=10,
        token_count=2,
        metadata={"url": f"https://example.test/{doc_id}", "title": "Doc"},
    )


DOCS = [
    make_doc("doc-a", "Payment intent documentation body."),
    make_doc("doc-b", "Webhook documentation body."),
]


def test_manifest_contains_provenance_and_hashes():
    manifest = build_manifest(
        documents=DOCS,
        chunks_by_strategy={
            "fixed_size": [make_chunk("c1", "doc-a")],
            "structure_aware": [make_chunk("c2", "doc-a", "structure_aware")],
        },
    )

    assert manifest["corpus_version"].startswith("corpus-")
    assert manifest["num_documents"] == 2
    assert len(manifest["documents"]) == 2
    first = manifest["documents"][0]
    assert len(first["sha256"]) == 64
    assert first["sha256"] == sha256_text("Payment intent documentation body.")
    assert manifest["chunks"]["fixed_size"]["num_chunks"] == 1
    assert manifest["chunks"]["structure_aware"]["num_chunks"] == 1
    assert manifest["tokenizer"]["name"] == "cl100k_base"
    assert manifest["embedding_model"] == "all-MiniLM-L6-v2"
    assert "python" in manifest["software"]


def test_corpus_version_is_stable_under_reordering():
    assert corpus_version(DOCS) == corpus_version(list(reversed(DOCS)))


def test_corpus_version_changes_when_content_changes():
    changed = [DOCS[0], make_doc("doc-b", "Different webhook documentation body.")]
    assert corpus_version(DOCS) != corpus_version(changed)


def test_verify_corpus_accepts_matching_corpus():
    manifest = build_manifest(DOCS, {"fixed_size": []})

    report = verify_corpus(manifest, DOCS)

    assert report["ok"] is True
    assert report["version_matches"] is True
    assert report["changed"] == []


def test_verify_corpus_reports_changed_missing_and_extra():
    manifest = build_manifest(DOCS, {"fixed_size": []})
    current = [
        make_doc("doc-a", "Payment intent documentation body, edited."),
        make_doc("doc-c", "Brand new page."),
    ]

    report = verify_corpus(manifest, current)

    assert report["ok"] is False
    assert report["changed"] == ["doc-a"]
    assert report["missing"] == ["doc-b"]
    assert report["extra"] == ["doc-c"]
    assert report["version_matches"] is False


def test_manifest_roundtrip(tmp_path):
    manifest = build_manifest(DOCS, {"fixed_size": []})
    path = write_manifest(manifest, tmp_path / "corpus_manifest.json")

    loaded = load_manifest(path)

    assert loaded["corpus_version"] == manifest["corpus_version"]
    assert loaded["documents"] == manifest["documents"]


def test_manifest_rejects_empty_corpus():
    with pytest.raises(ValueError):
        build_manifest([], {"fixed_size": []})


def test_manifest_rejects_unknown_format(tmp_path):
    path = tmp_path / "corpus_manifest.json"
    path.write_text(json.dumps({"manifest_format_version": 999}))

    with pytest.raises(ValueError):
        load_manifest(path)


def test_manifest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")
