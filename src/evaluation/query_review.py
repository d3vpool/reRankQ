"""
Human query review, quality heuristics and the approved evaluation set.

Reviewing a query (is it a good evaluation question?) is a separate human task
from judging chunks (is this chunk relevant?). This module stores review
decisions append-only, keeps the original wording and every edit, and builds the
approved dataset from ``accepted`` queries only.

Nothing here marks a query accepted on its own: automated checks produce
warnings for a human to act on.
"""

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from src.evaluation.eval_dataset import CATEGORIES, QUERY_REVIEW_STATUSES
from src.evaluation.splits import jaccard, normalise_text, query_tokens

REVIEW_FORMAT_VERSION = 1
APPROVED_FORMAT_VERSION = 1

REVIEW_FIELDS = (
    "query_id",
    "original_query",
    "current_query",
    "original_category",
    "current_category",
    "source_doc_ids",
    "source_section",
    "source_passage",
    "source_chunk_id",
    "corpus_version",
    "review_status",
    "reviewer_id",
    "review_timestamp",
    "review_note",
    "revision_history",
)

REQUIRED_REVIEW_FIELDS = (
    "query_id",
    "original_query",
    "current_query",
    "original_category",
    "current_category",
    "source_doc_ids",
    "source_section",
    "corpus_version",
    "review_status",
    "reviewer_id",
    "review_timestamp",
    "revision_history",
)


class ReviewError(ValueError):
    """Raised when a review decision is not structurally valid."""


class ReviewStore:
    """Append-only store of query review decisions (latest record wins)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: list[dict] = []
        self.reload()

    # -- persistence ---------------------------------------------------------

    def reload(self) -> list[dict]:
        if not self.path.exists():
            self._records = []
            return self._records
        records = []
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        self._records = records
        return self._records

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._records.append(record)

    # -- writing -------------------------------------------------------------

    def record(
        self,
        *,
        query: dict,
        review_status: str,
        reviewer_id: str,
        review_note: str = "",
        query_text: Optional[str] = None,
        category: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Append one review decision.

        ``query`` is the original candidate record. Edits never overwrite the
        original wording: the new wording is stored separately and a revision
        entry is appended to the history.
        """
        if review_status not in QUERY_REVIEW_STATUSES:
            raise ReviewError(
                f"review_status must be one of {QUERY_REVIEW_STATUSES}, got {review_status!r}"
            )
        if not reviewer_id or not reviewer_id.strip():
            raise ReviewError("reviewer_id is required")
        if not query or not query.get("query_id"):
            raise ReviewError("query record with query_id is required")

        previous = self.latest().get(query["query_id"])
        original_query = (previous or {}).get("original_query") or query["query"]
        original_category = (previous or {}).get("original_category") or query["category"]
        base_query = (previous or {}).get("current_query") or original_query
        base_category = (previous or {}).get("current_category") or original_category
        revisions = list((previous or {}).get("revision_history") or [])

        new_query = (query_text if query_text is not None else base_query).strip()
        new_category = category or base_category
        if not new_query:
            raise ReviewError("current_query cannot be empty")
        if new_category not in CATEGORIES:
            raise ReviewError(f"category must be one of {CATEGORIES}, got {new_category!r}")

        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        if new_query != base_query or new_category != base_category:
            revisions.append(
                {
                    "timestamp": timestamp,
                    "reviewer_id": reviewer_id.strip(),
                    "from_query": base_query,
                    "to_query": new_query,
                    "from_category": base_category,
                    "to_category": new_category,
                    "note": (review_note or "").strip(),
                }
            )

        record = {
            "query_id": query["query_id"],
            "original_query": original_query,
            "current_query": new_query,
            "original_category": original_category,
            "current_category": new_category,
            "source_doc_ids": list(query.get("source_doc_ids") or []),
            "source_section": query.get("source_section") or "",
            "source_passage": query.get("source_passage") or "",
            "source_chunk_id": query.get("source_chunk_id"),
            "corpus_version": query.get("corpus_version"),
            "review_status": review_status,
            "reviewer_id": reviewer_id.strip(),
            "review_timestamp": timestamp,
            "review_note": (review_note or "").strip(),
            "revision_history": revisions,
        }
        self._append(record)
        return record

    # -- reading -------------------------------------------------------------

    def records(self) -> list[dict]:
        return list(self._records)

    def latest(self) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for record in self._records:
            latest[record["query_id"]] = record
        return latest

    def get(self, query_id: str) -> Optional[dict]:
        return self.latest().get(query_id)

    def status(self, query_id: str) -> str:
        record = self.get(query_id)
        return record["review_status"] if record else "candidate"

    def status_counts(self) -> dict:
        counts = {status: 0 for status in QUERY_REVIEW_STATUSES}
        for record in self.latest().values():
            counts[record["review_status"]] = counts.get(record["review_status"], 0) + 1
        return counts

    def export_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self._records:
                f.write(json.dumps(record) + "\n")
        return path

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for record in self.latest().values():
                row = {field: record.get(field, "") for field in REVIEW_FIELDS}
                row["source_doc_ids"] = ",".join(record.get("source_doc_ids") or [])
                row["revision_history"] = json.dumps(record.get("revision_history") or [])
                writer.writerow(row)
        return path


# ---------------------------------------------------------------------------
# Effective query view and progress
# ---------------------------------------------------------------------------

def effective_queries(queries: Iterable[dict], store: ReviewStore) -> list[dict]:
    """Candidate queries with the latest review applied (wording, category, status)."""
    latest = store.latest()
    merged = []
    for query in queries:
        review = latest.get(query["query_id"])
        current = dict(query)
        current["original_query"] = query["query"]
        current["original_category"] = query["category"]
        if review:
            current["query"] = review["current_query"]
            current["category"] = review["current_category"]
            current["validation_status"] = review["review_status"]
            current["review"] = review
        else:
            current["validation_status"] = "candidate"
            current["review"] = None
        merged.append(current)
    return merged


def review_progress(queries: Iterable[dict], store: ReviewStore) -> dict:
    queries = list(queries)
    merged = effective_queries(queries, store)
    by_status = {status: 0 for status in QUERY_REVIEW_STATUSES}
    by_category_status: dict[str, dict[str, int]] = {}
    for query in merged:
        by_status[query["validation_status"]] = by_status.get(query["validation_status"], 0) + 1
        bucket = by_category_status.setdefault(
            query["category"], {status: 0 for status in QUERY_REVIEW_STATUSES}
        )
        bucket[query["validation_status"]] += 1

    reviewed = len(queries) - by_status["candidate"]
    return {
        "total": len(queries),
        "by_status": by_status,
        "by_category_status": by_category_status,
        "reviewed": reviewed,
        "unreviewed": by_status["candidate"],
        "percent_reviewed": (100.0 * reviewed / len(queries)) if queries else 0.0,
        "accepted": by_status["accepted"],
        "is_fully_reviewed": bool(queries) and by_status["candidate"] == 0,
    }


def next_unreviewed_query(
    queries: Iterable[dict],
    store: ReviewStore,
    category: Optional[str] = None,
    only_status: Optional[str] = None,
) -> Optional[str]:
    latest = store.latest()
    for query in queries:
        if category and query["category"] != category:
            continue
        status = latest[query["query_id"]]["review_status"] if query["query_id"] in latest else "candidate"
        if only_status and status != only_status:
            continue
        if status == "candidate" or only_status == status:
            return query["query_id"]
    return None


# ---------------------------------------------------------------------------
# Validation and quality heuristics
# ---------------------------------------------------------------------------

def validate_review_record(record: dict) -> None:
    missing = [field for field in REQUIRED_REVIEW_FIELDS if field not in record]
    if missing:
        raise ReviewError(f"review record missing fields {missing}")
    if record["review_status"] not in QUERY_REVIEW_STATUSES:
        raise ReviewError(f"invalid review_status {record['review_status']!r}")
    if record["current_category"] not in CATEGORIES:
        raise ReviewError(f"invalid current_category {record['current_category']!r}")
    if not record["reviewer_id"]:
        raise ReviewError("reviewer_id is required")
    if not record["review_timestamp"]:
        raise ReviewError("review_timestamp is required")
    if not record["current_query"]:
        raise ReviewError("current_query is required")
    if (
        record["current_query"] != record["original_query"]
        or record["current_category"] != record["original_category"]
    ) and not record["revision_history"]:
        raise ReviewError("edited query must record revision history")


def review_validation_problems(records: Iterable[dict], queries: Iterable[dict]) -> list[str]:
    """Structural problems in the review store, independent of human approval."""
    queries = {query["query_id"]: query for query in queries}
    problems: list[str] = []
    for record in records:
        query_id = record.get("query_id")
        try:
            validate_review_record(record)
        except ReviewError as error:
            problems.append(f"{query_id}: {error}")
            continue
        query = queries.get(query_id)
        if query is None:
            problems.append(f"{query_id}: review references an unknown query")
            continue
        if record["original_query"] != query["query"]:
            problems.append(f"{query_id}: original_query does not match the candidate query")
        if record["original_category"] != query["category"]:
            problems.append(f"{query_id}: original_category does not match the candidate query")
        if record["corpus_version"] != query.get("corpus_version"):
            problems.append(f"{query_id}: review corpus version does not match the query")
        if record["review_status"] == "accepted" and not record["review_timestamp"]:
            problems.append(f"{query_id}: accepted query is missing review metadata")
        if record["review_status"] in ("rejected", "needs_revision") and not record["review_note"]:
            problems.append(f"{query_id}: {record['review_status']} decision has no review note")
    return problems


def _corpus_vocabulary(documents: Iterable) -> set[str]:
    vocabulary: set[str] = set()
    for document in documents:
        vocabulary |= query_tokens(document.content)
    return vocabulary


def query_quality_warnings(
    queries: Iterable[dict],
    documents: Optional[Iterable] = None,
    chunks_by_strategy: Optional[dict] = None,
    near_duplicate_threshold: float = 0.6,
    min_tokens: int = 4,
) -> list[dict]:
    """
    Heuristic warnings for human review. Never rejects or accepts anything.

    Each warning is ``{query_id, kind, message}``; a human decides what to do.
    """
    queries = list(queries)
    warnings: list[dict] = []

    def warn(query_id: str, kind: str, message: str) -> None:
        warnings.append({"query_id": query_id, "kind": kind, "message": message})

    vocabulary = _corpus_vocabulary(documents) if documents is not None else None
    doc_ids = {document.doc_id for document in documents} if documents is not None else None
    doc_titles = (
        {normalise_text(document.title) for document in documents} if documents is not None else set()
    )
    chunk_ids = (
        {
            strategy: {chunk.chunk_id for chunk in chunks}
            for strategy, chunks in chunks_by_strategy.items()
        }
        if chunks_by_strategy
        else None
    )

    normalised_counts: dict[str, list[str]] = {}
    for query in queries:
        normalised_counts.setdefault(normalise_text(query["query"]), []).append(query["query_id"])

    for query in queries:
        query_id = query["query_id"]
        text = query.get("query") or ""
        tokens = query_tokens(text)

        if len(tokens) < min_tokens:
            warn(query_id, "short_query", f"query has only {len(tokens)} content tokens")
        if not query.get("source_doc_ids"):
            warn(query_id, "missing_source_docs", "no source_doc_ids recorded")
        if not query.get("source_section"):
            warn(query_id, "missing_source_section", "no source_section recorded")
        if not query.get("source_passage"):
            warn(query_id, "missing_source_passage", "no source_passage recorded")
        if query.get("category") not in CATEGORIES:
            warn(query_id, "invalid_category", f"category {query.get('category')!r} is not allowed")

        section = normalise_text(query.get("source_section") or "")
        section_parts = {normalise_text(part) for part in (query.get("source_section") or "").split(">")}
        heading_copy = bool(section) and normalise_text(text) == section
        heading_copy = heading_copy or normalise_text(text) in (section_parts | doc_titles)
        if heading_copy:
            warn(query_id, "heading_copy", "query wording copies a heading or document title")

        passage = normalise_text(query.get("source_passage") or "")
        if len(tokens) >= min_tokens and passage and normalise_text(text) in passage:
            warn(query_id, "answer_verbatim", "query wording appears verbatim in the grounding passage")
        if "##" in (query.get("source_passage") or "") or "\n" in (query.get("source_passage") or ""):
            warn(query_id, "passage_markup", "grounding passage contains heading markup or newlines")

        if len(normalised_counts[normalise_text(text)]) > 1:
            warn(query_id, "duplicate_query", "identical wording appears more than once")
        for other in queries:
            if other["query_id"] == query_id:
                continue
            similarity = jaccard(tokens, query_tokens(other["query"]))
            if similarity >= near_duplicate_threshold and normalise_text(other["query"]) != normalise_text(text):
                warn(
                    query_id,
                    "near_duplicate",
                    f"{similarity:.2f} token overlap with {other['query_id']}",
                )

        if vocabulary is not None:
            unknown = tokens - vocabulary
            if tokens and len(unknown) / len(tokens) > 0.5:
                warn(
                    query_id,
                    "possibly_unanswerable",
                    f"unknown to corpus: {sorted(unknown)[:5]}",
                )
        if doc_ids is not None:
            missing_docs = [doc for doc in query.get("source_doc_ids") or [] if doc not in doc_ids]
            if missing_docs:
                warn(query_id, "unknown_source_doc", f"documents not in corpus: {missing_docs}")
        if chunk_ids is not None:
            chunk_id = query.get("source_chunk_id")
            if chunk_id and not any(chunk_id in ids for ids in chunk_ids.values()):
                warn(query_id, "unknown_source_chunk", f"chunk {chunk_id} not in any strategy")

    order = {query["query_id"]: index for index, query in enumerate(queries)}
    warnings.sort(key=lambda w: (order.get(w["query_id"], 0), w["kind"]))
    return warnings


def query_review_report(
    queries: Iterable[dict],
    records: Iterable[dict],
    documents: Optional[Iterable] = None,
    chunks_by_strategy: Optional[dict] = None,
    manifest: Optional[dict] = None,
) -> dict:
    """Counts, warnings and structural problems for query review."""
    queries = list(queries)
    records = list(records)
    merged = []
    latest = {record["query_id"]: record for record in records}
    for query in queries:
        review = latest.get(query["query_id"])
        entry = dict(query)
        if review:
            entry["query"] = review["current_query"]
            entry["category"] = review["current_category"]
            entry["validation_status"] = review["review_status"]
        else:
            entry["validation_status"] = "candidate"
        merged.append(entry)

    warnings = query_quality_warnings(merged, documents, chunks_by_strategy)
    problems = review_validation_problems(records, queries)

    if manifest:
        for query in queries:
            if query.get("corpus_version") != manifest.get("corpus_version"):
                problems.append(
                    f"{query['query_id']}: corpus version {query.get('corpus_version')!r} "
                    f"does not match manifest {manifest.get('corpus_version')!r}"
                )

    by_status = {status: 0 for status in QUERY_REVIEW_STATUSES}
    for entry in merged:
        by_status[entry["validation_status"]] = by_status.get(entry["validation_status"], 0) + 1

    warnings_by_kind: dict[str, int] = {}
    attention: set[str] = set()
    for warning in warnings:
        warnings_by_kind[warning["kind"]] = warnings_by_kind.get(warning["kind"], 0) + 1
        attention.add(warning["query_id"])

    return {
        "num_queries": len(queries),
        "by_status": by_status,
        "by_category": dict(
            sorted(
                {
                    category: sum(1 for entry in merged if entry["category"] == category)
                    for category in {entry["category"] for entry in merged}
                }.items()
            )
        ),
        "by_category_status": {
            category: dict(
                sorted(
                    {
                        status: sum(
                            1
                            for entry in merged
                            if entry["category"] == category
                            and entry["validation_status"] == status
                        )
                        for status in QUERY_REVIEW_STATUSES
                    }.items()
                )
            )
            for category in sorted({entry["category"] for entry in merged})
        },
        "warnings": warnings,
        "warnings_by_kind": dict(sorted(warnings_by_kind.items())),
        "queries_requiring_attention": sorted(attention),
        "structural_problems": problems,
        "is_structurally_valid": not problems,
        "human_approved": by_status["accepted"],
        "is_fully_reviewed": bool(queries) and by_status["candidate"] == 0,
    }


# ---------------------------------------------------------------------------
# Candidate-pool baseline and staleness
# ---------------------------------------------------------------------------

POOL_BASELINE_FILENAME = "candidate_pool_baseline.json"


def build_pool_baseline(queries: Iterable[dict], source: str, created_at: Optional[str] = None) -> dict:
    return {
        "baseline_format_version": 1,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "source": source,
        "wording": {query["query_id"]: query["query"] for query in queries},
    }


def write_pool_baseline(queries: Iterable[dict], path: str | Path, source: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_pool_baseline(queries, source), indent=2) + "\n")
    return path


def load_pool_baseline(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No candidate-pool baseline at {path}")
    return json.loads(path.read_text())


def candidate_pool_status(
    queries: Iterable[dict],
    baseline: Optional[dict],
    store: Optional[ReviewStore] = None,
) -> dict:
    """
    Compare each query's current wording with the wording its pool was built from.

    A changed wording makes the pool stale: candidates were retrieved for the
    old query and cannot be treated as the pool for the new one.
    """
    wording = (baseline or {}).get("wording", {})
    entries = []
    for query in queries:
        current = query["query"]
        if store is not None:
            review = store.get(query["query_id"])
            if review:
                current = review["current_query"]
        if query["query_id"] not in wording:
            status = "missing_baseline"
            baseline_query = None
        elif wording[query["query_id"]] != current:
            status = "stale"
            baseline_query = wording[query["query_id"]]
        else:
            status = "current"
            baseline_query = wording[query["query_id"]]
        entries.append(
            {
                "query_id": query["query_id"],
                "current_query": current,
                "pool_query": baseline_query,
                "status": status,
            }
        )

    counts = {"current": 0, "stale": 0, "missing_baseline": 0}
    for entry in entries:
        counts[entry["status"]] += 1
    return {
        "entries": entries,
        "stale": [entry["query_id"] for entry in entries if entry["status"] == "stale"],
        "missing_baseline": [
            entry["query_id"] for entry in entries if entry["status"] == "missing_baseline"
        ],
        "counts": counts,
        "is_consistent": counts["stale"] == 0 and counts["missing_baseline"] == 0,
    }


def replace_query_pools(
    candidates: Iterable[dict],
    provenance: Iterable[dict],
    query_id: str,
    new_candidates: Iterable[dict],
    new_provenance: Iterable[dict],
) -> tuple[list[dict], list[dict]]:
    """Replace one query's pool rows, keeping rows for every other query."""
    kept_candidates = [row for row in candidates if row.get("query_id") != query_id]
    kept_provenance = [row for row in provenance if row.get("query_id") != query_id]
    kept_candidates.extend(new_candidates)
    kept_provenance.extend(new_provenance)
    kept_candidates.sort(key=lambda c: (c["query_id"], c["chunking_strategy"], c["candidate_id"]))
    kept_provenance.sort(key=lambda p: (p["query_id"], p["chunking_strategy"], p["candidate_id"]))
    return kept_candidates, kept_provenance


# ---------------------------------------------------------------------------
# Approved dataset
# ---------------------------------------------------------------------------

def build_approved_dataset(
    queries: Iterable[dict],
    store: ReviewStore,
    split_groups: Optional[dict[str, str]] = None,
    pool_status: Optional[dict] = None,
) -> tuple[list[dict], dict]:
    """
    Rows for ``accepted`` queries only, plus metadata.

    Rejected and needs-revision queries are never included, and their review
    records stay in the review store for auditability.
    """
    latest = store.latest()
    pool_entries = {
        entry["query_id"]: entry for entry in (pool_status or {}).get("entries", [])
    }

    approved = []
    for query in queries:
        review = latest.get(query["query_id"])
        if not review or review["review_status"] != "accepted":
            continue
        pool_entry = pool_entries.get(query["query_id"], {})
        approved.append(
            {
                "query_id": query["query_id"],
                "query": review["current_query"],
                "category": review["current_category"],
                "original_query": review["original_query"],
                "original_category": review["original_category"],
                "source_doc_ids": list(review.get("source_doc_ids") or []),
                "source_section": review.get("source_section") or "",
                "source_passage": review.get("source_passage") or "",
                "source_chunk_id": review.get("source_chunk_id"),
                "corpus_version": review.get("corpus_version"),
                "split": query.get("split"),
                "split_group": (split_groups or {}).get(query["query_id"]),
                "review": {
                    "status": review["review_status"],
                    "reviewer_id": review["reviewer_id"],
                    "review_timestamp": review["review_timestamp"],
                    "review_note": review["review_note"],
                    "revision_history": review["revision_history"],
                },
                "pool": {
                    "query_wording": pool_entry.get("pool_query"),
                    "status": pool_entry.get("status", "unknown"),
                    "is_stale": pool_entry.get("status") == "stale",
                },
            }
        )

    approved.sort(key=lambda row: row["query_id"])
    version = approved_dataset_version(approved)
    meta = {
        "approved_format_version": APPROVED_FORMAT_VERSION,
        "approved_version": version,
        "num_queries": len(approved),
        "by_split": {
            split: sum(1 for row in approved if row["split"] == split)
            for split in ("dev", "test")
        },
        "by_category": {
            category: sum(1 for row in approved if row["category"] == category)
            for category in sorted({row["category"] for row in approved})
        },
        "stale_pool_count": sum(1 for row in approved if row["pool"]["is_stale"]),
        "awaiting_human_review": len(approved) == 0,
    }
    return approved, meta


def approved_dataset_version(records: Iterable[dict]) -> str:
    """Deterministic hash of the approved content (wording, category, split)."""
    rows = sorted(
        f"{row['query_id']}\t{row['query']}\t{row['category']}\t{row['split']}\t{row['corpus_version']}"
        for row in records
    )
    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return f"approved-{digest[:12]}"


def validate_approved_dataset(records: Iterable[dict], queries: Optional[Iterable[dict]] = None) -> list[str]:
    """Problems with the approved set: only accepted, complete, split-assigned."""
    records = list(records)
    problems: list[str] = []
    known_ids = {query["query_id"] for query in queries} if queries is not None else None
    for row in records:
        query_id = row.get("query_id")
        if known_ids is not None and query_id not in known_ids:
            problems.append(f"{query_id}: approved query is not in the candidate dataset")
        if not row.get("query"):
            problems.append(f"{query_id}: approved wording is empty")
        if row.get("category") not in CATEGORIES:
            problems.append(f"{query_id}: invalid category {row.get('category')!r}")
        if row.get("split") not in ("dev", "test"):
            problems.append(f"{query_id}: invalid split {row.get('split')!r}")
        review = row.get("review") or {}
        if review.get("status") != "accepted":
            problems.append(f"{query_id}: approved row is not marked accepted")
        if not review.get("reviewer_id"):
            problems.append(f"{query_id}: approved row is missing reviewer_id")
        if not review.get("review_timestamp"):
            problems.append(f"{query_id}: approved row is missing review_timestamp")
        if row.get("query") != row.get("original_query") and not review.get("revision_history"):
            problems.append(f"{query_id}: edited wording without revision history")
    return problems
