"""
Evaluation dataset construction.

Covers three steps that must stay separable:

1. ``generate_candidate_queries`` — rule-based, corpus-grounded query proposals.
   They are always marked ``candidate``: nothing here is ground truth.
2. ``build_candidate_pools`` — pools candidates from BM25, dense, hybrid RRF and
   hybrid+rerank for both chunking strategies, deduplicated per (query,
   strategy) and exposed to a judge under an anonymous candidate id.
3. ``assign_splits`` / ``validate_dataset`` — a reproducible dev/test split and
   structural validation.

No relevance grade is ever inferred from retrieval rank or score.
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

DATASET_FORMAT_VERSION = 1

CATEGORIES = (
    "terminology",
    "conceptual",
    "api_usage",
    "procedural",
    "troubleshooting",
    "contextual",
)

SYSTEMS = ("bm25", "dense", "hybrid", "hybrid_rerank")
STRATEGIES = ("fixed_size", "structure_aware")

JUDGMENT_STATES = ("pending", "judged")

# Human review states for a query. Only ``accepted`` queries may enter the
# approved evaluation dataset.
QUERY_REVIEW_STATUSES = ("candidate", "accepted", "rejected", "needs_revision")

RELEVANCE_GRADES = {
    0: "Not relevant — does not help answer the query.",
    1: "Marginally relevant — related terminology or background only.",
    2: "Relevant — answers a meaningful part of the query.",
    3: "Highly relevant — directly and sufficiently answers the query.",
}

SPLIT_SEED = 13
SPLIT_VERSION = "split-v1"
TEST_FRACTION = 0.25
POOL_DEPTH = 3
RERANK_CANDIDATES = 6

QUERY_REQUIRED_FIELDS = (
    "query_id",
    "query",
    "category",
    "generation_method",
    "source_doc_ids",
    "source_section",
    "source_passage",
    "corpus_version",
    "validation_status",
    "created_at",
    "split",
)


# ---------------------------------------------------------------------------
# Candidate query blueprints
# ---------------------------------------------------------------------------
# Each blueprint is (question, category, topic, keyword alternatives). A
# blueprint only becomes a query if the frozen corpus actually contains one of
# its keywords; the passage that grounds it is recorded as provenance.
QUERY_BLUEPRINTS = [
    # terminology
    ("What is a PaymentIntent?", "terminology", "payment_intent", ["PaymentIntent", "payment intent"]),
    ("What is a SetupIntent used for?", "terminology", "setup_intent", ["SetupIntent"]),
    ("What is an idempotency key?", "terminology", "idempotency", ["Idempotency-Key", "idempotency key"]),
    ("What does asynchronous capture mean?", "terminology", "async_capture", ["asynchronous capture"]),
    ("What is a PaymentMethod object?", "terminology", "payment_method", ["PaymentMethod"]),
    ("What is a PaymentRecord?", "terminology", "payment_record", ["PaymentRecord"]),
    ("What is a Checkout Session?", "terminology", "checkout_session", ["Checkout Session", "checkout session"]),
    ("What does SCA mean in Stripe payments?", "terminology", "sca", ["Strong Customer Authentication", "SCA"]),
    ("What is manual capture of a payment?", "terminology", "manual_capture", ["manual capture", "capture_method"]),

    # conceptual
    ("How does a PaymentIntent progress from creation to completion?", "conceptual", "pi_lifecycle", ["requires_payment_method", "requires_confirmation"]),
    ("How do PaymentIntents and SetupIntents differ?", "conceptual", "intent_compare", ["SetupIntent"]),
    ("How does asynchronous capture change the confirmation flow?", "conceptual", "async_capture", ["asynchronous capture"]),
    ("Why does Stripe recommend idempotency keys for retries?", "conceptual", "idempotency", ["idempotency"]),
    ("How does Strong Customer Authentication affect payment confirmation?", "conceptual", "sca", ["Strong Customer Authentication", "3D Secure"]),
    ("How does a payment status update reach my server?", "conceptual", "status_updates", ["webhook", "status update"]),
    ("What is the relationship between a PaymentRecord and a PaymentIntent?", "conceptual", "payment_record", ["PaymentRecord"]),
    ("When does a PaymentIntent move to requires_capture?", "conceptual", "manual_capture", ["requires_capture", "manual capture"]),
    ("What happens to a payment that requires additional customer action?", "conceptual", "next_action", ["next_action", "requires_action"]),

    # api usage
    ("Which API endpoint creates a PaymentIntent?", "api_usage", "pi_create", ["/v1/payment_intents", "PaymentIntent"]),
    ("Which parameters are required when creating a SetupIntent?", "api_usage", "si_create", ["SetupIntent"]),
    ("How do I retrieve a PaymentMethod object?", "api_usage", "pm_retrieve", ["PaymentMethod"]),
    ("Which endpoint lists all payment intents for a customer?", "api_usage", "pi_list", ["payment_intents", "customer"]),
    ("How do I send an Idempotency-Key header with a request?", "api_usage", "idempotency", ["Idempotency-Key", "idempotency"]),
    ("Which API creates a Checkout Session?", "api_usage", "checkout_session", ["Checkout Session", "checkout session"]),
    ("How do I cancel a PaymentIntent that is not yet confirmed?", "api_usage", "pi_cancel", ["cancel", "canceled"]),
    ("Which fields does the PaymentIntent object expose for next actions?", "api_usage", "next_action", ["next_action"]),
    ("How do I confirm a PaymentIntent on the server?", "api_usage", "pi_confirm", ["confirm", "confirmation"]),
    ("How do I expand an object in an API response?", "api_usage", "expand", ["expand"]),

    # procedural
    ("What steps are required to accept a payment with the Payment Intents API?", "procedural", "pi_flow", ["accept a payment", "payment intent"]),
    ("How do I set up a webhook endpoint to monitor subscriptions?", "procedural", "webhook_setup", ["webhook"]),
    ("What is the process for migrating from older payment APIs?", "procedural", "migration", ["migration", "older"]),
    ("How do I opt in to asynchronous capture for an existing integration?", "procedural", "async_capture", ["asynchronous capture"]),
    ("What should I implement after finishing a basic subscription integration?", "procedural", "subscription_next", ["subscription", "next steps"]),
    ("How do I prepare a development environment before integrating?", "procedural", "dev_env", ["development environment", "API keys"]),
    ("How do I handle a payment that needs multiple actions?", "procedural", "next_action", ["next_action", "handle_next_action"]),

    # troubleshooting
    ("Why would a PaymentIntent remain in requires_payment_method status?", "troubleshooting", "pi_status", ["requires_payment_method"]),
    ("Why might a card payment fail with an authentication error?", "troubleshooting", "sca", ["authentication", "3D Secure"]),
    ("Why can duplicate charges happen when retrying a request?", "troubleshooting", "idempotency", ["duplicate", "idempotency"]),
    ("Why would a webhook delivery be retried?", "troubleshooting", "webhook_retry", ["retry", "webhook"]),
    ("What happens when a payment is declined during confirmation?", "troubleshooting", "declined", ["declined", "decline"]),
    ("Why might asynchronous capture reduce confirmation latency?", "troubleshooting", "async_capture", ["asynchronous capture", "latency"]),
    ("Why would an integration poll payment status instead of using webhooks?", "troubleshooting", "polling", ["polling", "webhook"]),
    ("What does an incomplete payment status indicate?", "troubleshooting", "incomplete", ["incomplete", "requires_payment_method"]),

    # contextual
    ("When should I use a SetupIntent instead of a PaymentIntent?", "contextual", "intent_compare", ["SetupIntent"]),
    ("When is asynchronous capture appropriate?", "contextual", "async_capture", ["asynchronous capture"]),
    ("When do I need a Checkout Session rather than a direct PaymentIntent integration?", "contextual", "checkout_compare", ["Checkout Session", "PaymentIntent"]),
    ("Which integration path suits a no-code user compared with a developer?", "contextual", "no_code", ["no-code", "no code"]),
    ("When should I use test API keys?", "contextual", "test_keys", ["test mode", "test key", "test keys"]),
    ("When is Strong Customer Authentication required for a payment?", "contextual", "sca", ["Strong Customer Authentication"]),
    ("Which payment methods can be saved for future use?", "contextual", "saved_methods", ["saved", "payment method"]),
    ("When should an integration use the older payment APIs?", "contextual", "legacy", ["older", "legacy"]),

    # spares (used only if earlier blueprints fail to ground)
    ("What is the Charge object used for?", "terminology", "charge", ["Charge object", "charge"]),
    ("How do I collect a payment method for future payments?", "conceptual", "saved_methods", ["setup future usage", "save"]),
    ("What is a customer portal session?", "terminology", "customer_portal", ["customer portal", "Billing Portal"]),
    ("How do I list payment methods attached to a customer?", "api_usage", "pm_list", ["payment_methods", "customer"]),
    ("What does the payment intent status tell me?", "conceptual", "pi_status", ["status"]),
]

TARGET_QUERY_COUNT = 50


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _first_sentence_with(text: str, keyword: str, limit: int = 400) -> str:
    lower = text.lower()
    position = lower.find(keyword.lower())
    if position == -1:
        return " ".join(text.split())[:limit]

    start = max(text.rfind(". ", 0, position), text.rfind("\n", 0, position))
    start = 0 if start == -1 else start + 1
    end = text.find(". ", position)
    end = len(text) if end == -1 else end + 1
    return " ".join(text[start:end].split())[:limit]


def generate_candidate_queries(
    documents: list,
    chunks: list,
    corpus_version: str,
    created_at: Optional[str] = None,
    target_count: int = TARGET_QUERY_COUNT,
) -> tuple[list[dict], list[dict]]:
    """
    Propose candidate queries grounded in real corpus passages.

    Returns ``(queries, skipped)``. Skipped blueprints are reported rather than
    silently dropped, and every produced query is ``validation_status:
    candidate`` — never ``human_verified``.
    """
    from datetime import datetime, timezone

    created_at = created_at or datetime.now(timezone.utc).isoformat()
    docs_by_id = {doc.doc_id: doc for doc in documents}
    ordered_chunks = sorted(chunks, key=lambda c: (c.doc_id, c.chunk_index))

    queries: list[dict] = []
    skipped: list[dict] = []
    seen_queries: set[str] = set()
    per_topic = Counter()

    for question, category, topic, keywords in QUERY_BLUEPRINTS:
        if len(queries) >= target_count:
            break

        normalised_question = _normalise(question)
        if normalised_question in seen_queries:
            skipped.append({"query": question, "reason": "duplicate_query"})
            continue
        if per_topic[topic] >= 3:
            skipped.append({"query": question, "reason": "topic_quota"})
            continue

        match = None
        matched_keyword = None
        for chunk in ordered_chunks:
            lowered = chunk.content.lower()
            keyword = next((k for k in keywords if k.lower() in lowered), None)
            if keyword is None:
                continue
            # Prefer chunks that carry heading context; they make better provenance.
            if match is None or (not match.heading_context and chunk.heading_context):
                match = chunk
                matched_keyword = keyword
            if match.heading_context:
                break

        if match is None:
            skipped.append({"query": question, "reason": "not_grounded_in_corpus"})
            continue

        doc = docs_by_id.get(match.doc_id)
        if doc is None:
            skipped.append({"query": question, "reason": "missing_source_document"})
            continue

        seen_queries.add(normalised_question)
        per_topic[topic] += 1

        queries.append(
            {
                "query_id": f"q_{len(queries) + 1:03d}",
                "query": question,
                "category": category,
                "generation_method": "rule_blueprint+corpus_grounding",
                "source_doc_ids": [doc.doc_id],
                "source_section": (
                    match.heading_context
                    or match.metadata.get("header")
                    or doc.section
                ),
                "source_passage": _first_sentence_with(match.content, matched_keyword),
                "source_chunk_id": match.chunk_id,
                "corpus_version": corpus_version,
                "validation_status": "candidate",
                "created_at": created_at,
                "split": None,
            }
        )

    return queries, skipped


# ---------------------------------------------------------------------------
# Candidate pooling
# ---------------------------------------------------------------------------

def candidate_id(query_id: str, strategy: str, chunk_id: str) -> str:
    """Anonymous, deterministic candidate identifier shown to judges."""
    digest = hashlib.sha1(f"{query_id}|{strategy}|{chunk_id}".encode()).hexdigest()
    return f"cand_{digest[:10]}"


def build_candidate_pools(
    queries: Iterable[dict],
    retrievers_by_strategy: dict[str, dict],
    chunks_by_strategy: dict[str, list],
    corpus_version: str,
    depth: int = POOL_DEPTH,
    rerank_candidates: int = RERANK_CANDIDATES,
) -> tuple[list[dict], list[dict]]:
    """
    Pool candidates across the four systems for each chunking strategy.

    Returns ``(candidates, provenance)``. ``candidates`` is what a judge sees
    (no system names, no ranks). ``provenance`` records which systems proposed
    each candidate, at which rank, and with which score.
    """
    chunks_lookup = {
        strategy: {chunk.chunk_id: chunk for chunk in chunks}
        for strategy, chunks in chunks_by_strategy.items()
    }

    candidates: list[dict] = []
    provenance: list[dict] = []

    for query in queries:
        for strategy, retrievers in retrievers_by_strategy.items():
            lookup = chunks_lookup.get(strategy, {})
            contributions: dict[str, list[dict]] = {}

            def add(system: str, results: list[dict]):
                for result in results:
                    chunk_id = result["chunk_id"]
                    if chunk_id not in lookup:
                        continue  # never pool a chunk from another corpus/strategy
                    contributions.setdefault(chunk_id, []).append(
                        {
                            "system": system,
                            "rank": result.get("rank"),
                            "score": result.get("score", result.get("rerank_score")),
                        }
                    )

            add("bm25", retrievers["bm25"].retrieve(query["query"], depth))
            add("dense", retrievers["dense"].retrieve(query["query"], depth))

            hybrid_results = retrievers["hybrid"].retrieve(query["query"], depth)
            add("hybrid", hybrid_results)

            rerank_input = retrievers["hybrid"].retrieve(
                query["query"], max(depth, rerank_candidates)
            )
            add(
                "hybrid_rerank",
                retrievers["reranker"].rerank(
                    query["query"], rerank_input, top_k=depth
                ),
            )

            for chunk_id, sources in contributions.items():
                chunk = lookup[chunk_id]
                best_rank = min(
                    (source["rank"] for source in sources if source["rank"] is not None),
                    default=None,
                )
                cid = candidate_id(query["query_id"], strategy, chunk_id)

                candidates.append(
                    {
                        "candidate_id": cid,
                        "query_id": query["query_id"],
                        "chunking_strategy": strategy,
                        "chunk_id": chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "heading_context": chunk.heading_context,
                        "heading_path": chunk.heading_path,
                        "url": chunk.metadata.get("url", ""),
                        "title": chunk.metadata.get("title", ""),
                        "corpus_version": corpus_version,
                    }
                )
                provenance.append(
                    {
                        "candidate_id": cid,
                        "query_id": query["query_id"],
                        "chunking_strategy": strategy,
                        "chunk_id": chunk_id,
                        "systems": sources,
                        "num_systems": len({source["system"] for source in sources}),
                        "best_rank": best_rank,
                    }
                )

    candidates.sort(key=lambda c: (c["query_id"], c["chunking_strategy"], c["candidate_id"]))
    provenance.sort(key=lambda p: (p["query_id"], p["chunking_strategy"], p["candidate_id"]))
    return candidates, provenance


# ---------------------------------------------------------------------------
# Splits and validation
# ---------------------------------------------------------------------------

def assign_splits(
    queries: list[dict],
    test_fraction: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> str:
    """
    Assign a stratified dev/test split, deterministically.

    Queries are grouped by category, sorted, shuffled with a fixed seed, and the
    first ``round(n * test_fraction)`` queries per category go to ``test``.
    Returns the split version (content hash of the assignment).
    """
    rng = random.Random(seed)
    by_category: dict[str, list[str]] = defaultdict(list)
    for query in queries:
        by_category[query["category"]].append(query["query_id"])

    assignment: dict[str, str] = {}
    for category in sorted(by_category):
        ids = sorted(by_category[category])
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test_fraction))
        test_ids = set(ids[:n_test])
        for query_id in ids:
            assignment[query_id] = "test" if query_id in test_ids else "dev"

    for query in queries:
        query["split"] = assignment[query["query_id"]]

    digest = hashlib.sha256(
        "\n".join(f"{qid}:{split}" for qid, split in sorted(assignment.items())).encode()
    ).hexdigest()
    return f"split-{digest[:8]}"


def validate_dataset(
    queries: list[dict],
    candidates: list[dict],
    judgments: list[dict],
    chunks_by_strategy: dict[str, list],
    manifest: Optional[dict] = None,
) -> dict:
    """
    Structural validation. Reports problems; never repairs or infers grades.
    """
    problems: list[str] = []
    chunks_lookup = {
        strategy: {chunk.chunk_id: chunk for chunk in chunks}
        for strategy, chunks in chunks_by_strategy.items()
    }

    query_ids = [q.get("query_id") for q in queries]
    duplicate_queries = [qid for qid, n in Counter(query_ids).items() if n > 1]
    if duplicate_queries:
        problems.append(f"duplicate query ids: {sorted(duplicate_queries)}")

    candidate_ids = [c.get("candidate_id") for c in candidates]
    duplicate_candidates = [cid for cid, n in Counter(candidate_ids).items() if n > 1]
    if duplicate_candidates:
        problems.append(f"duplicate candidate ids: {sorted(duplicate_candidates)}")

    by_query = defaultdict(list)
    for candidate in candidates:
        by_query[candidate.get("query_id")].append(candidate)

    manifest_version = (manifest or {}).get("corpus_version")

    for query in queries:
        missing = [field for field in QUERY_REQUIRED_FIELDS if field not in query]
        if missing:
            problems.append(f"{query.get('query_id')}: missing fields {missing}")
        if query.get("category") not in CATEGORIES:
            problems.append(f"{query.get('query_id')}: invalid category {query.get('category')!r}")
        if query.get("split") not in ("dev", "test"):
            problems.append(f"{query.get('query_id')}: invalid split {query.get('split')!r}")
        if query.get("validation_status") not in QUERY_REVIEW_STATUSES:
            problems.append(
                f"{query.get('query_id')}: invalid validation_status "
                f"{query.get('validation_status')!r}"
            )
        if manifest_version and query.get("corpus_version") != manifest_version:
            problems.append(
                f"{query.get('query_id')}: corpus version {query.get('corpus_version')!r} "
                f"does not match manifest {manifest_version!r}"
            )
        if not by_query.get(query.get("query_id")):
            problems.append(f"{query.get('query_id')}: no candidate chunks")

    candidate_lookup = {}
    for candidate in candidates:
        cid = candidate.get("candidate_id")
        candidate_lookup[cid] = candidate
        strategy = candidate.get("chunking_strategy")
        if strategy not in STRATEGIES:
            problems.append(f"{cid}: invalid chunking strategy {strategy!r}")
            continue
        if manifest_version and candidate.get("corpus_version") != manifest_version:
            problems.append(f"{cid}: corpus version mismatch")
        chunk = chunks_lookup.get(strategy, {}).get(candidate.get("chunk_id"))
        if chunk is None:
            problems.append(f"{cid}: chunk {candidate.get('chunk_id')!r} not found in {strategy}")
            continue
        if chunk.doc_id != candidate.get("doc_id"):
            problems.append(f"{cid}: doc_id does not match chunk metadata")
        if not candidate.get("content"):
            problems.append(f"{cid}: empty candidate content")

    judged_pairs = Counter()
    grade_distribution = Counter()
    judged = 0
    for judgment in judgments:
        qid = judgment.get("query_id")
        cid = judgment.get("candidate_id")
        judged_pairs[(qid, cid)] += 1

        grade = judgment.get("relevance_grade")
        if not isinstance(grade, int) or isinstance(grade, bool) or grade not in RELEVANCE_GRADES:
            problems.append(f"judgment {qid}/{cid}: invalid relevance grade {grade!r}")
        status = judgment.get("judgment_status")
        if status not in JUDGMENT_STATES:
            problems.append(f"judgment {qid}/{cid}: invalid state {status!r}")
        if status == "judged":
            judged += 1
            grade_distribution[grade] += 1

        candidate = candidate_lookup.get(cid)
        if candidate is None:
            problems.append(f"judgment {qid}/{cid}: unknown candidate")
            continue
        if candidate.get("query_id") != qid:
            problems.append(f"judgment {qid}/{cid}: query mismatch")
        if judgment.get("chunk_id") != candidate.get("chunk_id"):
            problems.append(f"judgment {qid}/{cid}: chunk mismatch")
        if judgment.get("doc_id") != candidate.get("doc_id"):
            problems.append(f"judgment {qid}/{cid}: doc mismatch")
        if judgment.get("chunking_strategy") != candidate.get("chunking_strategy"):
            problems.append(f"judgment {qid}/{cid}: chunking strategy mismatch")

    superseded = sum(n - 1 for n in judged_pairs.values() if n > 1)

    judged_candidate_ids = {
        j.get("candidate_id")
        for j in judgments
        if j.get("judgment_status") == "judged"
    }
    pending = [c for c in candidates if c["candidate_id"] not in judged_candidate_ids]

    fully_judged_queries = 0
    for query_id in query_ids:
        query_candidates = by_query.get(query_id, [])
        if query_candidates and all(
            c["candidate_id"] in judged_candidate_ids for c in query_candidates
        ):
            fully_judged_queries += 1

    system_coverage = Counter()
    strategy_coverage = Counter(c["chunking_strategy"] for c in candidates)

    return {
        "corpus_version": manifest_version,
        "num_queries": len(queries),
        "queries_by_category": dict(Counter(q.get("category") for q in queries)),
        "queries_by_split": dict(Counter(q.get("split") for q in queries)),
        "num_candidates": len(candidates),
        "candidates_by_strategy": dict(strategy_coverage),
        "judged_candidates": len(judged_candidate_ids),
        "pending_candidates": len(pending),
        "fully_judged_queries": fully_judged_queries,
        "grade_distribution": {str(grade): grade_distribution[grade] for grade in sorted(RELEVANCE_GRADES)},
        "superseded_judgment_records": superseded,
        "system_coverage": dict(system_coverage),
        "problems": problems,
        "is_structurally_valid": not problems,
        "is_fully_judged": bool(candidates) and not pending,
    }


def attach_system_coverage(report: dict, provenance: list[dict]) -> dict:
    """Fill the per-system candidate counts from provenance records."""
    coverage = Counter()
    for record in provenance:
        for system in {source["system"] for source in record.get("systems", [])}:
            coverage[system] += 1
    report["system_coverage"] = dict(coverage)
    return report


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
