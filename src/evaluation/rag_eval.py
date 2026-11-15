"""
RAG answer evaluation, kept separate from retrieval evaluation.

Three layers:

1. Retrieval context for an answer is captured exactly as used, so answer
   claims can be traced back to chunks.
2. Deterministic citation checks (how many citations resolve, how many cited
   chunks are judged relevant, how much judged-relevant retrieved context is
   cited) need no LLM.
3. Answer quality (faithfulness, relevance, completeness) is *human* judgment.
   An optional LLM judge may pre-fill suggestions, but its scores are labelled
   ``llm_judge`` and never presented as human ground truth.

Without provider credentials, answer generation returns ``not_executed`` and
the rest of the evaluation pipeline keeps working.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
HUMAN_SCORE_FIELDS = (
    "faithfulness",
    "answer_relevance",
    "citation_correctness",
    "citation_completeness",
    "completeness",
)


class GenerationNotExecuted(RuntimeError):
    """Raised when answer generation cannot run in this environment."""


def parse_citations(answer: str) -> list[int]:
    """Unique citation numbers, in order of first appearance."""
    seen: list[int] = []
    for match in CITATION_PATTERN.finditer(answer or ""):
        index = int(match.group(1))
        if index not in seen:
            seen.append(index)
    return seen


def evaluate_citations(
    answer: str,
    context_chunks: list[dict],
    grades: Optional[dict[str, int]] = None,
    threshold: int = 2,
) -> dict:
    """
    Citation correctness/completeness against the exact retrieved context.

    ``context_chunks`` is the ordered list given to the model, so citation ``[n]``
    refers to ``context_chunks[n - 1]``. Correctness needs human grades; without
    them it is reported as unknown rather than assumed correct.
    """
    cited = parse_citations(answer)
    valid = [index for index in cited if 1 <= index <= len(context_chunks)]
    invalid = [index for index in cited if index < 1 or index > len(context_chunks)]

    result = {
        "citations": cited,
        "num_citations": len(cited),
        "invalid_citations": invalid,
        "valid_citation_rate": (len(valid) / len(cited)) if cited else None,
        "cited_chunk_ids": [context_chunks[index - 1]["chunk_id"] for index in valid],
        "citation_correctness": None,
        "citation_completeness": None,
    }
    if grades is None:
        return result

    relevant_ids = {chunk_id for chunk_id, grade in grades.items() if grade >= threshold}
    cited_ids = set(result["cited_chunk_ids"])
    if relevant_ids:
        result["citation_completeness"] = len(cited_ids & relevant_ids) / len(relevant_ids)
    if cited_ids:
        result["citation_correctness"] = len(cited_ids & relevant_ids) / len(cited_ids)
    return result


def build_answer_record(
    query: dict,
    ranked: list[dict],
    chunks_by_id: dict[str, dict],
    answer: Optional[str] = None,
    *,
    experiment_id: str,
    prompt_version: str = "rag-prompt-v1",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    latency_ms: Optional[float] = None,
    token_usage: Optional[dict] = None,
) -> dict:
    """Answer record with the exact retrieved context preserved."""
    context = []
    for item in ranked:
        chunk = chunks_by_id.get(item["chunk_id"], {})
        context.append(
            {
                "rank": item["rank"],
                "chunk_id": item["chunk_id"],
                "doc_id": item.get("doc_id"),
                "url": chunk.get("metadata", {}).get("url"),
                "content": chunk.get("content"),
            }
        )
    return {
        "answer_id": f"{experiment_id}:{query['query_id']}",
        "experiment_id": experiment_id,
        "query_id": query["query_id"],
        "query": query["query"],
        "split": query.get("split"),
        "prompt_version": prompt_version,
        "provider": provider,
        "model": model,
        "retrieved_context": context,
        "answer": answer,
        "latency_ms": latency_ms,
        "token_usage": token_usage,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def provider_credentials_available(provider: Optional[str] = None) -> bool:
    provider = provider or "anthropic"
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    return bool(key and os.environ.get(key))


def generate_answer(
    query: str,
    ranked: list[dict],
    chunks_by_id: dict[str, dict],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Generate one answer, or report clearly that generation is not available.

    Never raises for missing credentials: the caller records ``not_executed``.
    """
    import time

    provider = provider or "anthropic"
    if not provider_credentials_available(provider):
        return {
            "status": "not_executed",
            "reason": f"{provider} credentials are not configured; generation skipped",
            "answer": None,
        }

    from src.generation.rag import RAGGenerator

    chunks = []
    for item in ranked:
        chunk = chunks_by_id.get(item["chunk_id"], {})
        chunks.append(
            {
                "content": chunk.get("content", ""),
                "citation": {
                    "url": chunk.get("metadata", {}).get("url"),
                    "title": chunk.get("metadata", {}).get("title"),
                    "chunk_id": item["chunk_id"],
                },
            }
        )

    generator = RAGGenerator(provider=provider, model=model)
    started = time.perf_counter()
    response = generator.generate(query, chunks)
    return {
        "status": "ok",
        "answer": response.answer,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "provider": provider,
        "model": generator.model,
        "token_usage": None,
    }


class HumanAnswerEvalStore:
    """Append-only store of human answer-quality ratings."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: list[dict] = []
        self.reload()

    def reload(self) -> list[dict]:
        if not self.path.exists():
            self._records = []
            return self._records
        self._records = [
            json.loads(line) for line in self.path.read_text().splitlines() if line.strip()
        ]
        return self._records

    def record(
        self,
        *,
        answer_id: str,
        reviewer_id: str,
        scores: dict,
        note: str = "",
        timestamp: Optional[str] = None,
    ) -> dict:
        if not answer_id:
            raise ValueError("answer_id is required")
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        cleaned = {}
        for field in HUMAN_SCORE_FIELDS:
            value = scores.get(field)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
                raise ValueError(f"{field} must be an integer in 1..5, got {value!r}")
            cleaned[field] = value
        if not cleaned:
            raise ValueError("at least one score is required")

        record = {
            "answer_id": answer_id,
            "reviewer_id": reviewer_id.strip(),
            "scores": cleaned,
            "note": (note or "").strip(),
            "source": "human",
            "judgment_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records.append(record)
        return record

    def latest(self) -> dict[str, dict]:
        latest = {}
        for record in self._records:
            latest[record["answer_id"]] = record
        return latest

    def summarize(self, answer_ids: Iterable[str]) -> dict:
        latest = self.latest()
        rows = [latest[answer_id] for answer_id in answer_ids if answer_id in latest]
        summary = {"num_answers": len(list(answer_ids)) if not isinstance(answer_ids, list) else len(answer_ids)}
        summary["num_human_rated"] = len(rows)
        for field in HUMAN_SCORE_FIELDS:
            values = [
                row["scores"][field] for row in rows if field in row.get("scores", {})
            ]
            summary[field] = (sum(values) / len(values)) if values else None
        return summary


def evaluate_answers(
    records: Iterable[dict],
    grades_by_query: dict[str, dict[str, int]],
    human_store: Optional[HumanAnswerEvalStore] = None,
) -> dict:
    """
    Deterministic citation metrics plus whatever human ratings exist.

    Answer quality is never reported as measured when no human has rated it.
    """
    records = list(records)
    per_answer = []
    for record in records:
        grades = grades_by_query.get(record["query_id"], {})
        citations = evaluate_citations(
            record.get("answer") or "", record.get("retrieved_context", []), grades
        )
        human = human_store.latest().get(record["answer_id"]) if human_store else None
        per_answer.append(
            {
                "answer_id": record["answer_id"],
                "query_id": record["query_id"],
                "status": "answered" if record.get("answer") else "not_generated",
                **citations,
                "human_scores": human["scores"] if human else None,
            }
        )

    generated = [row for row in per_answer if row["status"] == "answered"]
    human_rated = [row for row in per_answer if row["human_scores"]]

    citation_values = [
        row["citation_correctness"]
        for row in generated
        if row["citation_correctness"] is not None
    ]
    return {
        "num_answers": len(per_answer),
        "num_generated": len(generated),
        "num_human_rated": len(human_rated),
        "citation_correctness_mean": (
            sum(citation_values) / len(citation_values) if citation_values else None
        ),
        "coverage_with_grades": (
            len(citation_values) / len(generated) if generated else 0.0
        ),
        "answer_quality": {
            "status": "human_rated" if human_rated else "not_evaluated",
            "reason": None
            if human_rated
            else "no human answer ratings recorded; LLM scores are not human ground truth",
            "per_field": {
                field: (
                    sum(row["human_scores"].get(field, 0) for row in human_rated)
                    / max(1, sum(1 for row in human_rated if field in row["human_scores"]))
                )
                if human_rated
                else None
                for field in HUMAN_SCORE_FIELDS
            },
        },
        "per_answer": per_answer,
    }
