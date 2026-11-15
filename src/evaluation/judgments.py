"""
Human relevance judgments.

Judgments are append-only JSONL: a new record for an existing candidate
supersedes the previous one, and nothing is deleted. That keeps a judge's
earlier work recoverable and makes accidental loss unlikely. Unjudged
candidates are *pending*; they are never treated as grade 0.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from src.evaluation.eval_dataset import JUDGMENT_STATES, RELEVANCE_GRADES

JUDGMENT_FIELDS = (
    "query_id",
    "candidate_id",
    "chunk_id",
    "doc_id",
    "chunking_strategy",
    "relevance_grade",
    "judge_id",
    "judgment_status",
    "judgment_timestamp",
    "rationale",
)


class JudgmentStore:
    """Append-only store of human relevance judgments."""

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
        query_id: str,
        candidate_id: str,
        chunk_id: str,
        doc_id: str,
        chunking_strategy: str,
        relevance_grade: int,
        judge_id: str,
        rationale: str = "",
        judgment_status: str = "judged",
        timestamp: Optional[str] = None,
    ) -> dict:
        """Validate and append one judgment record."""
        if not isinstance(relevance_grade, int) or isinstance(relevance_grade, bool):
            raise ValueError(f"relevance_grade must be an int, got {relevance_grade!r}")
        if relevance_grade not in RELEVANCE_GRADES:
            raise ValueError(
                f"relevance_grade must be one of {sorted(RELEVANCE_GRADES)}, got {relevance_grade}"
            )
        if judgment_status not in JUDGMENT_STATES:
            raise ValueError(
                f"judgment_status must be one of {JUDGMENT_STATES}, got {judgment_status!r}"
            )
        if not judge_id or not judge_id.strip():
            raise ValueError("judge_id is required")
        for name, value in (
            ("query_id", query_id),
            ("candidate_id", candidate_id),
            ("chunk_id", chunk_id),
            ("doc_id", doc_id),
            ("chunking_strategy", chunking_strategy),
        ):
            if not value:
                raise ValueError(f"{name} is required")

        record = {
            "query_id": query_id,
            "candidate_id": candidate_id,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunking_strategy": chunking_strategy,
            "relevance_grade": relevance_grade,
            "judge_id": judge_id.strip(),
            "judgment_status": judgment_status,
            "judgment_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "rationale": (rationale or "").strip(),
        }
        self._append(record)
        return record

    # -- reading -------------------------------------------------------------

    def records(self) -> list[dict]:
        return list(self._records)

    def latest_by_candidate(self) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for record in self._records:
            latest[record["candidate_id"]] = record
        return latest

    def get(self, candidate_id: str) -> Optional[dict]:
        return self.latest_by_candidate().get(candidate_id)

    def judged_candidate_ids(self) -> set[str]:
        return {
            candidate_id
            for candidate_id, record in self.latest_by_candidate().items()
            if record.get("judgment_status") == "judged"
        }

    def progress(self, candidates: Iterable[dict]) -> dict:
        """Progress across a candidate pool; pending is counted separately."""
        candidates = list(candidates)
        judged_ids = self.judged_candidate_ids()
        judged = [c for c in candidates if c["candidate_id"] in judged_ids]
        pending = [c for c in candidates if c["candidate_id"] not in judged_ids]

        by_strategy = {}
        for strategy in sorted({c["chunking_strategy"] for c in candidates}):
            subset = [c for c in candidates if c["chunking_strategy"] == strategy]
            subset_judged = [c for c in subset if c["candidate_id"] in judged_ids]
            by_strategy[strategy] = {
                "total": len(subset),
                "judged": len(subset_judged),
                "pending": len(subset) - len(subset_judged),
            }

        by_query = {}
        for candidate in candidates:
            entry = by_query.setdefault(
                candidate["query_id"], {"total": 0, "judged": 0, "pending": 0}
            )
            entry["total"] += 1
            if candidate["candidate_id"] in judged_ids:
                entry["judged"] += 1
            else:
                entry["pending"] += 1

        return {
            "total": len(candidates),
            "judged": len(judged),
            "pending": len(pending),
            "percent_judged": (100.0 * len(judged) / len(candidates)) if candidates else 0.0,
            "by_strategy": by_strategy,
            "by_query": by_query,
        }

    # -- export --------------------------------------------------------------

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
            writer = csv.DictWriter(f, fieldnames=JUDGMENT_FIELDS)
            writer.writeheader()
            for record in self._records:
                writer.writerow({field: record.get(field, "") for field in JUDGMENT_FIELDS})
        return path


def next_unjudged_query(
    queries: Iterable[dict],
    candidates: Iterable[dict],
    store: JudgmentStore,
    strategy: Optional[str] = None,
    split: Optional[str] = None,
) -> Optional[str]:
    """First query (in dataset order) with pending candidates, for resume."""
    judged = store.judged_candidate_ids()
    pending_by_query: dict[str, int] = {}
    for candidate in candidates:
        if strategy and candidate["chunking_strategy"] != strategy:
            continue
        if candidate["candidate_id"] not in judged:
            pending_by_query[candidate["query_id"]] = pending_by_query.get(candidate["query_id"], 0) + 1

    for query in queries:
        if split and query.get("split") != split:
            continue
        if pending_by_query.get(query["query_id"], 0) > 0:
            return query["query_id"]
    return None
