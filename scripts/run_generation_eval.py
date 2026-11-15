#!/usr/bin/env python
"""
RAG answer evaluation, separate from retrieval evaluation.

Captures the exact retrieved context, optionally generates answers, computes
deterministic citation metrics, and summarises human answer ratings if any
exist. Missing provider credentials or approved queries are reported as
``not_executed`` instead of failing.

Usage:
    python scripts/run_generation_eval.py --strategy structure_aware
    python scripts/run_generation_eval.py --strategy structure_aware --provider anthropic
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.eval_dataset import load_jsonl, write_jsonl  # noqa: E402
from src.evaluation.rag_eval import (  # noqa: E402
    HumanAnswerEvalStore,
    build_answer_record,
    evaluate_answers,
    generate_answer,
    provider_credentials_available,
)
from src.evaluation.retrieval_runner import RetrievalRunner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="structure_aware", choices=("fixed_size", "structure_aware"))
    parser.add_argument("--config", default="hybrid_rerank")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="")
    parser.add_argument("--split", default="all", choices=("dev", "test", "all"))
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--output-root", default="results/generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = RetrievalRunner(eval_dir=args.eval_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / f"{args.strategy}-{args.config}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    queries = runner.queries(query_set="approved", split=args.split)
    summary_path = output_dir / "summary.json"

    if not queries:
        summary = {
            "status": "not_executed",
            "reason": (
                "no human-approved queries: approved_queries.jsonl is empty. "
                "Generation evaluation only uses approved queries."
            ),
            "corpus_version": runner.corpus_version,
            "query_version": runner.query_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Generation evaluation not executed: {summary['reason']}")
        print(f"Wrote {summary_path}")
        return 0

    if args.limit:
        queries = queries[: args.limit]

    chunks = runner.load_retrievers(args.strategy)["bm25"].chunks
    chunks_by_id = {chunk.chunk_id: chunk.to_dict() for chunk in chunks}

    creds = provider_credentials_available(args.provider)
    print(f"approved queries: {len(queries)} · provider credentials: {'present' if creds else 'missing'}")

    records = []
    statuses = {}
    for index, query in enumerate(queries, start=1):
        rows = runner.run_config(
            args.strategy, args.config, [query], top_k=args.context_k
        )
        ranked = rows[0]["ranked"]
        generation = generate_answer(
            query["query"],
            ranked,
            chunks_by_id,
            provider=args.provider,
            model=args.model or None,
        )
        record = build_answer_record(
            query,
            ranked,
            chunks_by_id,
            answer=generation.get("answer"),
            experiment_id=f"gen-{args.strategy}-{args.config}",
            provider=generation.get("provider", args.provider),
            model=generation.get("model"),
            latency_ms=generation.get("latency_ms"),
            token_usage=generation.get("token_usage"),
        )
        record["generation_status"] = generation["status"]
        if generation["status"] != "ok":
            record["generation_reason"] = generation.get("reason")
        statuses[generation["status"]] = statuses.get(generation["status"], 0) + 1
        records.append(record)
        if index % 10 == 0:
            print(f"  {index}/{len(queries)}")

    write_jsonl(output_dir / "answers.jsonl", records)

    qrels: dict[str, dict[str, int]] = {}
    latest: dict[str, dict] = {}
    for judgment in load_jsonl(Path(args.eval_dir) / "judgments.jsonl"):
        latest[judgment["candidate_id"]] = judgment
    for judgment in latest.values():
        if judgment.get("judgment_status") == "judged" and judgment["chunking_strategy"] == args.strategy:
            qrels.setdefault(judgment["query_id"], {})[judgment["chunk_id"]] = int(
                judgment["relevance_grade"]
            )

    human_store = HumanAnswerEvalStore(Path(args.eval_dir) / "answer_ratings.jsonl")
    evaluation = evaluate_answers(records, qrels, human_store)
    evaluation.update(
        {
            "strategy": args.strategy,
            "retrieval_config": args.config,
            "split": args.split,
            "corpus_version": runner.corpus_version,
            "query_version": runner.query_version,
            "generation_statuses": statuses,
            "provider": args.provider,
            "prompt_version": "rag-prompt-v1",
            "note": (
                "Retrieval accuracy and answer quality are separate measurements. "
                "Answer quality is reported only from human ratings."
            ),
        }
    )
    summary_path.write_text(json.dumps(evaluation, indent=2, default=str) + "\n")

    print()
    print(f"generation statuses : {statuses}")
    print(f"answers generated   : {evaluation['num_generated']}")
    print(f"human ratings       : {evaluation['num_human_rated']}")
    print(f"citation correctness: {evaluation['citation_correctness_mean']}")
    print(f"answer quality      : {evaluation['answer_quality']['status']}")
    print(f"\nWrote {output_dir / 'answers.jsonl'}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
