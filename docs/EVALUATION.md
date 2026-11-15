# Evaluation Methodology — reRankQ

This document describes how reRankQ is evaluated: the experimental matrix, the
metrics, the statistical treatment, latency benchmarking, ablations, RAG answer
evaluation and how to reproduce each artifact.

**Status:** the evaluation *infrastructure* is complete and exercised. The
*scientific* evaluation has not produced results, because no query has been
human-approved and no relevance judgment exists. Every metric-producing command
refuses to run in that state rather than emit placeholder numbers.

---

## 1. Research question

Does hybrid retrieval (BM25 + dense RRF) and cross-encoder reranking improve
ranking quality over either retriever alone, on Stripe API documentation; and
does structure-aware chunking beat fixed-size chunking? At what latency cost?

Answering it requires an approved query set, human graded relevance judgments,
and a held-out split that has never been used for tuning.

## 2. System architecture

```
crawler → documents → chunker (fixed_size | structure_aware) → chunks
                                    │
        ┌───────────────────────────┼────────────────────────────┐
        │                           │                            │
      BM25                       dense (MiniLM, FAISS)      hybrid RRF
        │                           │                            │
        └───────────────┬───────────┴────────────┬───────────────┘
                        │                        │
                candidate pool          cross-encoder reranker
                                                 │
                                              RAG generation
```

Retrieval configurations: `bm25`, `dense`, `hybrid_rrf`, `hybrid_rerank`.
Chunking strategies: `fixed_size`, `structure_aware`. Together: eight primary
experimental configurations.

## 3. Experimental matrix

| Dimension | Values |
|---|---|
| Chunking | `fixed_size`, `structure_aware` |
| Retrieval | `bm25`, `dense`, `hybrid_rrf`, `hybrid_rerank` |
| Split | `dev`, `test` (and `all` for descriptive work) |

Each experiment gets an id derived from a hash of: corpus version, approved
query version, split version, split, chunking strategy, retrieval configuration
and retrieval parameters. Identical inputs reproduce the id; any change produces
a new one. `experiment.json` additionally records the timestamp, model names,
software versions, random seed, hardware snapshot, judgment policy and query
set.

Experiments are written to `results/experiments/<experiment_id>/` and an
existing directory is never overwritten.

## 4. Dataset and versions

* Corpus: `data/eval/corpus_manifest.json`, 30 Stripe documents, version
  `corpus-fde4c9c42bf3`, 144 fixed-size and 222 structure-aware chunks.
* Queries: `data/eval/queries.jsonl` (50 candidates), `query_reviews.jsonl`
  (human decisions), `approved_queries.jsonl` (accepted only).
* Split: `data/eval/split_assignment.json`, group-aware, version
  `split-f4a04d80`, 35 dev / 15 test.
* Judgments: `data/eval/judgments.jsonl`, append-only, graded 0–3, status
  `pending` or `judged`.

Before any run the corpus is re-hashed against the manifest, and approved
queries must match both the corpus version and the split assignment. A mismatch
aborts the run.

## 5. Retrieval metrics

Binary relevance threshold: **grade ≥ 2 is relevant** (grade 1 is marginal, not
relevant). This threshold is used only for binary metrics; graded metrics use
the raw 0–3 grades.

| Metric | Definition |
|---|---|
| Recall@K | judged-relevant chunks retrieved in top K / judged-relevant chunks (pool recall) |
| Precision@K | relevant retrieved in top K / K |
| MRR | 1 / rank of the first judged-relevant chunk |
| MAP | mean of precision at each relevant hit, divided by judged-relevant count |
| nDCG@K | DCG@K / ideal DCG@K, gain `2^grade − 1`, discount `log2(rank + 1)` |

K ∈ {1, 3, 5, 10}. Ideal DCG uses the query's own judged grades, so it is
attainable with the available judgments.

### Judgment gating

* Unjudged candidates are **unknown**, never grade 0.
* Default policy `strict_topk`: a query is evaluated only if every chunk in its
  top-10 is judged. Excluded queries are counted by reason
  (`unjudged_retrieved_chunks`, `no_judged_relevant_chunks`, `incomplete_pool`).
* If no query qualifies, the aggregate is `blocked` with a reason and **no
  metric values are written**.
* Macro averages are taken over evaluated queries only, and the number of
  evaluated queries is stored beside every number.

## 6. Pooling bias and coverage

The candidate pool was built from four systems at depth 3. A relevant chunk that
no system returned cannot be judged, so recall here is **pool recall**, not
corpus recall. `coverage_report.json` records pool size, judged and pending
counts per query and strategy; `scripts/build_deeper_pool.py` builds a top-10 or
top-20 pool additively, preserving the existing pool, provenance and judgments.

Measured on the current data: depth-10 pooling adds 1192 candidates (598 → 1790
pairs), which shows how much the depth-3 pool misses at k=10.

## 7. Split methodology

Group-aware split over connected query groups (shared source document, shared
section, near-duplicate wording), version `split-f4a04d80`. Dev drives all
tuning; test is evaluated once for reporting. Ablations refuse `--split test`.

## 8. Statistical methodology

* Resampling unit: **queries**, not chunks.
* Paired bootstrap over per-query metric differences; percentile CI at 95%,
  5000 iterations, seed 13.
* Two-sided bootstrap tail probability, reported with the interval.
* Effect size: Cohen's d_z for paired differences plus win/loss/tie counts.
* Exact two-sided sign test over non-tied pairs.
* Fewer than 30 paired queries triggers an explicit low-power warning.
* No significance claim is made from a larger aggregate alone.

## 9. Latency methodology

`scripts/run_latency_benchmark.py` measures per-query retrieval latency after:

1. index/model load (reported as `index_init_seconds`),
2. one cold call (`cold_first_call_ms`),
3. three warm-up passes (excluded).

Then 10 measured passes over the query workload. Reported: p50, p95, mean,
standard deviation, measurement count. Hardware is read from `/proc`, never
guessed. Latency is machine-specific; absolute values are not portable.

Measured on this machine (12th Gen Intel i5-12500H, 16 threads, 15.2 GB RAM,
CUDA available; 50-query workload):

| Strategy | Config | p50 | p95 | Cold first call |
|---|---|---|---|---|
| fixed_size | bm25 | 0.16 ms | 0.23 ms | 0.3 ms |
| fixed_size | dense | 2.51 ms | 2.66 ms | 4749.7 ms |
| fixed_size | hybrid_rrf | 2.79 ms | 3.07 ms | 3.0 ms |
| fixed_size | hybrid_rerank | 137.98 ms | 142.78 ms | 2664.9 ms |
| structure_aware | bm25 | 0.30 ms | 0.44 ms | 0.3 ms |
| structure_aware | dense | 3.03 ms | 3.48 ms | 1200.3 ms |
| structure_aware | hybrid_rrf | 3.50 ms | 3.98 ms | 3.4 ms |
| structure_aware | hybrid_rerank | 140.49 ms | 149.58 ms | 128.4 ms |

These are measured retrieval latencies, not accuracy results.

## 10. Ablation methodology

Supported: RRF constant (10/30/60), RRF weights (50/50, 70/30), candidate depth
(10/20/40), rerank depth (6/20).

Not supported, and reported as such: chunk size/overlap (fixed by the frozen
chunk set; changing them needs re-chunking and a new corpus snapshot) and BM25
k1/b (baked into the index build).

`run_ablation` refuses the test split. The selected configuration is evaluated on
test once, through `run_retrieval_experiments.py --split test`.

## 11. RAG answer evaluation

Separate from retrieval. `scripts/run_generation_eval.py`:

1. captures the exact retrieved context (chunk ids, text, URL, rank);
2. generates answers if provider credentials exist, otherwise records
   `not_executed` and continues;
3. computes deterministic citation metrics (valid citation rate, correctness
   against judged-relevant chunks, completeness);
4. summarises **human** answer ratings from `data/eval/answer_ratings.jsonl`
   (`scripts/rate_answers.py`).

Answer quality is never inferred from retrieval metrics, and no LLM score is
labelled human. With no approved queries, the run records `not_executed`.

## 12. Artifact structure

```
results/experiments/<experiment_id>/
    experiment.json            # identity, versions, models, status
    query_results.jsonl        # rankings per query (rank, chunk, doc, score)
    aggregate_metrics.json     # metrics or an explicit blocked status
    latency.json               # per-query latency summary + index init
    coverage_report.json       # pool/judgment coverage
    environment.json           # hardware, python, package versions
    judgments_snapshot.json    # how many judged pairs were used
    statistical_comparisons.json  # after analysis
results/experiments/batches/<batch_id>.json
results/experiments/analysis/<batch_id>_error_analysis.json
results/benchmarks/latency-<timestamp>/latency.json
results/ablations/<strategy>-<timestamp>/ablation.json
results/generation/<run>/answers.jsonl, summary.json
```

## 13. Reproduction

```bash
# 0. environment
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. corpus + indexes + candidate dataset (already present in this repo)
python scripts/build_eval_dataset.py

# 2. human tasks (required before metrics exist)
streamlit run scripts/review_queries.py       # accept queries
streamlit run scripts/judge_relevance.py      # grade 0-3

# 3. approved set + frozen split
python scripts/build_approved_dataset.py

# 4. experiments
python scripts/run_retrieval_experiments.py --split dev
python scripts/run_retrieval_experiments.py --split test   # once, final
python scripts/analyze_experiments.py --latest --metric ndcg@5
python scripts/run_ablation.py --strategy structure_aware
python scripts/run_latency_benchmark.py
python scripts/build_deeper_pool.py --depth 10             # if coverage is short
python scripts/run_generation_eval.py --strategy structure_aware

# 5. reporting
streamlit run scripts/results_app.py
```

Pipeline validation without human data (rankings only, never metrics):

```bash
python scripts/run_retrieval_experiments.py \
    --query-set candidates --descriptive --split all
```

## 14. Environment requirements

Python 3.11+; `numpy`, `pandas`, `scikit-learn`, `scipy`; `sentence-transformers`
(all-MiniLM-L6-v2 and cross-encoder/ms-marco-MiniLM-L-6-v2, cached locally);
`faiss-cpu`; `rank-bm25`; `tiktoken`; `streamlit`. No LLM key is required for
retrieval evaluation, latency, ablations or results browsing. A provider key is
optional and only affects answer generation.

## 15. Known limitations

* No approved queries and no judgments exist yet, so there are **no retrieval
  metrics** in this repository.
* Judgments, when collected, will be single-annotator; no inter-annotator
  agreement.
* Pool depth 3 limits recall to pooled chunks.
* Query set is small (50 candidates, 15 test), so confidence intervals will be
  wide and power low.
* Chunk-size and BM25-parameter ablations are unsupported by the current frozen
  index design.
* Latency numbers are for one machine, one workload and one process.
* Generation quality requires human ratings; none exist.

## 16. Engineering vs scientific completion

| Item | State |
|---|---|
| Eight configurations executable | yes (descriptive run, batch `20261115T130150Z`) |
| Metrics implemented and unit-tested | yes |
| Judgment gating and coverage reporting | yes |
| Statistical comparisons | implemented; not executed (no metrics) |
| Latency benchmarking | yes, measured |
| Error analysis | implemented; not executed (needs judgments) |
| Ablation framework | implemented; metrics blocked |
| Generation evaluation | implemented; `not_executed` (no approved queries/keys) |
| Artifacts versioned and traceable | yes |
| Real performance claims | **none** — requires human review and judgments |
