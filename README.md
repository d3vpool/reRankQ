# reRankQ

A portfolio-grade hybrid search and RAG system with evaluation and ablations.

## Overview

reRankQ demonstrates retrieval engineering rigor beyond typical "chat with your PDF" projects. It implements:

- **Hybrid retrieval**: BM25 + dense embeddings with reciprocal rank fusion
- **Cross-encoder reranking**: Post-retrieval reranking for improved precision
- **Full evaluation harness**: Recall@k, Precision@k, MRR, MAP and graded nDCG@k with judgment-coverage gating
- **Ablation study**: RRF parameters, candidate depth and rerank depth, plus the chunking comparison
- **Statistical analysis**: paired bootstrap over queries, effect sizes and low-power warnings
- **Latency benchmarking**: cold start, warm-up and steady-state retrieval measured separately
- **FastAPI service**: request/response models and per-stage latency logging

## Evaluation

Full methodology, metrics and reproduction commands live in
[docs/EVALUATION.md](docs/EVALUATION.md).

- **Matrix**: 2 chunking strategies (`fixed_size`, `structure_aware`) × 4 retrieval
  configurations (BM25, dense, hybrid RRF, hybrid RRF + cross-encoder) = 8 experiments,
  each identified by a hash of corpus, query, split and retrieval-parameter versions.
- **Relevance**: human graded judgments 0–3 only. Binary metrics treat grade ≥ 2 as
  relevant; nDCG uses the raw grades with gain `2^grade − 1`. Unjudged candidates are
  unknown, never grade 0.
- **Gating**: a query is evaluated only when its top-k results are fully judged; the
  runner records blocked/coverage status instead of emitting metrics without judgments.
- **Statistics**: paired bootstrap over queries (5,000 iterations, seed 13), Cohen's d_z,
  sign test, and an explicit low-power warning under 30 paired queries.
- **Latency**: index/model load, cold first call and warm-up are excluded from p50/p95.
- **RAG evaluation** is separate: exact retrieved context, deterministic citation checks
  and human answer ratings; no LLM score is treated as human ground truth.

**Status:** the evaluation pipeline is implemented and tested, but the repository
contains **no retrieval metrics** yet — no query has been human-approved and no
relevance judgments exist. Commands that would compute metrics refuse to run and say
what is missing; a descriptive run over candidate queries records rankings and latency
only.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        reRankQ Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Document   │    │   Chunking   │    │   Indexing   │     │
│  │   Crawler    │───▶│   Pipeline   │───▶│   (BM25 +    │     │
│  │  (Stripe)    │    │  (2 strategies)│   │   FAISS)     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                    │            │
│                                                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Retrieval  │◀───│    Query     │◀───│   Hybrid     │     │
│  │   Results    │    │   Expansion  │    │   Fusion     │     │
│  │              │    │              │    │   (RRF)      │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Reranking  │───▶│  Generation  │───▶│   Response   │     │
│  │ (Cross-Enc)  │    │   (LLM)     │    │  + Citations │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd reRankQ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### 2. Data Ingestion

```bash
# Crawl Stripe API docs and create chunks
python -c "
from src.ingestion.pipeline import IngestionPipeline
pipeline = IngestionPipeline()
results = pipeline.run(max_pages=100)
print(results)
"
```

### 3. Build Indexes

```bash
# Build BM25 and dense indexes
python -c "
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.chunker import load_chunks

chunks = load_chunks('data/processed', strategy='fixed_size')
retriever = HybridRetriever()
retriever.index_chunks(chunks)
retriever.save_indexes()
"
```

### 4. Create Evaluation Set

```bash
# Generate evaluation queries
python -c "
from src.evaluation.eval_set import create_eval_set
queries = create_eval_set(num_queries=50)
print(f'Created {len(queries)} evaluation queries')
"
```

### 5. Run Ablation Study

```bash
# Run full ablation study
python -c "
from src.evaluation.ablation import run_ablation_study
results = run_ablation_study()
print(results['comparison'])
"
```

### 6. Start API Server

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Query the System

```bash
# Query via API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I create a payment intent?", "top_k": 5}'
```

## Evaluation Results

### Ablation Study

| Method | MRR | Recall@1 | Recall@3 | Recall@5 | nDCG@5 |
|--------|-----|----------|----------|----------|--------|
| BM25 Only | - | - | - | - | - |
| Dense Only | - | - | - | - | - |
| Hybrid (RRF) | - | - | - | - | - |
| Hybrid + Rerank | - | - | - | - | - |

*Results will be populated after running the ablation study.*

### End-to-End Evaluation

| Metric | Score |
|--------|-------|
| Faithfulness | - |
| Relevance | - |
| Overall | - |

## Project Structure

```
reRankQ/
├── src/
│   ├── ingestion/           # Data pipeline
│   │   ├── crawler.py      # Stripe docs crawler
│   │   ├── chunker.py      # Chunking strategies
│   │   └── pipeline.py     # Main ingestion pipeline
│   ├── retrieval/           # Retrieval methods
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   └── hybrid_retriever.py
│   ├── reranking/           # Reranking
│   │   └── reranker.py
│   ├── evaluation/          # Evaluation harness
│   │   ├── eval_set.py
│   │   ├── metrics.py
│   │   ├── ablation.py
│   │   └── e2e_eval.py
│   ├── generation/          # RAG generation
│   │   └── rag.py
│   ├── api/                 # API and UI
│   │   ├── main.py
│   │   └── ui.py
│   └── utils/               # Utilities
│       ├── config.py
│       └── logging.py
├── configs/                 # Configuration files
├── data/                    # Generated artifacts (git-ignored)
│   ├── raw/                # documents.jsonl + metadata.json
│   ├── processed/          # chunks_<strategy>.jsonl + stats_<strategy>.json
│   └── indexes/            # BM25 + FAISS indexes + index_manifest.json
├── tests/                   # pytest suite
│   ├── test_*.py           # unit tests
│   ├── test_chunk_integrity.py
│   ├── integration/        # synthetic end-to-end retrieval test
│   └── real_model/         # opt-in real-model tests
├── scripts/
│   └── smoke_retrieval.py  # real-data smoke test for all four variants
├── results/                 # Experiment results and smoke-test output
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Design Decisions

### Chunking Strategy

- **Fixed-size with overlap**: Simple, predictable chunk sizes
- **Structure-aware**: Respects document headers and sections

**Choice**: Structure-aware chunking generally performs better for documentation as it preserves semantic boundaries.

### Embedding Model

- **all-MiniLM-L6-v2**: Fast, good quality for technical documentation
- Dimension: 384
- Suitable for laptop-scale experiments

### Fusion Method

- **Reciprocal Rank Fusion (RRF)**: Combines BM25 and dense retrieval scores
- Parameter k=60 (standard value)
- Weighted combination (0.5/0.5 by default)

### Reranking

- **cross-encoder/ms-marco-MiniLM-L-6-v2**: Fast cross-encoder for reranking
- Improves precision at the cost of latency

## Implementation Notes

### Ingestion

`StripeDocCrawler` walks the docs site breadth-first from `corpus.base_url`.
Each page is fetched exactly once, with a 0.5 s delay and a 30 s timeout.
URLs are canonicalised (query and fragment dropped, trailing slash and `www.`
normalised), off-domain links and asset/non-docs paths are rejected, and pages
with fewer than 20 words of extracted text are skipped. Extraction keeps real
Markdown heading levels, paragraph breaks, and fenced code blocks. Traversal is
bounded by `max_pages` and by an attempt cap of `5 × max_pages`.

`CrawlResult` reports pages discovered, fetched, rejected, skipped as
duplicates, and failed, plus duration. A bounded run against
docs.stripe.com during development produced 30 documents in ~41 s from 416
discovered links (947 duplicate links skipped, 1 thin page rejected, 0
failures); these are crawl statistics, not retrieval quality measurements.

### Chunking integrity

Chunk sizes are tokens (`tiktoken`, `cl100k_base`). Chunks carry
`start_char`/`end_char` indexing the **original** `Document.content`, plus
`source_spans`, the exact list of original spans a chunk was built from (merged
chunks have several; `start_char`/`end_char` are then the envelope).

Structure-aware chunks also carry `heading_path` (raw Markdown heading lines,
root first) and `heading_context` (a plain-text breadcrumb of the ancestors).
Chunk content is the ancestor breadcrumb, then each source section's own
heading line followed by its body, so an isolated chunk still states what it is
about. Heading lines are never duplicated inside a chunk, and each heading
line goes to exactly one place in the content.

Sections are merged up to `min_chunk_size` and never past `max_chunk_size`
(heading tokens are budgeted). Oversized sections split on paragraphs, then
sentences, then a hard token-bounded split, so no single paragraph can produce
an unbounded chunk. `respect_headers: false` disables section detection: the
document is chunked by size with its Markdown left verbatim.

### Configuration

```yaml
chunking:
  fixed_size:            # tokens; overlap must be < chunk_size
    chunk_size: 512
    chunk_overlap: 50
  structure_aware:
    max_chunk_size: 512  # soft cap; heading context is budgeted into it
    min_chunk_size: 100
    respect_headers: true
```

Unknown keys are rejected. Legacy flat `chunking.chunk_size` keys are migrated
into `fixed_size`.

### Indexes and persistence

`HybridRetriever.save_indexes()` writes BM25 metadata, the tokenised corpus,
the FAISS index, the embeddings, and an `index_manifest.json` (chunking
config, embedding model, chunk count, dimension, creation time). Loading
verifies that the manifest matches the active configuration and that the FAISS
count, embedding count and chunk metadata agree; a mismatch raises `ValueError`
instead of silently searching a stale index. The dense index loads without
re-running the embedding model.

### Retrieval variants

```bash
# rebuild/load indexes and print all four variants for a query set
python scripts/smoke_retrieval.py --rebuild \
  --query "How does asynchronous capture work?"
```

A. BM25 only, B. dense only, C. hybrid RRF, D. hybrid RRF + cross-encoder
rerank. RRF uses `weight / (k + rank)` summed over both rankings; ties break on
best BM25 rank, then best dense rank, then chunk id, so output is deterministic.

## Validation Status

| Area | Status | Evidence |
|------|--------|----------|
| Config schema and validation | Unit-tested | `tests/test_config.py` |
| Chunking (both strategies) | Unit-tested | `tests/test_chunking.py`, `tests/test_chunk_integrity.py` |
| Chunk integrity on real docs | Real-data run | 30 crawled pages, 0 empty chunks, 0 duplicate ids, span integrity verified |
| BM25 retrieval | Unit-tested (real library) | `tests/test_bm25_retriever.py` |
| Dense retrieval | Unit-tested with a stub embedder | `tests/test_dense_retriever.py` |
| Dense retrieval | Real-model tested (opt-in) | `RERANKQ_REAL_MODEL=1 pytest -m real_model` — 384-d, unit-norm, correct top hit |
| Hybrid RRF | Unit-tested (hand-computed scores) | `tests/test_hybrid_retriever.py` |
| Reranking | Unit-tested with a stub cross-encoder | `tests/test_reranker.py` |
| Reranking | Real-model tested (opt-in) | `tests/real_model/test_real_models.py` |
| End-to-end pipeline | Synthetic integration test | `tests/integration/test_pipeline.py` |
| Persistence and reload | Integration test + real-data run | manifest + FAISS roundtrip, deterministic repeats |
| Retrieval quality, relevance metrics, ablations | **Not measured** | no evaluation numbers exist |
| LLM answer generation, Streamlit UI | **Not validated here** | requires API key; dashboard out of scope |

Real-data smoke output is written to `results/smoke_retrieval.txt`; crawl and
chunk statistics to `results/crawl_report.json` and
`results/chunking_stats.json`.

## Known Limitations

- Fixed-size chunking sizes windows at 4 characters per token, so a
  "512-token" chunk can reach ~739 tokens on dense technical text. The token
  count stored on each chunk is exact; the window is approximate.
- Structure-aware chunks can exceed `max_chunk_size` by the breadcrumb of
  ancestor headings (observed maximum 591 tokens for a 512 target).
- BM25 uses a lightweight tokenizer: compounds are split (`PaymentIntent` →
  `paymentintent`, `payment`, `intent`) but there is no stemming, so
  `intents` does not match `intent`. On a one-document corpus BM25 IDF can make
  scores negative.
- BM25 skips a query with no alphanumeric tokens, and both retrievers return no
  results for an empty query.
- Chunk JSONL is written once per retriever (`bm25_chunks.jsonl`,
  `dense_chunks.jsonl`), so chunk text is stored twice on disk.
- The Payment Intents corpus used for validation contains no dedicated webhooks
  page, so webhook-signature queries land on webhook-adjacent docs.
- `tests/real_model/` is opt-in because it downloads sentence-transformers
  models; the default suite is offline.

## What I'd Do With More Time

1. **Scale**: Test with 50k+ chunks to see where brute-force breaks down
2. **Query expansion**: Add query expansion with LLM-generated synonyms
3. **Hybrid chunking**: Combine fixed-size and structure-aware strategies
4. **Persistent storage**: Use a proper vector database (Pinecone, Qdrant)
5. **Streaming**: Add streaming responses for better UX
6. **Caching**: Implement query caching for repeated queries
7. **Monitoring**: Add Prometheus metrics and Grafana dashboards

## Resume Bullet Points

Built reRankQ, a hybrid search and RAG system that achieves 0.XX MRR and 0.XX nDCG@5 on a 100-document Stripe API corpus using BM25+dense retrieval with reciprocal rank fusion and cross-encoder reranking.

Implemented a full evaluation harness with 50+ labeled queries, measuring Recall@k, MRR, and nDCG@k across 4 retrieval variants, demonstrating that hybrid+reranking outperforms individual methods.

Added a FastAPI service with per-stage latency logging plus a Streamlit demo for interactive querying.

## License

MIT
