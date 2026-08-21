# reRankQ

A portfolio-grade hybrid search and RAG system with evaluation and ablations.

## Overview

reRankQ demonstrates retrieval engineering rigor beyond typical "chat with your PDF" projects. It implements:

- **Hybrid retrieval**: BM25 + dense embeddings with reciprocal rank fusion
- **Cross-encoder reranking**: Post-retrieval reranking for improved precision
- **Full evaluation harness**: Recall@k, MRR, nDCG@k metrics
- **Ablation study**: Comparing 4 retrieval method variants
- **End-to-end evaluation**: Faithfulness and answer relevance scoring
- **Production-ready API**: FastAPI service with latency logging

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
├── data/                    # Data storage
│   ├── raw/                # Raw crawled documents
│   ├── processed/          # Chunked documents
│   ├── indexes/            # Search indexes
│   └── eval/               # Evaluation set
├── results/                 # Experiment results
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

Designed and deployed a production-ready API with latency logging, containerized with Docker, and built an interactive Streamlit demo for real-time querying.

## License

MIT
