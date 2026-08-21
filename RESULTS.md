# reRankQ Results

## Overview

This document contains the results of the ablation study and end-to-end evaluation for the reRankQ hybrid search and RAG system.

## Corpus Statistics

- **Source**: Stripe API Documentation
- **Documents Crawled**: 100
- **Total Chunks**: ~1,000 (fixed-size: 409, structure-aware: 630)
- **Evaluation Queries**: 50

## Ablation Study Results

### Retrieval Metrics

| Method | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@5 |
|--------|-----|----------|----------|----------|-----------|--------|
| BM25 Only | - | - | - | - | - | - |
| Dense Only | - | - | - | - | - | - |
| Hybrid (RRF) | - | - | - | - | - | - |
| Hybrid + Rerank | - | - | - | - | - | - |

*Note: Results will be populated after running the full evaluation.*

### Timing Analysis

| Stage | Mean Time | P95 Time |
|-------|-----------|----------|
| BM25 Index | - | - |
| BM25 Retrieve | - | - |
| Dense Index | - | - |
| Dense Retrieve | - | - |
| Hybrid Retrieve | - | - |
| Reranking | - | - |

## End-to-End Evaluation

### Faithfulness (LLM-as-Judge)

- **Mean Score**: -
- **Score Distribution**: -
- **Interpretation**: -

### Answer Relevance (LLM-as-Judge)

- **Mean Score**: -
- **Score Distribution**: -
- **Interpretation**: -

## Failure Case Analysis

### Case 1: [Query]

**Query**: 
**Expected**: 
**Retrieved**: 
**Generated Answer**: 
**Issue**: 
**Root Cause**: 

### Case 2: [Query]

**Query**: 
**Expected**: 
**Retrieved**: 
**Generated Answer**: 
**Issue**: 
**Root Cause**: 

### Case 3: [Query]

**Query**: 
**Expected**: 
**Retrieved**: 
**Generated Answer**: 
**Issue**: 
**Root Cause**: 

## Interpretation

### Key Findings

1. **Hybrid vs. Individual Methods**: 
2. **Impact of Reranking**: 
3. **Chunking Strategy Impact**: 

### Design Decisions Justification

1. **Chunk Size (512 tokens)**: 
2. **Overlap (50 tokens)**: 
3. **Embedding Model (all-MiniLM-L6-v2)**: 
4. **RRF Constant (k=60)**: 

## Resume-Ready Summary

Built reRankQ, a hybrid search and RAG system that:

- **Implements** BM25 + dense retrieval with reciprocal rank fusion
- **Achieves** improved retrieval metrics through cross-encoder reranking
- **Demonstrates** rigorous evaluation with 50+ labeled queries and multiple metrics
- **Deploys** a production-ready API with latency logging

### Key Metrics (To Be Filled)

- MRR: X.XX (vs. baseline X.XX)
- Recall@5: X.XX (X% improvement over BM25 alone)
- nDCG@5: X.XX
- End-to-end latency: XXms (retrieval: XXms, rerank: XXms, generation: XXms)

## Future Work

1. **Scale Testing**: Test with larger corpora (10k-50k chunks)
2. **Advanced Chunking**: Implement semantic chunking with sentence boundaries
3. **Query Expansion**: Add LLM-based query expansion
4. **Persistent Storage**: Migrate to production vector database
5. **Streaming**: Implement streaming responses for better UX
6. **Caching**: Add query result caching

## Appendix: Raw Results

### Detailed Per-Query Metrics

*Available in `results/metrics_*.json` files*

### Evaluation Query Set

*Available in `data/eval/eval_queries.jsonl`*

### Configuration

*Available in `configs/default.yaml`*
