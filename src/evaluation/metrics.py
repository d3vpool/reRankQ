"""
Retrieval evaluation metrics.
Implements Recall@k, MRR, and nDCG@k for ablation study.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.evaluation.eval_set import EvalQuery, EvalSetManager
from src.utils.config import get_config
from src.utils.logging import setup_logging, log_metrics


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval run."""
    method: str
    chunking_strategy: str
    k_values: list[int]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    num_queries: int
    per_query_metrics: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


class MetricsCalculator:
    """Calculate retrieval evaluation metrics."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging()
    
    def recall_at_k(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """
        Calculate Recall@k.
        
        Proportion of relevant items retrieved in top-k results.
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        intersection = retrieved_at_k.intersection(relevant_set)
        return len(intersection) / len(relevant_set)
    
    def mean_reciprocal_rank(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        1/rank of first relevant item.
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """
        Calculate nDCG@k (Normalized Discounted Cumulative Gain).
        
        Measures ranking quality considering position.
        """
        relevant_set = set(relevant_ids)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate ideal DCG
        ideal_dcg = 0.0
        num_relevant = min(len(relevant_set), k)
        for i in range(num_relevant):
            ideal_dcg += 1.0 / math.log2(i + 2)
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg
    
    def calculate_metrics(
        self,
        queries: list[EvalQuery],
        retrieval_results: list[list[dict]],
        method: str,
        chunking_strategy: str,
        k_values: Optional[list[int]] = None,
    ) -> RetrievalMetrics:
        """
        Calculate all metrics for a retrieval run.
        
        Args:
            queries: Evaluation queries with relevance labels
            retrieval_results: List of retrieval results for each query
            method: Retrieval method name
            chunking_strategy: Chunking strategy used
            k_values: List of k values to evaluate at
        
        Returns:
            RetrievalMetrics with all calculated metrics
        """
        if k_values is None:
            k_values = self.config.evaluation.k_values
        
        # Track per-query metrics
        per_query_metrics = []
        
        # Track metrics across queries
        recall_values = {k: [] for k in k_values}
        mrr_values = []
        ndcg_values = {k: [] for k in k_values}
        
        for query, results in zip(queries, retrieval_results):
            # Extract retrieved chunk IDs
            retrieved_ids = [r["chunk_id"] for r in results]
            relevant_ids = query.relevant_chunk_ids
            
            # Calculate metrics
            query_metrics = {
                "query_id": query.query_id,
                "query": query.query[:50],
            }
            
            for k in k_values:
                recall = self.recall_at_k(retrieved_ids, relevant_ids, k)
                ndcg = self.ndcg_at_k(retrieved_ids, relevant_ids, k)
                
                recall_values[k].append(recall)
                ndcg_values[k].append(ndcg)
                
                query_metrics[f"recall@{k}"] = recall
                query_metrics[f"ndcg@{k}"] = ndcg
            
            mrr = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            mrr_values.append(mrr)
            query_metrics["mrr"] = mrr
            
            per_query_metrics.append(query_metrics)
        
        # Calculate averages
        avg_recall = {k: np.mean(v) for k, v in recall_values.items()}
        avg_mrr = np.mean(mrr_values)
        avg_ndcg = {k: np.mean(v) for k, v in ndcg_values.items()}
        
        metrics = RetrievalMetrics(
            method=method,
            chunking_strategy=chunking_strategy,
            k_values=k_values,
            recall_at_k=avg_recall,
            mrr=avg_mrr,
            ndcg_at_k=avg_ndcg,
            num_queries=len(queries),
            per_query_metrics=per_query_metrics,
        )
        
        return metrics
    
    def save_metrics(
        self,
        metrics: RetrievalMetrics,
        output_dir: str = "results",
    ) -> Path:
        """Save metrics to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"metrics_{metrics.method}_{metrics.chunking_strategy}.json"
        filepath = output_path / filename
        
        # Save metrics
        with open(filepath, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved metrics to {filepath}")
        return filepath
    
    def load_metrics(self, filepath: str) -> RetrievalMetrics:
        """Load metrics from disk."""
        with open(filepath) as f:
            data = json.load(f)
        
        return RetrievalMetrics(**data)
    
    def compare_methods(
        self,
        metrics_list: list[RetrievalMetrics],
    ) -> dict:
        """
        Compare multiple retrieval methods.
        
        Returns a summary table for ablation study.
        """
        comparison = {
            "methods": [],
            "summary": {},
        }
        
        for metrics in metrics_list:
            method_summary = {
                "method": metrics.method,
                "chunking": metrics.chunking_strategy,
                "num_queries": metrics.num_queries,
                "mrr": metrics.mrr,
            }
            
            for k in metrics.k_values:
                method_summary[f"recall@{k}"] = metrics.recall_at_k.get(k, 0.0)
                method_summary[f"ndcg@{k}"] = metrics.ndcg_at_k.get(k, 0.0)
            
            comparison["methods"].append(method_summary)
        
        # Create summary table
        comparison["summary"] = {
            "best_mrr": max(metrics_list, key=lambda m: m.mrr).method,
            "best_recall_5": max(metrics_list, key=lambda m: m.recall_at_k.get(5, 0.0)).method,
            "best_ndcg_5": max(metrics_list, key=lambda m: m.ndcg_at_k.get(5, 0.0)).method,
        }
        
        return comparison


def calculate_retrieval_metrics(
    queries: list[EvalQuery],
    retrieval_results: list[list[dict]],
    method: str,
    chunking_strategy: str,
    output_dir: str = "results",
) -> RetrievalMetrics:
    """Calculate and save retrieval metrics."""
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(
        queries, retrieval_results, method, chunking_strategy
    )
    calculator.save_metrics(metrics, output_dir)
    return metrics
