"""
Ablation study comparing all retrieval variants.
Runs BM25, dense, hybrid, and hybrid+reranking, producing results table.
"""

import json
import time
from pathlib import Path
from typing import Optional

from src.evaluation.eval_set import EvalQuery, EvalSetManager
from src.evaluation.metrics import MetricsCalculator, RetrievalMetrics
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker
from src.ingestion.chunker import Chunk, load_chunks
from src.utils.config import get_config
from src.utils.logging import setup_logging, log_metrics, LatencyTracker


class AblationStudy:
    """Run ablation study comparing retrieval methods."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging()
        self.latency_tracker = LatencyTracker()
        self.metrics_calculator = MetricsCalculator()
    
    def run_ablation(
        self,
        eval_queries: list[EvalQuery],
        chunks: list[Chunk],
        chunking_strategy: str = "fixed_size",
        output_dir: str = "results",
        skip_dense: bool = False,
        skip_reranking: bool = False,
    ) -> dict:
        """
        Run full ablation study.
        
        Args:
            eval_queries: Evaluation queries with relevance labels
            chunks: Chunks to index and search
            chunking_strategy: Which chunking strategy to use
            output_dir: Directory to save results
            skip_dense: Skip dense retrieval (for testing without sentence-transformers)
            skip_reranking: Skip reranking (for testing without cross-encoder)
        
        Returns:
            Dictionary with all metrics and timing
        """
        self.logger.info("Starting ablation study...")
        self.logger.info(f"Methods: BM25, Dense, Hybrid, Hybrid+Rerank")
        self.logger.info(f"Chunking: {chunking_strategy}")
        self.logger.info(f"Queries: {len(eval_queries)}")
        
        results = {
            "chunking_strategy": chunking_strategy,
            "num_queries": len(eval_queries),
            "num_chunks": len(chunks),
            "methods": {},
            "timing": {},
            "comparison": None,
        }
        
        # Extract queries
        queries = [q.query for q in eval_queries]
        
        all_metrics: list[RetrievalMetrics] = []
        
        # 1. BM25 Only
        self.logger.info("\n=== Method 1: BM25 Only ===")
        bm25_retriever = BM25Retriever()
        
        start_time = self.latency_tracker.start("bm25_index")
        bm25_retriever.index_chunks(chunks)
        self.latency_tracker.end("bm25_index", start_time)
        
        start_time = self.latency_tracker.start("bm25_retrieve")
        bm25_results = bm25_retriever.batch_retrieve(queries, top_k=10)
        self.latency_tracker.end("bm25_retrieve", start_time)
        
        bm25_metrics = self.metrics_calculator.calculate_metrics(
            eval_queries, bm25_results, "bm25_only", chunking_strategy
        )
        all_metrics.append(bm25_metrics)
        results["methods"]["bm25_only"] = self._format_metrics(bm25_metrics)
        
        self.logger.info(f"BM25 MRR: {bm25_metrics.mrr:.4f}")
        self.logger.info(f"BM25 Recall@5: {bm25_metrics.recall_at_k.get(5, 0):.4f}")
        
        # 2. Dense Only (optional)
        if not skip_dense:
            self.logger.info("\n=== Method 2: Dense Only ===")
            dense_retriever = DenseRetriever()
            
            start_time = self.latency_tracker.start("dense_index")
            dense_retriever.index_chunks(chunks)
            self.latency_tracker.end("dense_index", start_time)
            
            start_time = self.latency_tracker.start("dense_retrieve")
            dense_results = dense_retriever.batch_retrieve(queries, top_k=10)
            self.latency_tracker.end("dense_retrieve", start_time)
            
            dense_metrics = self.metrics_calculator.calculate_metrics(
                eval_queries, dense_results, "dense_only", chunking_strategy
            )
            all_metrics.append(dense_metrics)
            results["methods"]["dense_only"] = self._format_metrics(dense_metrics)
            
            self.logger.info(f"Dense MRR: {dense_metrics.mrr:.4f}")
            self.logger.info(f"Dense Recall@5: {dense_metrics.recall_at_k.get(5, 0):.4f}")
        else:
            self.logger.info("\n=== Method 2: Dense Only (SKIPPED) ===")
            results["methods"]["dense_only"] = {"status": "skipped"}
        
        # 3. Hybrid (BM25 + Dense with RRF)
        if not skip_dense:
            self.logger.info("\n=== Method 3: Hybrid (BM25 + Dense) ===")
            hybrid_retriever = HybridRetriever(bm25_retriever, dense_retriever)
            
            start_time = self.latency_tracker.start("hybrid_retrieve")
            hybrid_results = hybrid_retriever.batch_retrieve(queries, top_k=10)
            self.latency_tracker.end("hybrid_retrieve", start_time)
            
            hybrid_metrics = self.metrics_calculator.calculate_metrics(
                eval_queries, hybrid_results, "hybrid", chunking_strategy
            )
            all_metrics.append(hybrid_metrics)
            results["methods"]["hybrid"] = self._format_metrics(hybrid_metrics)
            
            self.logger.info(f"Hybrid MRR: {hybrid_metrics.mrr:.4f}")
            self.logger.info(f"Hybrid Recall@5: {hybrid_metrics.recall_at_k.get(5, 0):.4f}")
        else:
            self.logger.info("\n=== Method 3: Hybrid (SKIPPED - needs dense) ===")
            results["methods"]["hybrid"] = {"status": "skipped"}
        
        # 4. Hybrid + Cross-Encoder Reranking
        if not skip_dense and not skip_reranking:
            self.logger.info("\n=== Method 4: Hybrid + Reranking ===")
            reranker = CrossEncoderReranker()
            
            start_time = self.latency_tracker.start("rerank")
            reranked_results = []
            for query, hybrid_result in zip(queries, hybrid_results):
                reranked = reranker.rerank(query, hybrid_result, top_k=10)
                reranked_results.append(reranked)
            self.latency_tracker.end("rerank", start_time)
            
            rerank_metrics = self.metrics_calculator.calculate_metrics(
                eval_queries, reranked_results, "hybrid_rerank", chunking_strategy
            )
            all_metrics.append(rerank_metrics)
            results["methods"]["hybrid_rerank"] = self._format_metrics(rerank_metrics)
            
            self.logger.info(f"Hybrid+Rerank MRR: {rerank_metrics.mrr:.4f}")
            self.logger.info(f"Hybrid+Rerank Recall@5: {rerank_metrics.recall_at_k.get(5, 0):.4f}")
        else:
            self.logger.info("\n=== Method 4: Hybrid + Reranking (SKIPPED) ===")
            results["methods"]["hybrid_rerank"] = {"status": "skipped"}
        
        # Compare methods
        if len(all_metrics) > 1:
            comparison = self.metrics_calculator.compare_methods(all_metrics)
            results["comparison"] = comparison
        
        # Save timing
        results["timing"] = self.latency_tracker.get_all_stats()
        
        # Save results
        self._save_results(results, output_dir)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _format_metrics(self, metrics: RetrievalMetrics) -> dict:
        """Format metrics for storage."""
        return {
            "method": metrics.method,
            "num_queries": metrics.num_queries,
            "mrr": metrics.mrr,
            "recall_at_k": metrics.recall_at_k,
            "ndcg_at_k": metrics.ndcg_at_k,
        }
    
    def _save_results(self, results: dict, output_dir: str) -> None:
        """Save ablation results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        results_file = output_path / "ablation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save summary table as CSV
        self._save_results_table(results, output_path / "ablation_table.csv")
        
        self.logger.info(f"Saved ablation results to {output_path}")
    
    def _save_results_table(self, results: dict, filepath: Path) -> None:
        """Save results as a formatted table."""
        lines = ["Method,MRR,Recall@1,Recall@3,Recall@5,Recall@10,nDCG@5"]
        
        for method_name, method_data in results["methods"].items():
            if method_data.get("status") == "skipped":
                lines.append(f"{method_name},skipped,skipped,skipped,skipped,skipped,skipped")
                continue
            
            mrr = method_data["mrr"]
            recall = method_data["recall_at_k"]
            ndcg = method_data["ndcg_at_k"]
            
            line = f"{method_name},{mrr:.4f}"
            for k in [1, 3, 5, 10]:
                line += f",{recall.get(k, 0):.4f}"
            line += f",{ndcg.get(5, 0):.4f}"
            lines.append(line)
        
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
    
    def _print_summary(self, results: dict) -> None:
        """Print summary table to console."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)
        print(f"Chunking: {results['chunking_strategy']}")
        print(f"Queries: {results['num_queries']}")
        print(f"Chunks: {results['num_chunks']}")
        print("-" * 80)
        
        header = f"{'Method':<20} {'MRR':>8} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'nDCG@5':>8}"
        print(header)
        print("-" * 80)
        
        for method_name, method_data in results["methods"].items():
            if method_data.get("status") == "skipped":
                print(f"{method_name:<20} {'skipped':>8}")
                continue
            
            mrr = method_data["mrr"]
            recall = method_data["recall_at_k"]
            ndcg = method_data["ndcg_at_k"]
            
            row = f"{method_name:<20} {mrr:>8.4f}"
            for k in [1, 3, 5, 10]:
                row += f" {recall.get(k, 0):>8.4f}"
            row += f" {ndcg.get(5, 0):>8.4f}"
            print(row)
        
        print("=" * 80)
        
        # Print timing
        if results.get("timing"):
            print("\nTiming:")
            for stage, stats in results["timing"].items():
                print(f"  {stage}: {stats['mean']:.2f}s mean ({stats['count']} runs)")


def run_ablation_study(
    chunking_strategy: str = "fixed_size",
    num_queries: int = 50,
    output_dir: str = "results",
    skip_dense: bool = False,
    skip_reranking: bool = False,
) -> dict:
    """Run the ablation study end-to-end."""
    study = AblationStudy()
    
    # Load or create eval set
    eval_manager = EvalSetManager()
    try:
        eval_queries = eval_manager.load_eval_set()
    except FileNotFoundError:
        # Create eval set
        from src.ingestion.crawler import load_documents
        documents = load_documents("data/raw")
        chunks = load_chunks("data/processed", strategy=chunking_strategy)
        eval_queries = eval_manager.create_eval_set(documents, chunks, num_queries)
    
    # Load chunks
    chunks = load_chunks("data/processed", strategy=chunking_strategy)
    
    # Run ablation
    return study.run_ablation(
        eval_queries,
        chunks,
        chunking_strategy,
        output_dir,
        skip_dense=skip_dense,
        skip_reranking=skip_reranking,
    )


if __name__ == "__main__":
    # Run ablation with BM25 only (no sentence-transformers needed)
    results = run_ablation_study(
        chunking_strategy="fixed_size",
        skip_dense=True,
        skip_reranking=True,
    )
