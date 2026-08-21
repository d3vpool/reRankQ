"""
Main ingestion pipeline for reRankQ.
Orchestrates crawling, chunking, and storage.
"""

import json
import time
from pathlib import Path
from typing import Optional

from src.ingestion.crawler import StripeDocCrawler, load_documents, Document
from src.ingestion.chunker import chunk_documents, Chunk
from src.utils.config import get_config
from src.utils.logging import setup_logging, log_stage, log_metrics, LatencyTracker


class IngestionPipeline:
    """Main ingestion pipeline."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging()
        self.latency_tracker = LatencyTracker()
    
    def run(
        self,
        max_pages: Optional[int] = None,
        chunking_strategy: str = "both",
        force_crawl: bool = False,
    ) -> dict:
        """
        Run the full ingestion pipeline.
        
        Args:
            max_pages: Maximum pages to crawl (None = use config)
            chunking_strategy: "fixed_size", "structure_aware", or "both"
            force_crawl: If True, re-crawl even if data exists
        
        Returns:
            Dictionary with pipeline results
        """
        results = {
            "start_time": time.time(),
            "crawl": {},
            "chunking": {},
            "stats": {},
        }
        
        # Check if we need to crawl
        raw_dir = Path("data/raw")
        docs_file = raw_dir / "documents.jsonl"
        
        if docs_file.exists() and not force_crawl:
            self.logger.info("Loading existing documents...")
            documents = load_documents("data/raw")
            results["crawl"]["documents_loaded"] = len(documents)
        else:
            # Crawl
            self.logger.info("Starting crawl...")
            crawler = StripeDocCrawler()
            
            start_time = self.latency_tracker.start("crawl")
            crawl_result = crawler.crawl(max_pages=max_pages)
            self.latency_tracker.end("crawl", start_time)
            
            # Save documents
            crawler.save_documents("data/raw")
            
            documents = crawl_result.documents
            results["crawl"] = {
                "urls_crawled": crawl_result.urls_crawled,
                "urls_failed": crawl_result.urls_failed,
                "documents": len(documents),
                "duration": crawl_result.duration_seconds,
                "stats": crawl_result.stats,
            }
        
        self.logger.info(f"Working with {len(documents)} documents")
        
        # Chunk documents
        self.logger.info(f"Chunking with strategy: {chunking_strategy}")
        start_time = self.latency_tracker.start("chunking")
        chunk_results = chunk_documents(documents, strategy=chunking_strategy)
        self.latency_tracker.end("chunking", start_time)
        
        results["chunking"] = {}
        for strat, chunks in chunk_results.items():
            results["chunking"][strat] = {
                "num_chunks": len(chunks),
                "avg_tokens": sum(c.token_count for c in chunks) / max(len(chunks), 1),
            }
        
        # Compute overall stats
        all_chunks = []
        for chunks in chunk_results.values():
            all_chunks.extend(chunks)
        
        results["stats"] = {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "strategies_used": list(chunk_results.keys()),
            "latency": self.latency_tracker.get_all_stats(),
        }
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "ingestion_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Ingestion complete. Results saved to {results_file}")
        log_metrics(results["stats"], "ingestion", self.logger)
        
        return results
    
    def get_corpus_stats(self) -> dict:
        """Get statistics about the ingested corpus."""
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        
        stats = {
            "raw": {},
            "processed": {},
        }
        
        # Raw docs stats
        docs_file = raw_dir / "documents.jsonl"
        if docs_file.exists():
            documents = load_documents("data/raw")
            stats["raw"] = {
                "num_documents": len(documents),
                "avg_doc_length": sum(len(d.content) for d in documents) / max(len(documents), 1),
                "sections": list(set(d.section for d in documents)),
            }
        
        # Processed chunks stats
        for strategy in ["fixed_size", "structure_aware"]:
            chunks_file = processed_dir / f"chunks_{strategy}.jsonl"
            if chunks_file.exists():
                stats_file = processed_dir / f"stats_{strategy}.json"
                if stats_file.exists():
                    with open(stats_file) as f:
                        stats["processed"][strategy] = json.load(f)
        
        return stats


def main():
    """Run ingestion pipeline."""
    pipeline = IngestionPipeline()
    results = pipeline.run(max_pages=50)  # Start with small crawl for testing
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
