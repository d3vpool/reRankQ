"""
Structured logging for reRankQ.
Provides consistent logging across all pipeline stages.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True
) -> logging.Logger:
    """Set up logging for the application."""
    logger = logging.getLogger("rerankq")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    return logger


@contextmanager
def log_stage(stage_name: str, logger: Optional[logging.Logger] = None):
    """Context manager to log pipeline stage timing and status."""
    if logger is None:
        logger = logging.getLogger("rerankq")
    
    logger.info(f"Starting stage: {stage_name}")
    start_time = time.time()
    
    try:
        yield
        elapsed = time.time() - start_time
        logger.info(f"Completed stage: {stage_name} in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed stage: {stage_name} after {elapsed:.2f}s: {e}")
        raise


def log_metrics(
    metrics: dict[str, Any],
    stage: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log metrics in a structured format."""
    if logger is None:
        logger = logging.getLogger("rerankq")
    
    log_entry = {
        "type": "metrics",
        "stage": stage,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    logger.info(json.dumps(log_entry))


def log_retrieval_result(
    query: str,
    results: list[dict],
    method: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log retrieval results for debugging."""
    if logger is None:
        logger = logging.getLogger("rerankq")
    
    log_entry = {
        "type": "retrieval_result",
        "method": method,
        "query": query[:100],  # Truncate long queries
        "num_results": len(results),
        "top_result_score": results[0]["score"] if results else None,
    }
    
    logger.info(json.dumps(log_entry))


class LatencyTracker:
    """Track latency for different pipeline stages."""
    
    def __init__(self):
        self.stages: dict[str, list[float]] = {}
    
    def start(self, stage: str) -> float:
        """Start timing a stage."""
        return time.time()
    
    def end(self, stage: str, start_time: float) -> float:
        """End timing and record duration."""
        duration = time.time() - start_time
        if stage not in self.stages:
            self.stages[stage] = []
        self.stages[stage].append(duration)
        return duration
    
    def get_stats(self, stage: str) -> dict[str, float]:
        """Get statistics for a stage."""
        if stage not in self.stages:
            return {}
        
        durations = self.stages[stage]
        return {
            "count": len(durations),
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "total": sum(durations),
        }
    
    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all stages."""
        return {stage: self.get_stats(stage) for stage in self.stages}
    
    def reset(self):
        """Reset all tracked latencies."""
        self.stages.clear()


# Global latency tracker
latency_tracker = LatencyTracker()
