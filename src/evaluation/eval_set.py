"""
Evaluation set creation and management.
Creates labeled (query, relevant_doc_ids) pairs for retrieval evaluation.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.ingestion.crawler import Document, load_documents
from src.ingestion.chunker import Chunk, load_chunks
from src.utils.config import get_config
from src.utils.logging import setup_logging


@dataclass
class EvalQuery:
    """A single evaluation query with relevance labels."""
    query_id: str
    query: str
    relevant_doc_ids: list[str]
    relevant_chunk_ids: list[str]
    category: str  # e.g., "factual", "procedural", "conceptual"
    difficulty: str  # "easy", "medium", "hard"
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


class EvalSetManager:
    """Manages the evaluation set for retrieval."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging()
        self.queries: list[EvalQuery] = []
    
    def create_eval_set(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        num_queries: int = 50,
        output_dir: str = "data/eval",
    ) -> list[EvalQuery]:
        """
        Create evaluation set with labeled queries.
        
        This creates queries that are answerable from the corpus,
        with ground truth relevance labels.
        """
        self.logger.info(f"Creating eval set with {num_queries} queries...")
        
        # Group chunks by document
        doc_chunks: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if chunk.doc_id not in doc_chunks:
                doc_chunks[chunk.doc_id] = []
            doc_chunks[chunk.doc_id].append(chunk)
        
        # Create queries based on document content
        queries = []
        
        # 1. Factual queries (specific information from docs)
        factual_templates = [
            "What is {topic}?",
            "How does {topic} work?",
            "What are the requirements for {topic}?",
            "When should I use {topic}?",
            "What is the difference between {topic_a} and {topic_b}?",
        ]
        
        # 2. Procedural queries (how-to)
        procedural_templates = [
            "How do I set up {feature}?",
            "Steps to configure {feature}",
            "What's the process for {action}?",
            "How can I integrate {feature}?",
        ]
        
        # 3. Conceptual queries (understanding)
        conceptual_templates = [
            "Explain {concept}",
            "What is the purpose of {concept}?",
            "Why would I need {concept}?",
            "What are the benefits of {concept}?",
        ]
        
        # Extract topics from document titles and sections
        topics = []
        for doc in documents:
            # Extract meaningful topics
            title_parts = doc.title.replace(" | Stripe Documentation", "").split(" - ")
            for part in title_parts:
                part = part.strip()
                if len(part) > 3 and part.lower() not in ["stripe", "documentation"]:
                    topics.append(part)
        
        topics = list(set(topics))[:50]  # Limit to 50 topics
        
        # Generate queries
        query_id = 0
        
        for i in range(num_queries):
            # Select random topic
            topic = random.choice(topics)
            
            # Select query template based on category
            if i % 3 == 0:
                template = random.choice(factual_templates)
                category = "factual"
                difficulty = random.choice(["easy", "medium", "hard"])
            elif i % 3 == 1:
                template = random.choice(procedural_templates)
                category = "procedural"
                difficulty = random.choice(["medium", "hard"])
            else:
                template = random.choice(conceptual_templates)
                category = "conceptual"
                difficulty = random.choice(["easy", "medium"])
            
            # Fill template
            if "{topic_a}" in template and "{topic_b}" in template:
                topic_b = random.choice([t for t in topics if t != topic][:10] or ["Billing"])
                query = template.format(topic_a=topic, topic_b=topic_b)
            elif "{topic}" in template:
                query = template.format(topic=topic)
            elif "{feature}" in template:
                query = template.format(feature=topic)
            elif "{action}" in template:
                query = template.format(action=topic.lower())
            elif "{concept}" in template:
                query = template.format(concept=topic)
            else:
                query = f"Tell me about {topic}"
            
            # Find relevant documents and chunks
            relevant_doc_ids = []
            relevant_chunk_ids = []
            
            # Search for relevant docs by title/content match
            topic_lower = topic.lower()
            for doc in documents:
                if topic_lower in doc.title.lower() or topic_lower in doc.content.lower()[:500]:
                    relevant_doc_ids.append(doc.doc_id)
                    
                    # Add chunks from this doc
                    if doc.doc_id in doc_chunks:
                        for chunk in doc_chunks[doc.doc_id][:3]:  # Max 3 chunks per doc
                            relevant_chunk_ids.append(chunk.chunk_id)
            
            # Ensure at least one relevant doc
            if not relevant_doc_ids:
                # Fallback: use random doc
                random_doc = random.choice(documents)
                relevant_doc_ids = [random_doc.doc_id]
                if random_doc.doc_id in doc_chunks:
                    relevant_chunk_ids = [c.chunk_id for c in doc_chunks[random_doc.doc_id][:2]]
            
            # Limit to max relevant items
            relevant_doc_ids = relevant_doc_ids[:3]
            relevant_chunk_ids = relevant_chunk_ids[:5]
            
            eval_query = EvalQuery(
                query_id=f"q{query_id:03d}",
                query=query,
                relevant_doc_ids=relevant_doc_ids,
                relevant_chunk_ids=relevant_chunk_ids,
                category=category,
                difficulty=difficulty,
                metadata={
                    "topic": topic,
                    "template": template,
                },
            )
            
            queries.append(eval_query)
            query_id += 1
        
        # Save eval set
        self._save_eval_set(queries, output_dir)
        
        self.logger.info(f"Created eval set with {len(queries)} queries")
        self.logger.info(f"Categories: {self._get_category_counts(queries)}")
        self.logger.info(f"Difficulty: {self._get_difficulty_counts(queries)}")
        
        return queries
    
    def _save_eval_set(self, queries: list[EvalQuery], output_dir: str) -> None:
        """Save eval set to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save queries
        queries_file = output_path / "eval_queries.jsonl"
        with open(queries_file, "w") as f:
            for query in queries:
                f.write(json.dumps(query.to_dict()) + "\n")
        
        # Save summary stats
        stats = {
            "num_queries": len(queries),
            "categories": self._get_category_counts(queries),
            "difficulty": self._get_difficulty_counts(queries),
            "avg_relevant_docs": sum(len(q.relevant_doc_ids) for q in queries) / len(queries),
            "avg_relevant_chunks": sum(len(q.relevant_chunk_ids) for q in queries) / len(queries),
        }
        
        stats_file = output_path / "eval_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Saved eval set to {output_path}")
    
    def _get_category_counts(self, queries: list[EvalQuery]) -> dict[str, int]:
        """Get counts by category."""
        counts = {}
        for q in queries:
            counts[q.category] = counts.get(q.category, 0) + 1
        return counts
    
    def _get_difficulty_counts(self, queries: list[EvalQuery]) -> dict[str, int]:
        """Get counts by difficulty."""
        counts = {}
        for q in queries:
            counts[q.difficulty] = counts.get(q.difficulty, 0) + 1
        return counts
    
    def load_eval_set(self, input_dir: str = "data/eval") -> list[EvalQuery]:
        """Load eval set from disk."""
        input_path = Path(input_dir)
        queries_file = input_path / "eval_queries.jsonl"
        
        if not queries_file.exists():
            raise FileNotFoundError(f"No eval set found at {input_path}")
        
        self.queries = []
        with open(queries_file, "r") as f:
            for line in f:
                data = json.loads(line)
                self.queries.append(EvalQuery(**data))
        
        self.logger.info(f"Loaded eval set with {len(self.queries)} queries")
        return self.queries
    
    def validate_eval_set(self) -> dict:
        """Validate eval set quality."""
        if not self.queries:
            self.load_eval_set()
        
        issues = []
        
        for query in self.queries:
            # Check for empty fields
            if not query.query.strip():
                issues.append(f"{query.query_id}: Empty query")
            if not query.relevant_doc_ids:
                issues.append(f"{query.query_id}: No relevant docs")
            if not query.relevant_chunk_ids:
                issues.append(f"{query.query_id}: No relevant chunks")
            
            # Check query length
            if len(query.query.split()) < 3:
                issues.append(f"{query.query_id}: Query too short")
            if len(query.query.split()) > 30:
                issues.append(f"{query.query_id}: Query too long")
        
        return {
            "num_queries": len(self.queries),
            "num_issues": len(issues),
            "issues": issues[:10],  # Show first 10 issues
        }


def create_eval_set(
    num_queries: int = 50,
    output_dir: str = "data/eval",
) -> list[EvalQuery]:
    """Create evaluation set from existing documents and chunks."""
    manager = EvalSetManager()
    
    # Load documents and chunks
    documents = load_documents("data/raw")
    chunks = load_chunks("data/processed", strategy="fixed_size")
    
    # Create eval set
    return manager.create_eval_set(documents, chunks, num_queries, output_dir)


if __name__ == "__main__":
    # Create eval set
    queries = create_eval_set(num_queries=50)
    print(f"Created {len(queries)} evaluation queries")
    
    # Show sample
    if queries:
        print(f"\nSample query:")
        print(f"  Query: {queries[0].query}")
        print(f"  Relevant docs: {queries[0].relevant_doc_ids}")
        print(f"  Category: {queries[0].category}")
        print(f"  Difficulty: {queries[0].difficulty}")
