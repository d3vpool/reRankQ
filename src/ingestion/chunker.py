"""
Chunking strategies for document processing.
Implements fixed-size with overlap and structure-aware chunking.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import tiktoken

from src.ingestion.crawler import Document
from src.utils.config import get_config
from src.utils.logging import setup_logging


@dataclass
class Chunk:
    """Represents a chunk of text."""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    strategy: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


class TextChunker:
    """Base class for text chunking strategies."""
    
    def __init__(self, strategy_name: str):
        self.config = get_config()
        self.logger = setup_logging()
        self.strategy_name = strategy_name
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int, strategy: str) -> str:
        """Generate a stable chunk ID."""
        content = f"{doc_id}_{chunk_index}_{strategy}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Chunk a document. To be implemented by subclasses."""
        raise NotImplementedError


class FixedSizeChunker(TextChunker):
    """Fixed-size chunking with overlap."""
    
    def __init__(self):
        super().__init__("fixed_size")
        self.chunk_size = self.config.chunking.chunk_size
        self.chunk_overlap = self.config.chunking.chunk_overlap
    
    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split document into fixed-size chunks with overlap."""
        chunks = []
        text = doc.content
        text_length = len(text)
        
        if text_length == 0:
            return chunks
        
        # Calculate character positions for token-based chunking
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Find end position (approximate token limit)
            end = start + self.chunk_size * 4  # Rough estimate: 4 chars per token
            
            # If not at the end, try to find a natural break point
            if end < text_length:
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break > start + self.chunk_size // 2:
                        end = sentence_break + 2
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(doc.doc_id, chunk_index, self.strategy_name),
                    doc_id=doc.doc_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    strategy=self.strategy_name,
                    start_char=start,
                    end_char=end,
                    token_count=self.count_tokens(chunk_text),
                    metadata={
                        "section": doc.section,
                        "subsection": doc.subsection,
                        "url": doc.url,
                        "title": doc.title,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap * 4
            if start <= end - 4:  # Prevent infinite loop
                start = end
        
        return chunks


class StructureAwareChunker(TextChunker):
    """Structure-aware chunking that respects document headers."""
    
    def __init__(self):
        super().__init__("structure_aware")
        self.max_chunk_size = self.config.chunking.max_chunk_size if hasattr(self.config.chunking, 'max_chunk_size') else 512
        self.min_chunk_size = self.config.chunking.min_chunk_size if hasattr(self.config.chunking, 'min_chunk_size') else 100
    
    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        """Split text by markdown headers, returning (header, content) pairs."""
        # Match markdown headers
        header_pattern = re.compile(r'^(#{1,4}\s+.+)$', re.MULTILINE)
        
        parts = []
        last_end = 0
        current_header = "Introduction"
        
        for match in header_pattern.finditer(text):
            # Add content before this header
            content_before = text[last_end:match.start()].strip()
            if content_before:
                parts.append((current_header, content_before))
            
            current_header = match.group(1).strip()
            last_end = match.end()
        
        # Add remaining content
        remaining = text[last_end:].strip()
        if remaining:
            parts.append((current_header, remaining))
        
        return parts
    
    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_small_chunks(self, chunks: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Merge small chunks that are below minimum size."""
        if not chunks:
            return chunks
        
        merged = []
        current_header, current_content = chunks[0]
        
        for header, content in chunks[1:]:
            if self.count_tokens(current_content) < self.min_chunk_size:
                # Merge with current
                current_content = current_content + "\n\n" + content
            else:
                # Save current and start new
                merged.append((current_header, current_content))
                current_header = header
                current_content = content
        
        # Don't forget the last one
        if current_content:
            merged.append((current_header, current_content))
        
        return merged
    
    def _split_large_chunk(self, header: str, content: str) -> list[tuple[str, str]]:
        """Split a chunk that's too large."""
        if self.count_tokens(content) <= self.max_chunk_size:
            return [(header, content)]
        
        # Split by paragraphs first
        paragraphs = content.split("\n\n")
        result = []
        current = ""
        
        for para in paragraphs:
            if self.count_tokens(current + "\n\n" + para) > self.max_chunk_size:
                if current:
                    result.append((header, current.strip()))
                current = para
            else:
                current = current + "\n\n" + para if current else para
        
        if current:
            result.append((header, current.strip()))
        
        return result
    
    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split document respecting structure."""
        chunks = []
        
        # Split by headers
        sections = self._split_by_headers(doc.content)
        
        # Merge small sections
        sections = self._merge_small_chunks(sections)
        
        # Split large sections
        final_sections = []
        for header, content in sections:
            final_sections.extend(self._split_large_chunk(header, content))
        
        # Create chunks
        for chunk_index, (header, content) in enumerate(final_sections):
            chunk_text = content.strip()
            if not chunk_text:
                continue
            
            chunk = Chunk(
                chunk_id=self._generate_chunk_id(doc.doc_id, chunk_index, self.strategy_name),
                doc_id=doc.doc_id,
                content=chunk_text,
                chunk_index=chunk_index,
                strategy=self.strategy_name,
                start_char=0,  # Structure-aware doesn't track exact positions
                end_char=len(chunk_text),
                token_count=self.count_tokens(chunk_text),
                metadata={
                    "section": doc.section,
                    "subsection": doc.subsection,
                    "header": header,
                    "url": doc.url,
                    "title": doc.title,
                },
            )
            chunks.append(chunk)
        
        return chunks


def chunk_documents(
    documents: list[Document],
    strategy: str = "both",
    output_dir: str = "data/processed"
) -> dict[str, list[Chunk]]:
    """
    Chunk documents using specified strategy.
    
    Args:
        documents: List of documents to chunk
        strategy: "fixed_size", "structure_aware", or "both"
        output_dir: Directory to save chunked documents
    
    Returns:
        Dictionary mapping strategy name to list of chunks
    """
    logger = setup_logging()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if strategy in ["fixed_size", "both"]:
        logger.info("Chunking with fixed-size strategy...")
        fixed_chunker = FixedSizeChunker()
        fixed_chunks = []
        for doc in documents:
            chunks = fixed_chunker.chunk_document(doc)
            fixed_chunks.extend(chunks)
        results["fixed_size"] = fixed_chunks
        logger.info(f"Fixed-size: {len(fixed_chunks)} chunks from {len(documents)} documents")
    
    if strategy in ["structure_aware", "both"]:
        logger.info("Chunking with structure-aware strategy...")
        structure_chunker = StructureAwareChunker()
        structure_chunks = []
        for doc in documents:
            chunks = structure_chunker.chunk_document(doc)
            structure_chunks.extend(chunks)
        results["structure_aware"] = structure_chunks
        logger.info(f"Structure-aware: {len(structure_chunks)} chunks from {len(documents)} documents")
    
    # Save chunks
    for strat_name, chunks in results.items():
        chunks_file = output_path / f"chunks_{strat_name}.jsonl"
        with open(chunks_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
        
        # Save chunk stats
        stats = {
            "strategy": strat_name,
            "num_chunks": len(chunks),
            "avg_tokens": sum(c.token_count for c in chunks) / max(len(chunks), 1),
            "min_tokens": min(c.token_count for c in chunks) if chunks else 0,
            "max_tokens": max(c.token_count for c in chunks) if chunks else 0,
            "token_distribution": {
                "0-100": len([c for c in chunks if c.token_count < 100]),
                "100-200": len([c for c in chunks if 100 <= c.token_count < 200]),
                "200-300": len([c for c in chunks if 200 <= c.token_count < 300]),
                "300-400": len([c for c in chunks if 300 <= c.token_count < 400]),
                "400-500": len([c for c in chunks if 400 <= c.token_count < 500]),
                "500+": len([c for c in chunks if c.token_count >= 500]),
            },
        }
        
        stats_file = output_path / f"stats_{strat_name}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
    
    return results


def load_chunks(input_dir: str = "data/processed", strategy: str = "fixed_size") -> list[Chunk]:
    """Load chunks from disk."""
    input_path = Path(input_dir)
    chunks_file = input_path / f"chunks_{strategy}.jsonl"
    
    if not chunks_file.exists():
        raise FileNotFoundError(f"No chunks found at {chunks_file}")
    
    chunks = []
    with open(chunks_file, "r") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))
    
    return chunks


if __name__ == "__main__":
    # Test chunking
    from src.ingestion.crawler import load_documents
    
    documents = load_documents("data/raw")
    print(f"Loaded {len(documents)} documents")
    
    results = chunk_documents(documents, strategy="both")
    for strat, chunks in results.items():
        print(f"{strat}: {len(chunks)} chunks")
