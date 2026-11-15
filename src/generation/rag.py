"""
RAG generation with retrieval context.
Retrieves relevant chunks and generates answers with citations.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.utils.config import get_config
from src.utils.logging import setup_logging


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    query: str
    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


class RAGGenerator:
    """RAG generator that uses retrieved chunks to answer queries."""
    
    #: Base URL for OpenAI-compatible providers that are not OpenAI itself.
    DEFAULT_BASE_URLS = {"groq": "https://api.groq.com/openai/v1"}

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.config = get_config()
        self.logger = setup_logging()
        
        self.provider = (provider or self.config.generation.provider).lower()
        self.model = model or self.config.generation.model
        self.base_url = base_url or getattr(self.config.generation, "base_url", None)
        self.client = None
    
    def _init_client(self) -> None:
        """Initialize the provider client.

        Groq exposes an OpenAI-compatible chat API, so it is served by the
        ``openai`` client with a different base URL and API key.
        """
        import os
        
        if self.provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=api_key)
            return

        if self.provider in ("openai", "groq"):
            import openai

            if self.provider == "groq":
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set")
                base_url = self.base_url or self.DEFAULT_BASE_URLS["groq"]
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                base_url = self.base_url

            self.client = (
                openai.OpenAI(api_key=api_key, base_url=base_url)
                if base_url
                else openai.OpenAI(api_key=api_key)
            )
            return

        raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            citation = chunk.get("citation", {})
            source = citation.get("url", "Unknown source")
            title = citation.get("title", "Unknown")
            
            context_parts.append(
                f"[{i}] {title}\n"
                f"Source: {source}\n"
                f"Content: {chunk['content']}\n"
            )
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the RAG prompt."""
        return f"""You are a helpful assistant that answers questions about Stripe's API and documentation.

Based on the following retrieved context, answer the user's question. 

Important rules:
1. Only use information from the provided context
2. Cite your sources using the [number] format (e.g., [1], [2])
3. If the context doesn't contain enough information, say so clearly
4. Be concise but thorough
5. If there are multiple relevant pieces of information, synthesize them

Context:
{context}

Question: {query}

Answer:"""
    
    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        max_tokens: Optional[int] = None,
    ) -> RAGResponse:
        """
        Generate answer from retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved chunks with citations
            max_tokens: Maximum tokens for generation
        
        Returns:
            RAGResponse with answer and citations
        """
        if self.client is None:
            self._init_client()
        
        if max_tokens is None:
            max_tokens = self.config.generation.max_tokens
        
        # Format context
        context = self._format_context(retrieved_chunks)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Generate response
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.config.generation.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text
            usage = getattr(response, "usage", None)
        else:
            # OpenAI-compatible providers (OpenAI, Groq).
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.config.generation.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content
            usage = getattr(response, "usage", None)
        
        # Extract citations
        citations = []
        for chunk in retrieved_chunks:
            citation = chunk.get("citation", {})
            if citation:
                citations.append(citation)
        
        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            metadata={
                "provider": self.provider,
                "model": self.model,
                "num_chunks": len(retrieved_chunks),
                "token_usage": (
                    usage.model_dump() if hasattr(usage, "model_dump") else None
                ),
            },
        )
    
    def batch_generate(
        self,
        queries: list[str],
        batch_chunks: list[list[dict]],
        max_tokens: Optional[int] = None,
    ) -> list[RAGResponse]:
        """Generate answers for multiple queries."""
        return [
            self.generate(query, chunks, max_tokens)
            for query, chunks in zip(queries, batch_chunks)
        ]


class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation."""
    
    def __init__(
        self,
        retriever=None,
        generator: Optional[RAGGenerator] = None,
    ):
        self.config = get_config()
        self.logger = setup_logging()
        
        self.retriever = retriever
        self.generator = generator or RAGGenerator()
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = False,
    ) -> RAGResponse:
        """
        End-to-end RAG query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            use_reranking: Whether to apply reranking
        
        Returns:
            RAGResponse with answer and citations
        """
        # Retrieve
        if self.retriever is None:
            raise RuntimeError("No retriever configured")
        
        retrieved = self.retriever.retrieve(query, top_k=top_k * 2 if use_reranking else top_k)
        
        # Optional reranking
        if use_reranking:
            from src.reranking.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            retrieved = reranker.rerank(query, retrieved, top_k=top_k)
        
        # Generate
        response = self.generator.generate(query, retrieved)
        
        return response
    
    def save_response(self, response: RAGResponse, output_dir: str = "results") -> Path:
        """Save response to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename from query hash
        import hashlib
        query_hash = hashlib.md5(response.query.encode()).hexdigest()[:8]
        filepath = output_path / f"rag_response_{query_hash}.json"
        
        with open(filepath, "w") as f:
            json.dump(response.to_dict(), f, indent=2)
        
        return filepath


def get_rag_generator(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> RAGGenerator:
    """Get a RAG generator instance."""
    return RAGGenerator(provider, model)
