"""
End-to-end evaluation for RAG.
Measures faithfulness and answer relevance using LLM-as-judge.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.generation.rag import RAGResponse
from src.utils.config import get_config
from src.utils.logging import setup_logging


@dataclass
class E2EEvalResult:
    """Result of end-to-end evaluation."""
    query_id: str
    query: str
    answer: str
    faithfulness_score: float
    relevance_score: float
    overall_score: float
    faithfulness_reasoning: str
    relevance_reasoning: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class E2EEvaluator:
    """End-to-end evaluator for RAG responses."""
    
    # Scoring rubric for faithfulness
    FAITHFULNESS_RUBRIC = """You are evaluating whether an answer is faithful to the provided context.

Score the answer on a scale of 1-5:
1: Answer completely contradicts the context or makes up information
2: Answer partially contradicts or contains significant unsupported claims
3: Answer is mostly faithful but has minor unsupported claims
4: Answer is faithful with only minor omissions
5: Answer is completely faithful to the context

Context chunks used:
{context}

Question: {question}

Answer to evaluate: {answer}

Provide your evaluation as JSON:
{{
    "score": <1-5>,
    "reasoning": "<brief explanation>"
}}"""
    
    # Scoring rubric for relevance
    RELEVANCE_RUBRIC = """You are evaluating whether an answer is relevant and helpful for the question.

Score the answer on a scale of 1-5:
1: Answer is completely irrelevant to the question
2: Answer is mostly irrelevant with minor relevant elements
3: Answer is partially relevant but misses key points
4: Answer is relevant and addresses the main question
5: Answer is highly relevant, comprehensive, and helpful

Question: {question}

Answer to evaluate: {answer}

Provide your evaluation as JSON:
{{
    "score": <1-5>,
    "reasoning": "<brief explanation>"
}}"""
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.config = get_config()
        self.logger = setup_logging()
        
        self.provider = provider or self.config.generation.provider
        self.model = model or self.config.generation.model
        self.client = None
    
    def _init_client(self) -> None:
        """Initialize API client for LLM-as-judge."""
        import os
        
        if self.provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "openai":
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = openai.OpenAI(api_key=api_key)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM for scoring."""
        if self.client is None:
            self._init_client()
        
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
    
    def _parse_score(self, response: str) -> dict:
        """Parse score from LLM response."""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract score manually
        import re
        score_match = re.search(r'"score":\s*(\d)', response)
        if score_match:
            return {"score": int(score_match.group(1)), "reasoning": response}
        
        return {"score": 3, "reasoning": "Could not parse score"}
    
    def evaluate_faithfulness(
        self,
        response: RAGResponse,
    ) -> tuple[float, str]:
        """
        Evaluate faithfulness of the answer to the context.
        
        Returns:
            Tuple of (score, reasoning)
        """
        # Format context
        context = "\n\n".join([
            f"Chunk {i+1}: {chunk['content'][:500]}"
            for i, chunk in enumerate(response.retrieved_chunks[:5])
        ])
        
        prompt = self.FAITHFULNESS_RUBRIC.format(
            context=context,
            question=response.query,
            answer=response.answer,
        )
        
        llm_response = self._call_llm(prompt)
        result = self._parse_score(llm_response)
        
        score = result.get("score", 3) / 5.0  # Normalize to [0, 1]
        reasoning = result.get("reasoning", "No reasoning provided")
        
        return score, reasoning
    
    def evaluate_relevance(
        self,
        response: RAGResponse,
    ) -> tuple[float, str]:
        """
        Evaluate relevance of the answer to the question.
        
        Returns:
            Tuple of (score, reasoning)
        """
        prompt = self.RELEVANCE_RUBRIC.format(
            question=response.query,
            answer=response.answer,
        )
        
        llm_response = self._call_llm(prompt)
        result = self._parse_score(llm_response)
        
        score = result.get("score", 3) / 5.0  # Normalize to [0, 1]
        reasoning = result.get("reasoning", "No reasoning provided")
        
        return score, reasoning
    
    def evaluate_response(
        self,
        response: RAGResponse,
        query_id: str = "unknown",
    ) -> E2EEvalResult:
        """
        Evaluate a single RAG response.
        
        Returns:
            E2EEvalResult with scores and reasoning
        """
        faithfulness_score, faithfulness_reasoning = self.evaluate_faithfulness(response)
        relevance_score, relevance_reasoning = self.evaluate_relevance(response)
        
        overall_score = (faithfulness_score + relevance_score) / 2
        
        return E2EEvalResult(
            query_id=query_id,
            query=response.query,
            answer=response.answer[:500],  # Truncate for storage
            faithfulness_score=faithfulness_score,
            relevance_score=relevance_score,
            overall_score=overall_score,
            faithfulness_reasoning=faithfulness_reasoning,
            relevance_reasoning=relevance_reasoning,
        )
    
    def evaluate_batch(
        self,
        responses: list[RAGResponse],
        query_ids: Optional[list[str]] = None,
    ) -> list[E2EEvalResult]:
        """
        Evaluate multiple RAG responses.
        
        Returns:
            List of E2EEvalResult
        """
        if query_ids is None:
            query_ids = [f"q{i:03d}" for i in range(len(responses))]
        
        results = []
        for response, query_id in zip(responses, query_ids):
            self.logger.info(f"Evaluating {query_id}...")
            result = self.evaluate_response(response, query_id)
            results.append(result)
        
        return results
    
    def calculate_aggregate_metrics(
        self,
        results: list[E2EEvalResult],
    ) -> dict:
        """
        Calculate aggregate metrics across all evaluations.
        
        Returns:
            Dictionary with aggregate metrics
        """
        faithfulness_scores = [r.faithfulness_score for r in results]
        relevance_scores = [r.relevance_score for r in results]
        overall_scores = [r.overall_score for r in results]
        
        return {
            "num_evaluations": len(results),
            "faithfulness": {
                "mean": sum(faithfulness_scores) / len(faithfulness_scores),
                "min": min(faithfulness_scores),
                "max": max(faithfulness_scores),
            },
            "relevance": {
                "mean": sum(relevance_scores) / len(relevance_scores),
                "min": min(relevance_scores),
                "max": max(relevance_scores),
            },
            "overall": {
                "mean": sum(overall_scores) / len(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores),
            },
        }
    
    def save_results(
        self,
        results: list[E2EEvalResult],
        output_dir: str = "results",
    ) -> Path:
        """Save evaluation results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        results_file = output_path / "e2e_eval_results.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + "\n")
        
        # Save aggregate metrics
        aggregate = self.calculate_aggregate_metrics(results)
        aggregate_file = output_path / "e2e_eval_aggregate.json"
        with open(aggregate_file, "w") as f:
            json.dump(aggregate, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {output_path}")
        return output_path


def evaluate_rag_responses(
    responses: list[RAGResponse],
    query_ids: Optional[list[str]] = None,
    output_dir: str = "results",
) -> dict:
    """
    Evaluate RAG responses end-to-end.
    
    Returns:
        Dictionary with individual and aggregate results
    """
    evaluator = E2EEvaluator()
    results = evaluator.evaluate_batch(responses, query_ids)
    aggregate = evaluator.calculate_aggregate_metrics(results)
    evaluator.save_results(results, output_dir)
    
    return {
        "individual_results": [r.to_dict() for r in results],
        "aggregate": aggregate,
    }
