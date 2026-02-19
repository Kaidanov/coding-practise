"""
Evaluation Framework for RAG Systems.

This module implements comprehensive evaluation metrics:
- Faithfulness/Groundedness: Is the answer supported by the context?
- Answer Relevance: Does the answer address the question?
- Context Relevance: Are retrieved documents relevant?
- Factual Correctness: Comparison with ground truth (when available)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import os
from openai import OpenAI
import numpy as np


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    faithfulness: float  # 0-1: Is answer grounded in context?
    answer_relevance: float  # 0-1: Does answer address the question?
    context_relevance: float  # 0-1: Are retrieved docs relevant?
    factual_correctness: Optional[float] = None  # 0-1: Match with ground truth
    overall_score: float = 0.0
    
    def __post_init__(self):
        """Calculate overall score as weighted average."""
        scores = [self.faithfulness, self.answer_relevance, self.context_relevance]
        if self.factual_correctness is not None:
            scores.append(self.factual_correctness)
        self.overall_score = np.mean(scores)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single query."""
    query: str
    answer: str
    context: List[str]
    ground_truth: Optional[str]
    metrics: EvaluationMetrics
    detailed_feedback: Dict[str, str]


class RAGEvaluator:
    """
    Evaluates RAG system responses using LLM-based evaluation.
    
    This approach uses an LLM as a judge to assess various quality dimensions.
    For production, consider also implementing:
    - Human evaluation
    - Automated test suites
    - A/B testing
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4"
    ):
        """
        Initialize the evaluator.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use for evaluation
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a RAG response across multiple dimensions.
        
        Args:
            query: The user's question
            answer: The generated answer
            context: List of context documents used
            ground_truth: Optional reference answer for comparison
        
        Returns:
            EvaluationResult with metrics and feedback
        """
        # Evaluate faithfulness (groundedness)
        faithfulness, faith_feedback = self._evaluate_faithfulness(answer, context)
        
        # Evaluate answer relevance
        relevance, rel_feedback = self._evaluate_answer_relevance(query, answer)
        
        # Evaluate context relevance
        ctx_relevance, ctx_feedback = self._evaluate_context_relevance(query, context)
        
        # Evaluate factual correctness if ground truth provided
        correctness = None
        correct_feedback = "No ground truth provided"
        if ground_truth:
            correctness, correct_feedback = self._evaluate_factual_correctness(
                answer, ground_truth
            )
        
        metrics = EvaluationMetrics(
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_relevance=ctx_relevance,
            factual_correctness=correctness
        )
        
        detailed_feedback = {
            "faithfulness": faith_feedback,
            "answer_relevance": rel_feedback,
            "context_relevance": ctx_feedback,
            "factual_correctness": correct_feedback
        }
        
        return EvaluationResult(
            query=query,
            answer=answer,
            context=context,
            ground_truth=ground_truth,
            metrics=metrics,
            detailed_feedback=detailed_feedback
        )
    
    def _evaluate_faithfulness(
        self, answer: str, context: List[str]
    ) -> tuple[float, str]:
        """
        Evaluate if the answer is faithful/grounded in the context.
        
        Key question: Can all claims in the answer be verified from the context?
        """
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""Evaluate the FAITHFULNESS of the answer based on the provided context.

Context:
{context_str}

Answer:
{answer}

Task: Determine if every claim in the answer is directly supported by the context.

Provide:
1. SCORE: A number from 0 to 1 where:
   - 1.0 = All claims fully supported by context
   - 0.5 = Some claims supported, some not verifiable
   - 0.0 = Claims contradict or have no support in context

2. REASONING: Explain which claims are supported and which aren't

Format:
SCORE: [0.0-1.0]
REASONING: [Your explanation]"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        score, reasoning = self._parse_evaluation_response(content)
        
        return score, reasoning
    
    def _evaluate_answer_relevance(
        self, query: str, answer: str
    ) -> tuple[float, str]:
        """
        Evaluate if the answer is relevant to the question.
        
        Key question: Does the answer actually address what was asked?
        """
        prompt = f"""Evaluate the RELEVANCE of the answer to the question.

Question:
{query}

Answer:
{answer}

Task: Determine if the answer directly addresses the question asked.

Provide:
1. SCORE: A number from 0 to 1 where:
   - 1.0 = Answer directly and completely addresses the question
   - 0.5 = Answer partially addresses the question
   - 0.0 = Answer is off-topic or doesn't address the question

2. REASONING: Explain how well the answer addresses the question

Format:
SCORE: [0.0-1.0]
REASONING: [Your explanation]"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        score, reasoning = self._parse_evaluation_response(content)
        
        return score, reasoning
    
    def _evaluate_context_relevance(
        self, query: str, context: List[str]
    ) -> tuple[float, str]:
        """
        Evaluate if the retrieved context is relevant to the question.
        
        Key question: Did we retrieve the right documents?
        """
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""Evaluate the RELEVANCE of the retrieved context to the question.

Question:
{query}

Retrieved Context:
{context_str}

Task: Determine if the context contains information relevant to answering the question.

Provide:
1. SCORE: A number from 0 to 1 where:
   - 1.0 = Context highly relevant and contains answer
   - 0.5 = Context somewhat relevant but incomplete
   - 0.0 = Context not relevant to the question

2. REASONING: Explain which parts of context are relevant or not

Format:
SCORE: [0.0-1.0]
REASONING: [Your explanation]"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        score, reasoning = self._parse_evaluation_response(content)
        
        return score, reasoning
    
    def _evaluate_factual_correctness(
        self, answer: str, ground_truth: str
    ) -> tuple[float, str]:
        """
        Evaluate factual correctness against ground truth.
        
        Key question: How well does the answer match the reference answer?
        """
        prompt = f"""Evaluate the FACTUAL CORRECTNESS of the answer against the ground truth.

Ground Truth (Reference Answer):
{ground_truth}

Generated Answer:
{answer}

Task: Determine if the generated answer is factually consistent with the ground truth.

Provide:
1. SCORE: A number from 0 to 1 where:
   - 1.0 = Factually identical or semantically equivalent
   - 0.5 = Partially correct with some inaccuracies
   - 0.0 = Factually incorrect or contradictory

2. REASONING: Explain factual matches and discrepancies

Format:
SCORE: [0.0-1.0]
REASONING: [Your explanation]"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        score, reasoning = self._parse_evaluation_response(content)
        
        return score, reasoning
    
    def _parse_evaluation_response(self, content: str) -> tuple[float, str]:
        """Parse the evaluation response to extract score and reasoning."""
        lines = content.strip().split('\n')
        score = 0.5  # Default
        reasoning = ""
        
        for i, line in enumerate(lines):
            if line.startswith("SCORE:"):
                score_str = line.replace("SCORE:", "").strip()
                try:
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = "\n".join(lines[i:]).replace("REASONING:", "").strip()
                break
        
        return score, reasoning


def evaluate_batch(
    evaluator: RAGEvaluator,
    test_cases: List[Dict[str, Any]]
) -> List[EvaluationResult]:
    """
    Evaluate multiple test cases in batch.
    
    Args:
        evaluator: RAGEvaluator instance
        test_cases: List of dicts with keys: query, answer, context, ground_truth
    
    Returns:
        List of EvaluationResult objects
    """
    results = []
    for case in test_cases:
        result = evaluator.evaluate(
            query=case["query"],
            answer=case["answer"],
            context=case.get("context", []),
            ground_truth=case.get("ground_truth")
        )
        results.append(result)
    return results


def summarize_evaluations(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Generate summary statistics from evaluation results.
    
    Args:
        results: List of EvaluationResult objects
    
    Returns:
        Dictionary with mean, min, max for each metric
    """
    if not results:
        return {}
    
    faithfulness_scores = [r.metrics.faithfulness for r in results]
    relevance_scores = [r.metrics.answer_relevance for r in results]
    context_scores = [r.metrics.context_relevance for r in results]
    overall_scores = [r.metrics.overall_score for r in results]
    
    correctness_scores = [
        r.metrics.factual_correctness for r in results
        if r.metrics.factual_correctness is not None
    ]
    
    summary = {
        "total_cases": len(results),
        "faithfulness": {
            "mean": np.mean(faithfulness_scores),
            "min": np.min(faithfulness_scores),
            "max": np.max(faithfulness_scores)
        },
        "answer_relevance": {
            "mean": np.mean(relevance_scores),
            "min": np.min(relevance_scores),
            "max": np.max(relevance_scores)
        },
        "context_relevance": {
            "mean": np.mean(context_scores),
            "min": np.min(context_scores),
            "max": np.max(context_scores)
        },
        "overall_score": {
            "mean": np.mean(overall_scores),
            "min": np.min(overall_scores),
            "max": np.max(overall_scores)
        }
    }
    
    if correctness_scores:
        summary["factual_correctness"] = {
            "mean": np.mean(correctness_scores),
            "min": np.min(correctness_scores),
            "max": np.max(correctness_scores)
        }
    
    return summary
